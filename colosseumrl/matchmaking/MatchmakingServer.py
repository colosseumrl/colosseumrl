import time
import secrets
import grpc
import argparse
import zmq

from queue import Queue, Empty
from concurrent import futures
from threading import Thread, Semaphore
from multiprocessing import Event
from typing import Type, Dict, List
from collections import OrderedDict
from spacetime import Node

from ..match_server import server_app
from ..data_model import ServerState, Player, Observation
from ..config import get_environment, available_environments
from ..BaseEnvironment import BaseEnvironment
from ..util import is_port_in_use
from ..rl_logging import init_logging, get_logger

from .grpc_gen.server_pb2 import QuickMatchReply, QuickMatchRequest
from .grpc_gen.server_pb2_grpc import MatchmakerServicer, add_MatchmakerServicer_to_server
from .RankingDatabase import RankingDatabase


logger = get_logger()

# Global ZMQ context that will be used for all communication between threads.
zmq_context = zmq.Context()


def match_server_args_factory(tick_rate: int, realtime: bool, observations_only: bool, env_config_string: str):
    """ Helper factory to make a argument dictionary for servers with varying ports """

    def match_server_args(port):
        arg_dict = {
            "tick_rate": tick_rate,
            "port": port,
            "realtime": realtime,
            "observations_only": observations_only,
            "config": env_config_string
        }
        return arg_dict

    return match_server_args


class MatchMakingHandler(MatchmakerServicer):
    """ GRPC connection handler.

        Clients will connect to the server and call this function to request a match. """

    def GetMatch(self, request, context):
        # Unique identity for this connection
        identity = request.username.encode() + secrets.token_bytes(8)

        # Prepare ZeroMQ connection to get added to the queue
        with zmq_context.socket(zmq.REQ) as socket:
            socket.connect("inproc://matchmaker_requests")

            # Wait until we are added to the queue
            socket.send_multipart((identity, request.SerializeToString()))
            status, response = socket.recv_multipart()
            if status == b"FAIL":
                return QuickMatchReply.FromString(response)

        # Setup new socket to communicate with matchmaking master
        with zmq_context.socket(zmq.DEALER) as socket:
            socket.setsockopt(zmq.IDENTITY, identity)
            socket.connect("inproc://matchmaker_responses")

            # Wait until a game has been assigned
            # Check every once in a while to see if the client is still alive
            while True:
                if not socket.poll(timeout=500):
                    if not context.is_active():
                        socket.send(request.username.encode())
                        return None
                else:
                    break

            response = QuickMatchReply.FromString(socket.recv(flags=zmq.NOBLOCK))
            return response


class MatchProcessJanitor(Thread):
    """ Simple thread to manage the lifetime of a game server. Will start the game server and
        close it when the game is finished and release any resources it was holding. """

    def __init__(self,
                 match_limit: Semaphore,
                 ports_to_use_queue: Queue,
                 database: RankingDatabase,
                 env_class: Type[BaseEnvironment],
                 match_server_args: Dict,
                 player_list: List,
                 whitelist: List = None):
        """ Create a janitor thread that will start a game and close it once it is finished.

        Parameters
        ----------
        match_limit : Semaphore
            Synchronization semaphore to make sure that we limit the number of simulations matches.
        ports_to_use_queue : Queue
            Queue holding available ports for this server.
        database : RankingDatabase
            Global database for player ranking.
        env_class : Type[BaseEnvironment]
            The game class to launch the server around.
        match_server_args : Dict
            Any arguments to pass to the match server as keyword star arguments.
        player_list : List[str]
            A list of usernames participating in this game. This will be the whitelist for the server.
        whitelist : List
            A list of usernames participating in this game. This will be the whitelist for the server. Again...
        """
        super().__init__()
        self.match_limit = match_limit
        self.match_server_args = match_server_args
        self.env_class = env_class
        self.ports_to_use_queue = ports_to_use_queue
        self.database = database
        self.player_list = player_list
        self.whitelist = whitelist
        self.ready = Event()

    def run(self) -> None:
        port = self.match_server_args['port']
        observation_type = Observation(self.env_class.observation_names())

        # App blocks until the server has ended
        app = Node(server_app, server_port=port, Types=[Player, ServerState])
        rankings = app.start(self.env_class, observation_type, self.match_server_args, self.whitelist, self.ready)
        del app

        # Update player information
        if isinstance(rankings, dict):
            self.database.update_ranking(rankings)

        for user in self.player_list:
            self.database.logoff(user)

        # Cleanup
        self.ports_to_use_queue.put(port)
        self.match_limit.release()


class MatchmakingLoginThread(Thread):
    def __init__(self, connection_queue: Queue, database: RankingDatabase):
        """ Thread for managing the login system on the matchmaking server.

        This thread will accept login requests from the GRPC function and add them to the queue of players for
        a new game.

        Parameters
        ----------
        connection_queue: Queue
            The queue to add new players to after they have been successfully logged in.
        database : RankingDatabase
            Global database for player ranking and password store.
        """
        super().__init__()

        self.queue: Queue = connection_queue
        self.database: RankingDatabase = database
        self.daemon = True

        self.socket = zmq_context.socket(zmq.REP)
        self.socket.bind("inproc://matchmaker_requests")
        print("Matchmaker Connector thread listening...")

    def __del__(self):
        self.socket.close()

    def run(self):
        while True:
            identity, serialized_request = self.socket.recv_multipart()
            request = QuickMatchRequest.FromString(serialized_request)

            # Login user and handle any errors
            username, password = request.username, request.password
            login_result = self.database.login(username, password)

            if login_result == RankingDatabase.LoginResult.NoUser:
                self.database.set(username, password)
                self.database.login(username, password)

            elif login_result == RankingDatabase.LoginResult.LoginDuplicate:
                response = QuickMatchReply(username=username, server="FAIL", auth_key="FAIL", ranking=0.0,
                                           response="Failed to login: Cannot login twice at the same time.")
                self.socket.send_multipart((b"FAIL", response.SerializeToString()))
                continue

            elif login_result == RankingDatabase.LoginResult.LoginFail:
                response = QuickMatchReply(username=username, server="FAIL", auth_key="FAIL", ranking=0.0,
                                           response="Failed to login: Wrong password.")
                self.socket.send_multipart((b"FAIL", response.SerializeToString()))
                continue

            # Add request to the queue and generate a token for them
            self.queue.put((identity, request, secrets.token_hex(32)))
            self.socket.send_multipart((b"SUCCESS", b""))


class MatchmakingThread(Thread):
    def __init__(self,
                 starting_port,
                 hostname,
                 max_simultaneous_games,
                 env_class,
                 tick_rate,
                 realtime,
                 observations_only,
                 env_config_string):
        """ Main matchmaking thread that is responsible for choosing players for each match
        and assigning a game server to them.

        Parameters
        ----------
        starting_port : int
            Port the begin making match server on
        hostname : str
            What hostname to start the game servers on.
        max_simultaneous_games : int
            Maximum number of game servers that will be running at any given time.
        env_class : Type[BaseEnvironment]
            What environment will the server be running.
        tick_rate : float
            What frame rate will the servers operate on.
        realtime : bool
            Whether or not the games will be realtime or will wait for player actions.
        observations_only : bool
            Whether or not we will send out the true server state if supported.
        env_config_string : str
            Configuration string to be passed to the server environments.
        """
        super().__init__()

        self.players_per_game = env_class(env_config_string).min_players
        self.env_class = env_class
        self.hostname = hostname
        self.daemon = True

        # Prepare our context and sockets
        self.socket = zmq_context.socket(zmq.ROUTER)
        self.socket.bind("inproc://matchmaker_responses")
        logger.info("Matchmaker thread running")

        # Semaphore for tracking the total number of games running
        self.match_limit = Semaphore(max_simultaneous_games)

        # Helper function to make arguments for match threads
        self.create_match_server_args = match_server_args_factory(tick_rate=tick_rate,
                                                                  realtime=realtime,
                                                                  observations_only=observations_only,
                                                                  env_config_string=env_config_string)

        # Keep track of the ports we can use and iterate through them as we start new servers
        self.ports_to_use = Queue()
        max_port = starting_port + 2 * max_simultaneous_games
        for port in range(starting_port, max_port):
            if not is_port_in_use(port):
                self.ports_to_use.put(port)
            else:
                logger.warn("Skipping port {}, already in use.".format(port))

        if self.ports_to_use.qsize() < max_simultaneous_games:
            raise OSError("Port range {} through {} does not have enough unallocated ports "
                          "to hold {} simultaneous games".format(starting_port, max_port, max_simultaneous_games))

        self.database = RankingDatabase("test.sqlite")

        self.connection_queue = Queue()
        self.connection_thread = MatchmakingLoginThread(self.connection_queue, self.database)

    def start(self) -> None:
        """ Start this thread along with the related login thread. """
        super().start()
        self.connection_thread.start()

    def select_players(self, requests):
        """ Select which players will be chosen for an upcoming game.

        At some point, this should use trueskill to create matches with roughly fairly matched opponents.

        Parameters
        ----------
        requests : OrderedDict
            All of the players requested with their username as their key and their request and token as the values.

        Returns
        -------
        List
            A list of players that have been chosen with the appropriate data.

        """
        players = []
        for _ in range(self.players_per_game):
            identity, (request, auth) = requests.popitem(last=False)
            players.append((identity, request, auth))
        return players

    def run(self) -> None:
        requests = OrderedDict()

        while True:
            # Wait for any new requests, and always recheck request queue after 5 seconds
            # This will be useful if we have a robust matchmaking system with rankings
            try:
                identity, request, authorization = self.connection_queue.get(timeout=5.0)
                requests[identity] = (request, authorization)
            except Empty:
                pass
            else:
                while not self.connection_queue.empty():
                    identity, request, authorization = self.connection_queue.get()
                    requests[identity] = (request, authorization)

            # Check if any clients have disconnected
            while True:
                try:
                    quitting_identity, quitting_username = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                else:
                    logger.debug("{} has quit the matchmaking queue unexpectedly.".format(quitting_identity))
                    self.database.logoff(quitting_username.decode())
                    requests.pop(quitting_identity, 0)

            # Once we have enough players for a game, start a game server and send the coordinates
            if len(requests) >= self.players_per_game:
                # Limit the number of games so we dont overload server
                self.match_limit.acquire()

                # Select Players using arbitrary method
                new_players = self.select_players(requests)
                whitelist = [player[2] for player in new_players]
                usernames = [player[1].username for player in new_players]

                # Create the game server
                match_port = self.ports_to_use.get()
                match_server_args = self.create_match_server_args(port=match_port)
                match_janitor = MatchProcessJanitor(match_limit=self.match_limit,
                                                    ports_to_use_queue=self.ports_to_use,
                                                    database=self.database,
                                                    env_class=self.env_class,
                                                    match_server_args=match_server_args,
                                                    player_list=usernames,
                                                    whitelist=whitelist)
                match_janitor.start()

                database_entries = self.database.get_multi(*usernames)
                database_entries = {name: ranking for name, _, ranking, _ in database_entries}
                match_janitor.ready.wait()

                # Send each player their assigned server.
                for identity, request, auth_key in new_players:
                    response = QuickMatchReply(username=request.username,
                                               server='{}:{}'.format(self.hostname, match_port),
                                               auth_key=auth_key,
                                               ranking=database_entries[request.username],
                                               response="")

                    self.socket.send_multipart((identity, response.SerializeToString()))


def serve(args):
    """ Main function for Matchmaking server

    Parameters
    ----------
    args : Dict
        Command line arguments

    """
    # Start the separate matchmaking thread
    matchmaker_thread = MatchmakingThread(
        hostname=args['hostname'],
        starting_port=args['game_port'],
        max_simultaneous_games=args['max_games'],
        env_class=get_environment(args['environment']),
        tick_rate=args['tick_rate'],
        realtime=args['realtime'],
        observations_only=args['observations_only'],
        env_config_string=args['config']
    )
    matchmaker_thread.start()

    # Start the GRPC callback server
    server = grpc.server(futures.ThreadPoolExecutor())
    add_MatchmakerServicer_to_server(MatchMakingHandler(), server)
    server.add_insecure_port('[::]:{}'.format(args['matchmaking_port']))
    server.start()
    logger.info("Matchmaking server listening on grpc://{}:{}...".format(args['hostname'], args['matchmaking_port']))
    try:
        one_day = 3600 * 24
        while True:
            time.sleep(one_day)
    except KeyboardInterrupt:
        server.stop(0)


def start_matchmaking_server(environment: str = 'test',
                             hostname: str = 'localhost',
                             matchmaking_port: int = 50051,
                             game_port: int = 21450,
                             max_games: int = 1,
                             tick_rate: int = 60,
                             realtime: bool = False,
                             observations_only: bool = False,
                             config: str = ''):
    serve(locals())


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""
    This script launches a matchmaking server for the colosseum framework. The purpose of this system
    is to allow for an easy way for players to join a match together without needing to coordinate join time, 
    server information, or limiting players.
    """)
    parser.add_argument("--environment", "-e", type=str, default="test",
                        help="The name of the environment. Choices are: {}".format(available_environments()))
    parser.add_argument("--hostname", type=str, default='localhost',
                        help="Hostname to start the matchmaking and game servers on. Defaults to 'localhost'")
    parser.add_argument("--matchmaking-port", type=int, default=50051,
                        help="Port to start matchmaking server on.")
    parser.add_argument("--game-port", type=int, default=21450,
                        help="Port to start game servers on. Will use a range starting at this port to this port"
                             "plus the number of games.")
    parser.add_argument("--max-games", "-m", type=int, default=1,
                        help="Number of games to run in parallel on this server.")
    parser.add_argument("--tick-rate", "-t", type=int, default=60,
                        help="The max tick rate that the server will run on.")
    parser.add_argument("--realtime", "-r", action="store_true",
                        help="With this flag on, the server will not wait for all of the clients to respond.")
    parser.add_argument("--observations-only", '-f', action='store_true',
                        help="With this flag on, the server will not push the true state of the game to the clients "
                             "along with observations")
    parser.add_argument("--config", '-c', type=str, default="",
                        help="Config string that will be passed into the environment constructor.")

    command_line_args = parser.parse_args()

    serve(vars(command_line_args))


if __name__ == '__main__':
    logger = init_logging()
    main()
