from concurrent import futures
import time
import logging
import secrets
import grpc
import argparse

import zmq

from queue import Queue
from threading import Thread, Semaphore
from typing import Type, Dict, List
from collections import deque


from .grpc_gen.server_pb2 import QuickMatchReply, QuickMatchRequest
from .grpc_gen.server_pb2_grpc import MatchmakerServicer, add_MatchmakerServicer_to_server
from rlcompetition.run_match_server import server_app
from rlcompetition.data_model import ServerState, Player, _Observation, Observation

from rlcompetition.config import ENVIRONMENT_CLASSES
from rlcompetition.base_environment import BaseEnvironment
from rlcompetition.util import log_params, is_port_in_use
from rlcompetition.rl_logging import init_logging

from spacetime import Node


_ONE_DAY_IN_SECONDS = 60 * 60 * 24




def match_server_args_factory(tick_rate, realtime, observations_only, env_config_string):
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

    def GetMatch(self, request, context):
        #  Prepare our context and sockets
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("ipc://matchmaker_requests")
        # print(request.SerializeToString(), QuickMatchRequest.FromString(request.SerializeToString()))
        socket.send(request.SerializeToString())
        return QuickMatchReply.FromString(socket.recv())


class MatchProcessJanitor(Thread):

    def __init__(self, match_limit: Semaphore, ports_to_use_queue, env_class, match_server_args):
        super().__init__()
        self.match_limit = match_limit
        self.match_server_args = match_server_args
        self.env_class = env_class
        self.ports_to_use_queue = ports_to_use_queue

    def run(self) -> None:
        port = self.match_server_args['port']

        observation_type: Type[_Observation] = Observation(self.env_class.observation_names())

        app = Node(server_app,
                   server_port=port,
                   Types=[Player, ServerState])
        app.start(self.env_class, observation_type, self.match_server_args)
        del app
        print("Janitor Finished")
        self.ports_to_use_queue.put(port)
        self.match_limit.release()


class MatchmakingThread(Thread):

    def __init__(self, starting_port, hostname, max_simultaneous_games, env_class, tick_rate, realtime, observations_only,
                 env_config_string):
        super().__init__()

        self.env_class = env_class
        self.hostname = hostname

        # Prepare our context and sockets
        context = zmq.Context()
        self.socket = context.socket(zmq.ROUTER)
        self.socket.bind("ipc://matchmaker_requests")
        print("Matchmaker thread listening...")

        # Semaphore for tracking the total number of games running
        self.match_limit = Semaphore(max_simultaneous_games)

        # Helper function to make arguments for match threads
        self.create_match_server_args = match_server_args_factory(tick_rate=tick_rate, realtime=realtime,
                                                                  observations_only=observations_only,
                                                                  env_config_string=env_config_string)

        self.players_per_game = env_class(env_config_string).min_players

        self.ports_to_use = Queue()

        max_port = starting_port + 2*max_simultaneous_games
        for port in range(starting_port, max_port):
            if not is_port_in_use(port):
               self.ports_to_use.put(port)
            else:
                logger.warn("Skipping port {}, already in use.".format(port))

        if self.ports_to_use.qsize() < max_simultaneous_games:
            raise OSError("Port range {} through {} does not have enough unallocated ports "
                          "to hold {} simultaneous games".format(starting_port, max_port, max_simultaneous_games))

    def run(self) -> None:

        match_requests = deque()

        while True:
            identity, _, serialized_request = self.socket.recv_multipart()
            request = QuickMatchRequest.FromString(serialized_request)

            print("Got request from {}".format(request.username))
            auth_key = secrets.token_hex(32)
            match_requests.append((identity, request, auth_key))

            if len(match_requests) >= self.players_per_game:

                self.match_limit.acquire()
                match_port = self.ports_to_use.get()
                match_server_args = self.create_match_server_args(port=match_port)

                match_janitor = MatchProcessJanitor(match_limit=self.match_limit, ports_to_use_queue=self.ports_to_use,
                                                    env_class=self.env_class,
                                                    match_server_args=match_server_args)
                match_janitor.start()

                for _ in range(self.players_per_game):
                    identity, request, auth_key = match_requests.pop()
                    response = QuickMatchReply(username=request.username,
                                               server='{}:{}'.format(self.hostname, match_port),
                                               auth_key=auth_key)

                    self.socket.send_multipart((identity, b"", response.SerializeToString()))





def serve():

    matchmaker_thread = MatchmakingThread(
        hostname='localhost',
        starting_port=21450,
        max_simultaneous_games=50,
        env_class=ENVIRONMENT_CLASSES['blokus'],
        tick_rate=20,
        realtime=False,
        observations_only=False,
        env_config_string="",

    )
    matchmaker_thread.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_MatchmakerServicer_to_server(MatchMakingHandler(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':

    logger = init_logging()

    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("environment", type=str,
    #                     help="The name of the environment. Choices are: ['blokus']")
    # parser.add_argument("--config", '-c', type=str, default="",
    #                     help="Config string that will be passed into the environment constructor.")
    # parser.add_argument("--starting_port", "-p", type=int, default=7676,
    #                     help="Server Port.")
    # parser.add_argument("--tick-rate", "-t", type=int, default=60,
    #                     help="The max tick rate that the server will run on.")
    # parser.add_argument("--realtime", "-r", action="store_true",
    #                     help="With this flag on, the server will not wait for all of the clients to respond.")
    # parser.add_argument("--observations-only", '-f', action='store_true',
    #                     help="With this flag on, the server will not push the true state of the game to the clients "
    #                          "along with observations")
    #
    # main_args = parser.parse_args()
    #
    # # env_class: Type[BaseEnvironment] = get_class(args.environment_class)
    # try:
    #     env_class: Type[BaseEnvironment] = ENVIRONMENT_CLASSES[main_args.environment]
    # except KeyError:
    #     raise ValueError("The \'environment\' argument must must be chosen from the following list: {}".format(
    #         ENVIRONMENT_CLASSES.keys()
    #     ))
    #
    #
    #
    # log_params(main_args)



    serve()