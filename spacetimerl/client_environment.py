import spacetime
import numpy as np
from spacetime import Dataframe, Application
from spacetimerl.data_model import ServerState, Player, Observation
from spacetimerl.frame_rate_keeper import FrameRateKeeper
from spacetimerl.base_environment import BaseEnvironment

from time import sleep, time
import struct
from typing import Callable, Type, Optional, List, Dict
import pickle
import logging

logger = logging.getLogger(__name__)


class ClientEnv:
    _TickRate = 60

    def __init__(self, dataframe: Dataframe, dimensions: List[str], observation_class: Type[Observation], host: str,
                 server_environment: Optional[Type[BaseEnvironment]] = None):

        self.player_df: Dataframe = dataframe
        self.observation_df: Dataframe = None

        self._server_state: ServerState = self.player_df.read_all(ServerState)[0]

        assert self._server_state.terminal == False
        print("Server joinable state: {}".format(self._server_state.server_no_longer_joinable))
        assert self._server_state.server_no_longer_joinable == False


        self._player: Player = None
        self._observation: Type[Observation] = None

        self._observation_class: Type[Observation] = observation_class
        self.dimensions: List[str] = dimensions

        self._host = host

        self._server_environment: Optional[BaseEnvironment] = None
        if server_environment is not None:
            self._server_environment = server_environment(self._server_state.env_config)

        self.fr: FrameRateKeeper = FrameRateKeeper(self._TickRate)

        self.is_connected = False

    def __pull(self):
        self.player_df.pull()
        self.player_df.checkout()

        if self.observation_df is not None:
            self.observation_df.pull()
            self.observation_df.checkout()

    def __push(self):
        self.player_df.commit()
        self.player_df.push()

    def __tick(self):
        self.fr.tick()

    @property
    def observation(self) -> Dict[str, np.ndarray]:
        """ Current Observation for this player. """
        if self._observation is None:
            raise ConnectionError("Not connected to server")

        return {dimension: getattr(self._observation, dimension) for dimension in self.dimensions}

    @property
    def terminal(self) -> bool:
        """ Whether the game is over or not. """
        assert self.is_connected
        return self._server_state.terminal

    @property
    def winners(self) -> Optional[List[int]]:
        """ The winners of the game. """
        assert self.is_connected
        return pickle.loads(self._server_state.winners)

    @property
    def server_environment(self) -> BaseEnvironment:
        return self._server_environment

    @property
    def full_state(self):
        """ Full server state for the game if the environment and the server support it. """
        if not self.server_environment.serializable():
            raise ValueError("Current Environment does not support full state for clients.")
        return self.server_environment.deserialize_state(self._server_state.serialized_state)

    def connect(self, name: str) -> int:
        """ Connect to the remote server and wait for the game to start.

        Parameters
        ----------
        name: str
            Your desired Player Name.

        Returns
        -------
        player_number: int
        """
        # Add this player to the game.
        self.__pull()
        self._player: Player = Player(name=name)
        self.player_df.add_one(Player, self._player)
        self.__push()
        sleep(0.1)

        # Check to see if adding our Player object to the dataframe worked.
        self.__push()
        if self.player_df.read_one(Player, self._player.pid) is None:
            logger.error("Server rejected adding your player, perhaps the max player limit has been reached.")
            self._player = None
            raise ConnectionError("Could not connect to server.")

        # Wait for game to start
        while self._player.number == -1:
            self.__tick()
            self.__pull()

        # logger.info("We are player number {}".format(self._player.number))

        # Connect to observation dataframe, and get the initial observation.
        assert self._player.observation_port > 0
        self.observation_df = Dataframe("{}_observation_df".format(self._player.name),
                                        [self._observation_class],
                                        details=(self._host, self._player.observation_port))
        self.__pull()
        self._observation = self.observation_df.read_all(self._observation_class)[0]

        # Make sure that our observation has the dimensions we were expecting.
        assert all([hasattr(self._observation, dimension) for dimension in self.dimensions])

        # Let the server know that we are ready to start.
        self._player.ready_for_start = True
        self.__push()

        self.is_connected = True
        # logger.info("Connected to server, ready for it to be player's turn.")

        return self._player.number

    def wait_for_turn(self):
        """ Block until it is your turn. This is usually only used in the beginning of the game.

        Returns
        -------
        observation:
            The player's observation once its turn has arrived.
        """
        assert self.is_connected

        while not self._player.turn:
            self.__tick()
            self.__pull()

        return self.observation

    def valid_actions(self):
        """ Get a list of all valid moves for the current state.

        Returns
        -------
        moves: list[str]
        """
        if self._server_environment is not None:
            return self._server_environment.valid_actions(self.full_state, self._player.number)
        else:
            raise NotImplementedError("No valid_action is implemented in this client and "
                                      "we do not have access to the full server environment")

    def step(self, action: str):
        """ Perform an action and send it to the server. This wil block until it is your turn again.

        Parameters
        ----------
        action: str
            Your action string.

        Returns
        -------
        observation
        reward
        terminal
        winners
        """
        assert self.is_connected

        if not self._server_state.terminal:
            self._player.action = action
            self._player.ready_for_action_to_be_taken = True
            self.__push()

            while not self._player.turn or self._player.ready_for_action_to_be_taken:
                self.__tick()
                self.__pull()

        reward = self._player.reward_from_last_turn
        terminal = self.terminal

        if terminal:
            winners = pickle.loads(self._server_state.winners)
            self._player.acknowledges_game_over = True
            self.__push()
        else:
            winners = None

        return self.observation, reward, terminal, winners


def client_app(dataframe: Dataframe, app: "RLApp", client_function: Callable,
               observation_class: Type[Observation], dimension_names: [str], host, *args, **kwargs):

    client_env = app.client_environment(dataframe=dataframe,
                           dimensions=dimension_names,
                           observation_class=observation_class,
                           server_environment=app.server_environment,
                           host=host)

    client_function(client_env, *args, **kwargs)


class RLApp:
    def __init__(self, host: str, port: int,
                 client_environment: Type[ClientEnv] = ClientEnv,
                 server_environment: Optional[Type[BaseEnvironment]] = None,
                 time_out: int = 0):
        self.client_environment = client_environment
        self.server_environment = server_environment
        self.host = host
        self.port = port
        self.time_out = time_out

    def __call__(self, main_func: Callable):
        # Get the dimensions required for the player dataframe
        start_time = time()

        while True:

            try:

                while True:
                    try:
                        df = Dataframe("dimension_getter", [ServerState], details=(self.host, self.port))
                    except ConnectionRefusedError as e:
                        if (time() - start_time) > self.time_out:
                            raise e
                    else:
                        break

                df.pull()
                df.checkout()

                if df.read_all(ServerState)[0].server_no_longer_joinable:
                    # This server is from an old game and just hasn't exited yet, wait for a new server.
                    sleep(0.1)
                    continue
                else:
                    break

            except (ConnectionResetError, struct.error):
                sleep(0.1)
                continue

        dimension_names: [str] = df.read_all(ServerState)[0].env_dimensions
        observation_class = Observation(dimension_names)
        del df

        def app(*args, **kwargs):
            client = Application(client_app,
                                 dataframe=(self.host, self.port),
                                 Types=[Player, observation_class, ServerState],
                                 version_by=spacetime.utils.enums.VersionBy.FULLSTATE)
            client.start(self, main_func, observation_class, dimension_names, self.host, *args, **kwargs)

        return app



