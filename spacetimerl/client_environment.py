# WIP
import spacetime
import numpy as np
from spacetime import Dataframe, Application
from spacetimerl.data_model import ServerState, Player, Observation
from spacetimerl.frame_rate_keeper import FrameRateKeeper
from spacetimerl.base_environment import BaseEnvironment

from time import sleep
from typing import Callable, Type, Optional, List, Dict
import pickle
import logging

logger = logging.getLogger(__name__)


class ClientEnv:
    TickRate = 60

    def __init__(self, dataframe: Dataframe, dimensions: List[str], observation_class: Type[Observation], host: str,
                 server_environment: Optional[Type[BaseEnvironment]] = None):

        self.player_df: Dataframe = dataframe
        self.observation_df: Dataframe = None

        self._server_state: ServerState = self.player_df.read_all(ServerState)[0]
        self._player: Player = None
        self._observation: Type[Observation] = None

        self._observation_class: Type[Observation] = observation_class
        self.dimensions: List[str] = dimensions

        self._host = host

        self._server_environment: Optional[BaseEnvironment] = None
        if server_environment is not None:
            self._server_environment = server_environment(self._server_state.env_config)

        self.fr: FrameRateKeeper = FrameRateKeeper(self.TickRate)

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
    def server_state(self) -> ServerState:
        """ Current full server state. """
        return self._server_state

    @property
    def terminal(self) -> bool:
        """ Whether the game is over or not. """
        return self.server_state.terminal

    @property
    def winners(self) -> Optional[List[int]]:
        """ The winners of the game. """
        return pickle.loads(self.server_state.winners)

    @property
    def server_environment(self) -> BaseEnvironment:
        return self._server_environment

    @property
    def full_state(self):
        """ Full server state for the game if the environment and the server support it. """
        if not self.server_environment.serializable():
            raise ValueError("Current Environment does not support full state for clients.")
        return self.server_environment.deserialize_state(self.server_state.serialized_state)

    def connect(self, name: str):
        """ Connect to the remote server and wait for the game to start. """
        # Add this player to the game.
        self.__pull()
        self._player: Player = Player(name=name)
        self.player_df.add_one(Player, self._player)
        self.__push()
        sleep(0.1)

        # Check to see if adding our Player object to the dataframe worked.
        self.__push()
        if self.player_df.read_one(Player, self._player.pid) is None:
            logger.info("Server rejected adding your player, perhaps the max player limit has been reached.")
            self._player = None
            raise ConnectionError("Could not connect to server.")

        # Wait for game to start
        while self._player.number == -1:
            self.__tick()
            self.__pull()

        logger.info("We are player number {}".format(self._player.number))

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

        logger.info("Connected to server, ready for it to be player's turn.")

    def wait_for_turn(self):
        while not self._player.turn:
            self.__tick()
            self.__pull()

        return self.observation

    def step(self, action: str):
        if not self.server_state.terminal:
            self._player.action = action
            self._player.ready_for_action_to_be_taken = True
            self.__push()

            while not self._player.turn or self._player.ready_for_action_to_be_taken:
                self.__tick()
                self.__pull()

        reward = self._player.reward_from_last_turn
        terminal = self.terminal

        if terminal:
            winners = pickle.loads(self.server_state.winners)
            self._player.acknowledges_game_over = True
            self.__push()
        else:
            winners = None

        return self.observation, reward, terminal, winners

    def render(self):
        print("No render defined for default client environment")


def client_app(dataframe: Dataframe, app: "RLApp", client_function: Callable,
               observation_class: Type[Observation], dimension_names: [str], host, *args, **kwargs):

    client_env = ClientEnv(dataframe=dataframe,
                           dimensions=dimension_names,
                           observation_class=observation_class,
                           server_environment=app.server_environment,
                           host=host)

    client_function(client_env, *args, **kwargs)


class RLApp:
    def __init__(self, host: str, port: int,
                 client_environment: Type[ClientEnv] = ClientEnv,
                 server_environment: Optional[Type[BaseEnvironment]] = None):
        self.client_environment = client_environment
        self.server_environment = server_environment
        self.host = host
        self.port = port

    def __call__(self, main_func: Callable):
        # Get the dimensions required for the player dataframe
        df = Dataframe("dimension_getter", [ServerState], details=(self.host, self.port))
        df.pull()
        df.checkout()
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



