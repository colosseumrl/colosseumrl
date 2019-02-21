# WIP
import spacetime
import numpy as np
from spacetime import Dataframe, Application
from spacetimerl.data_model import ServerState, Player, _Player
from spacetimerl.frame_rate_keeper import FrameRateKeeper
from spacetimerl.base_environment import BaseEnvironment

from time import sleep
from typing import Callable, Type, Optional, List, Dict
import pickle
import logging

logger = logging.getLogger(__name__)


class ClientEnv:
    TickRate = 60

    def __init__(self, dataframe: Dataframe, dimensions: List[str], player_class: Type[Player],
                 server_environment: Optional[Type[BaseEnvironment]] = None):
        self.dataframe: Dataframe = dataframe
        self._server_state: ServerState = self.dataframe.read_all(ServerState)[0]
        self.player: _Player = None

        self.player_class: Type[BaseEnvironment] = player_class
        self.dimensions: List[str] = dimensions

        self._server_environment: Optional[BaseEnvironment] = None
        if server_environment is not None:
            self._server_environment = server_environment(self._server_state.env_config)

        self.fr: FrameRateKeeper = FrameRateKeeper(self.TickRate)

    def __pull(self):
        self.dataframe.pull()
        self.dataframe.checkout()

    def __push(self):
        self.dataframe.commit()
        self.dataframe.push()

    def __tick(self):
        self.fr.tick()

    @property
    def observation(self) -> Dict[str, np.ndarray]:
        """ Current Observation for this player. """
        if self.player is None:
            raise ConnectionError("Not connected to server")

        return {dimension: getattr(self.player, dimension) for dimension in self.dimensions}

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
        """ Full serer state for the game if the environment and the server support it. """
        if not self.server_environment.serializable():
            raise ValueError("Current Environment does not support full state for clients.")
        return self.server_environment.unserialize_state(self.server_state.serialized_state)

    def connect(self, name: str):
        """ Connect to the remote server and wait for the game to start. """
        # Add this player to the game.
        self.__pull()
        self.player: _Player = self.player_class(name=name)
        self.dataframe.add_one(self.player_class, self.player)
        self.__push()
        sleep(0.1)

        # Check to see if it worked.
        self.__push()
        if self.dataframe.read_one(self.player_class, self.player.pid) is None:
            logger.info("Server rejected adding your player, perhaps the max player limit has been reached.")
            self.player = None
            raise ConnectionError("Could not connect to server.")
        logger.info("Connected to server, waiting for game to start...")

        # Wait for game to start
        while self.player.number == -1:
            self.__tick()
            self.__pull()

        logger.info("Game has started. We are player {}".format(self.player.number))

    def wait_for_turn(self):
        while not self.player.turn:
            self.__tick()
            self.__pull()

        return self.observation

    def step(self, action: str):
        if not self.server_state.terminal:
            self.player.action = action
            self.player.ready_for_action_to_be_taken = True
            self.__push()

            while not self.player.turn or self.player.ready_for_action_to_be_taken:
                self.__tick()
                self.__pull()

        reward = self.player.reward_from_last_turn
        terminal = self.terminal

        if terminal:
            winners = pickle.loads(self.server_state.winners)
            self.player.acknowledges_game_over = True
            self.__push()
        else:
            winners = None

        return self.observation, reward, terminal, winners

    def render(self):
        print("No render defined for default client environment")



def client_app(dataframe: Dataframe, app: "RLApp", client_function: Callable,
               player_class: Type[Player], dimension_names: [str], *args, **kwargs):

    client_env = ClientEnv(dataframe=dataframe,
                           dimensions=dimension_names,
                           player_class=player_class,
                           server_environment=app.server_environment)

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
        player_class = Player(dimension_names)
        del df

        def app(*args, **kwargs):
            client = Application(client_app,
                                 dataframe=(self.host, self.port),
                                 Types=[player_class, ServerState],
                                 version_by=spacetime.utils.enums.VersionBy.FULLSTATE)
            client.start(self, main_func, player_class, dimension_names, *args, **kwargs)

        return app



