# WIP
import spacetime
from spacetime import Dataframe, Application
from spacetimerl.data_model import ServerState, Player
from spacetimerl.frame_rate_keeper import FrameRateKeeper
from spacetimerl.base_environment import BaseEnvironment

from time import sleep
from typing import Callable, Type, Optional
import pickle
import logging

logger = logging.getLogger(__name__)


class ClientEnv:
    TickRate = 60

    def __init__(self, dataframe: Dataframe, player_class, dimensions):
        self.dataframe = dataframe
        self.player_class = player_class
        self.dimensions = dimensions

        self.fr = FrameRateKeeper(self.TickRate)
        self.player = None

    def __pull(self):
        self.dataframe.pull()
        self.dataframe.checkout()

    def __push(self):
        self.dataframe.commit()
        self.dataframe.push()

    def __tick(self):
        self.fr.tick()

    @property
    def observation(self):
        if self.player is None:
            raise ConnectionError("Not connected to server")

        return {dimension: getattr(self.player, dimension) for dimension in self.dimensions}

    @property
    def server_state(self):
        return self.dataframe.read_all(ServerState)[0]

    def connect(self, name: str):
        """ Connect to the remote server and wait for the game to start. """
        # Add this player to the game.
        self.__pull()
        self.player = self.player_class(name=name)
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

    def first_observation(self):
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
        terminal = self.server_state.terminal

        if terminal:
            winners = pickle.loads(self.server_state.winners)
            self.player.acknowledges_game_over = True
            self.__push()
        else:
            winners = None

        return self.observation, reward, terminal, winners


def client_app(dataframe: Dataframe, app: "RLApp", client_function: Callable,
               player_class: Type[Player], *args, **kwargs):
    pass


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
            client.start(main_func, player_class, *args, **kwargs)

        return app



