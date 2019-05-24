import dill
import numpy as np

from time import sleep
from typing import List, Type, Optional, Dict
from spacetime import Dataframe

from .BaseEnvironment import BaseEnvironment
from .data_model import Observation, ServerState, Player
from .FrameRateKeeper import FrameRateKeeper

import logging
logger = logging.getLogger(__name__)


class ClientEnvironment:
    _TickRate = 60

    def __init__(self,
                 dataframe: Dataframe,
                 dimensions: List[str],
                 observation_class: Type[Observation],
                 host: str,
                 server_environment: Optional[Type[BaseEnvironment]] = None,
                 auth_key: str = ''):

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
        self._auth_key = auth_key

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
        return dill.loads(self._server_state.winners)

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
        self._player: Player = Player(name=name, auth_key=self._auth_key)
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
            winners = dill.loads(self._server_state.winners)
            self._player.acknowledges_game_over = True
            self.__push()
        else:
            winners = None

        return self.observation, reward, terminal, winners