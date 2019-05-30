import dill
import numpy as np

from typing import List, Type, Optional, Dict, Tuple
from spacetime import Dataframe

from .data_model import Observation, ServerState, Player
from .FrameRateKeeper import FrameRateKeeper
from .BaseEnvironment import BaseEnvironment

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
        self.observation_df: Optional[Dataframe] = None

        self._server_state: ServerState = self.player_df.read_all(ServerState)[0]

        assert self._server_state.terminal is False, "Connecting to a server with no active game."
        assert self._server_state.server_no_longer_joinable is False, "Server is not accepting new connection."

        self._player: Optional[Player] = None
        self._dimensions: List[str] = dimensions
        self._observation: Observation = None
        self._observation_class: Type[Observation] = observation_class

        self._host: str = host
        self._auth_key: str = auth_key

        self._server_environment: Optional[BaseEnvironment] = None
        if server_environment is not None:
            self._server_environment = server_environment(self._server_state.env_config)

        self.fr: FrameRateKeeper = FrameRateKeeper(self._TickRate)
        self.connected: bool = False

    def pull_dataframe(self):
        self.player_df.pull()
        self.player_df.checkout()

        if self.observation_df is not None:
            self.observation_df.pull()
            self.observation_df.checkout()

    def push_dataframe(self):
        self.player_df.commit()
        self.player_df.push()

    def tick(self):
        return self.fr.tick()

    @property
    def observation(self) -> Dict[str, np.ndarray]:
        if self._observation is None:
            raise ConnectionError("Not connected to game server.")

        return {dimension: getattr(self._observation, dimension) for dimension in self.dimensions}

    @property
    def terminal(self) -> bool:
        assert self.connected, "Not connected to game server."
        return self._server_state.terminal

    @property
    def winners(self) -> Optional[List[int]]:
        assert self.connected, "Not connected to game server."
        return dill.loads(self._server_state.winners)

    @property
    def server_environment(self) -> Optional[BaseEnvironment]:
        return self._server_environment

    @property
    def dimensions(self) -> List[str]:
        return self._dimensions

    @property
    def full_state(self):
        """ Full server state for the game if the environment and the server support it. """
        if not self.server_environment.serializable():
            raise ValueError("Current Environment does not support full state for clients.")
        return self.server_environment.deserialize_state(self._server_state.serialized_state)

    def connect(self, username: str, timeout: Optional[float] = None) -> int:
        """ Connect to the remote server and wait for the game to start.

        Parameters
        ----------
        username: str
            Your desired Player Name.
        timeout: float
            Optional timout for how long to wait before abandoning connection

        Returns
        -------
        player_number: int
        """

        # Add this player to the game.
        self.pull_dataframe()
        self._player: Player = Player(name=username, auth_key=self._auth_key)
        self.player_df.add_one(Player, self._player)
        self.push_dataframe()

        # Check to see if adding our Player object to the dataframe worked.
        self.pull_dataframe()

        if timeout:
            self.fr.start_timeout(timeout)

        while True:
            if self.tick() and timeout:
                self._player = None
                raise ConnectionError("Timed out connecting to server.")

            # The server should remove our player object if it doesnt want us to connect.
            if self.player_df.read_one(Player, self._player.pid) is None:
                self._player = None
                raise ConnectionError("Server rejected adding your player.")

            # If the game start timed out, then we break out now.
            if self._server_state.terminal:
                self._player = None
                raise ConnectionError("Server could not successfully start game.")

            # If we have been given a player number, it means the server is ready for a game to start.
            if self._player.number >= 0:
                break

            self.pull_dataframe()

        # Connect to observation dataframe, and get the initial observation.
        assert self._player.observation_port > 0, "Server failed to create an observation dataframe."
        self.observation_df = Dataframe("{}_observation_df".format(self._player.name),
                                        [self._observation_class],
                                        details=(self._host, self._player.observation_port))

        # Receive the first observation and ensure correct game
        self.pull_dataframe()
        self._observation = self.observation_df.read_all(self._observation_class)[0]
        assert all([hasattr(self._observation, dimension) for dimension in self.dimensions]), \
            "Mismatch in game between server and client."

        # Let the server know that we are ready to start.
        self._player.ready_for_start = True
        self.push_dataframe()

        self.connected = True
        return self._player.number

    def wait_for_start(self, timeout: Optional[float] = None):
        """ Secondary name for to be clearer when starting game. """
        self.wait_for_turn(timeout)

    def wait_for_turn(self, timeout: Optional[float] = None):
        """ Block until it is your turn. This is usually only used in the beginning of the game.

        Parameters
        ----------
        timeout: float
            An optional hard timeout on waiting for the game to start.

        Returns
        -------
        observation:
            The player's observation once its turn has arrived.
        """
        assert self.connected, "Not connected to game server."

        if timeout:
            self.fr.start_timeout(timeout)

        while not self._player.turn:
            if self.terminal:
                raise ConnectionError("Server finished game while we were waiting.")

            if self.tick() and timeout:
                raise ConnectionError("Timed out waiting for a game.")

            self.pull_dataframe()

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

    def step(self, action: str) -> Tuple[Dict[str, np.ndarray], float, bool, Optional[List[int]]]:
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
        if not self.terminal:
            self._player.action = action
            self._player.ready_for_action_to_be_taken = True
            self.push_dataframe()

            while not self._player.turn or self._player.ready_for_action_to_be_taken:
                self.tick()
                self.pull_dataframe()

        reward = self._player.reward_from_last_turn
        terminal = self.terminal

        winners = None
        if terminal:
            winners = dill.loads(self._server_state.winners)
            self._player.acknowledges_game_over = True
            self.push_dataframe()

        return self.observation, reward, terminal, winners
