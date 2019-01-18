from multiprocessing import Pipe
import numpy as np
from typing import Tuple
import sys

from spacetimerl.Environment import BaseEnvironment


class RemoteLocalEnv(BaseEnvironment):

    def __init__(self, server_env):

        self.env = server_env
        self.state = self.new_state()

    def set_server_state(self, new_state: np.ndarray):
        self.state = new_state

    @property
    def state_shape(self) -> tuple:
        """ Property holding the numpy shape of a single state. """
        return self.env.state_shape

    @property
    def observation_shape(self) -> tuple:
        """ Property holding the numpy shape of a transformed observation state. """
        return self.env.observation_shape

    def new_state(self, num_players: int = 1) -> np.ndarray:
        """ Create a fresh state. This could return a fixed object or randomly initialized on, depending on the game. """
        return self.env.new_state(num_players)

    def add_player(self, state: np.ndarray) -> np.ndarray:
        return self.env.add_player(state)

    def next_observation(self, player: int, action: int) -> Tuple[np.ndarray, float, bool, int]:
        """
        Compute a single step in the game.

        Parameters
        ----------
        player: int
        action : int

        Returns
        -------
        new_observation : np.ndarray
        reward : float
        terminal : bool
        winner: int - Only matters if terminal = True
        """

        new_state, reward, terminal, winner = self.env.next_state(self.state, player, action)
        new_observation = self.env.state_to_observation(new_state, player)
        self.state = new_state

        return new_observation, reward, terminal, winner

    def valid_actions(self, state: np.ndarray = None) -> [int]:
        """ Valid actions for a specific state. """
        return self.env.valid_actions(self.state if state is None else state)

    def state_to_observation(self, state: np.ndarray, player: int) -> np.ndarray:
        """ Convert the raw game state to the observation for the agent.

        This can return different values for the different players. Default implementation is just the identity."""
        return self.env.state_to_observation(state, player)


