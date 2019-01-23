from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np


class BaseEnvironment(ABC):

    @property
    @abstractmethod
    def min_players(self) -> int:
        """ Property holding the number of players present required to play game. """
        raise NotImplementedError

    @property
    @abstractmethod
    def max_players(self) -> int:
        """ Property holding the max number of players present for a game. """
        raise NotImplementedError

    @property
    @abstractmethod
    def state_shape(self) -> tuple:
        """ Property holding the numpy shape of a single state. """
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_shape(self) -> tuple:
        """ Property holding the numpy shape of a transformed observation state. """
        raise NotImplementedError

    @abstractmethod
    def new_state(self, num_players: int = 1) -> np.ndarray:
        """ Create a fresh state. This could return a fixed object or randomly initialized on, depending on the game. """
        raise NotImplementedError

    @abstractmethod
    def add_player(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def next_state(self, state: np.ndarray, players: [int], action: str) -> Tuple[np.ndarray, float, bool, int, List[int]]:
        """
        Compute a single step in the game.

        Parameters
        ----------
        state : np.ndarray
        players: [int]
        action : str

        Returns
        -------
        new_state : np.ndarray
        reward : float
        terminal : bool
        winner: int - Only matters if terminal = True
        new_players: [int] List of players whos turn it is now.
        """
        raise NotImplementedError

    @abstractmethod
    def valid_actions(self, state: np.ndarray) -> [str]:
        """ Valid actions for a specific state. """
        raise NotImplementedError

    @abstractmethod
    def is_valid_action(self, state: np.ndarray, action: str) -> bool:
        """ Valid actions for a specific state. """
        raise NotImplementedError

    def state_to_observation(self, state: np.ndarray, player: int) -> np.ndarray:
        """ Convert the raw game state to the observation for the agent.

        This can return different values for the different players. Default implementation is just the identity."""
        return state
