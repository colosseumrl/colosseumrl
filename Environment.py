from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseEnvironment(ABC):

    @property
    @abstractmethod
    def state_shape(self) -> tuple:
        """ Property holding the numpy shape of a single state. """
        raise NotImplementedError

    @abstractmethod
    def new_state(self) -> np.ndarray:
        """ Create a fresh state. This could return a fixed object or randomly initialized on, depending on the game. """
        raise NotImplementedError

    @abstractmethod
    def next_state(self, state: np.ndarray, player: int, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Compute a single step in the game.

        Parameters
        ----------
        state : np.ndarray
        player: int
        action : int

        Returns
        -------
        new_state : np.ndarray
        reward : float
        terminal : bool

        """
        raise NotImplementedError

    @abstractmethod
    def valid_actions(self, state: np.ndarray) -> [int]:
        """ Valid actions for a specific state. """
        raise NotImplementedError

    def state_to_observation(self, state: np.ndarray, player: int) -> np.ndarray:
        """ Convert the raw game state to the observation for the agent.

        This can return different values for the different players. Default implementation is just the identity."""
        return state