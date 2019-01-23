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
    def new_state(self, num_players: int = 1) -> Tuple[np.ndarray, List[int]]:
        """ Create a fresh state. This could return a fixed object or randomly initialized on, depending on the game.

        Returns
        -------
        new_state : np.ndarray
            A state for the new game.
        new_players: [int]
            List of players whos turn it is now.
        """
        raise NotImplementedError

    def add_player(self, state: np.ndarray) -> np.ndarray:
        """ Add a new player to an already existing game.

        If your game cannot dynamically change, then you can leave this alone."""
        raise RuntimeError("Cannot add new players to an existing game.")


    @abstractmethod
    def next_state(self, state: np.ndarray, players: [int], actions: [str]) \
            -> Tuple[np.ndarray, List[int], List[float], bool, List[int]]:
        """
        Compute a single step in the game.

        Parameters
        ----------
        state : np.ndarray
            The current state of the game.
        players: [int]
            The players which are taking the given actions.
        actions : [str]
            The actions of each player.

        Returns
        -------
        new_state : np.ndarray
            The new state of the game.
        new_players: [int]
            List of players whos turn it is in the new state now.
        rewards : [float]
            The reward for each player that acted.
        terminal : bool
            Whether or not the game has ended.
        winners: [int]
            If the game has ended, who are the winners.
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
