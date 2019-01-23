from abc import ABC, abstractmethod
from typing import Tuple, List, Union
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
    def new_state(self, num_players: int = 1) -> Tuple[object, List[int]]:
        """ Create a fresh state. This could return a fixed object or randomly initialized on, depending on the game.

        Returns
        -------
        new_state : np.ndarray
            A state for the new game.
        new_players: [int]
            List of players whos turn it is now.
        """
        raise NotImplementedError

    def add_player(self, state: object) -> np.ndarray:
        """ OPTIONAL Add a new player to an already existing game.

        If your game cannot dynamically change, then you can leave these methods alone."""
        raise RuntimeError("Cannot add new players to an existing game.")

    def remove_player(self, state: object, player: int) -> np.ndarray:
        """ OPTIONAL Remove a player from the current game if they disconnect somehow. """
        raise RuntimeError("Cannot remove players from an existing game.")

    @abstractmethod
    def next_state(self, state: object, players: [int], actions: [str]) \
            -> Tuple[object, List[int], List[float], bool, Union[List[int], None]]:
        """
        Compute a single step in the game.

        Parameters
        ----------
        state : object
            The current state of the game.
        players: [int]
            The players which are taking the given actions.
        actions : [str]
            The actions of each player.

        Returns
        -------
        new_state : object
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
    def valid_actions(self, state: object) -> [str]:
        """ Valid actions for a specific state. """
        raise NotImplementedError

    @abstractmethod
    def is_valid_action(self, state: object, action: str) -> bool:
        """ Valid actions for a specific state. """
        raise NotImplementedError

    def state_to_observation(self, state: object, player: int) -> np.ndarray:
        """ Convert the raw game state to the observation for the agent.
        The observation must be able to be fed into your predictor.

        This can return different values for the different players. Default implementation is just the identity."""
        return np.array(state)
