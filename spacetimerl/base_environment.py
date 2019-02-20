from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Dict
import numpy as np


class BaseEnvironment(ABC):
    def __init__(self, config: str = ""):
        """
        Parameters
        ----------
        config : str
            Optional config string that will be passed into the constructor. You can use this however you like.
            Load options from string, have it point to a file and read it, etc.
        """
        self._config = config

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

    @staticmethod
    @abstractmethod
    def observation_names() -> List[str]:
        """ Static method for returning the names of the observation objects """
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_shape(self) -> Dict[str, tuple]:
        """ Maps each observation name to a numpy shape"""
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
    def valid_actions(self, state: object, player: int) -> [str]:
        """ Valid actions for a specific state. """
        raise NotImplementedError

    @abstractmethod
    def is_valid_action(self, state: object, player: int, action: str) -> bool:
        """ Valid actions for a specific state. """
        raise NotImplementedError

    def state_to_observation(self, state: object, player: int) -> Dict[str, np.ndarray]:
        """ Convert the raw game state to the observation for the agent. Maps each observation name into an observation.

        This can return different values for the different players. Default implementation is just the identity."""
        return {'state': np.array(state)}

    # Serialization Methods
    @staticmethod
    def serializable() -> bool:
        """ Whether or not this class supports serialization of the state.
            This is necessary to allow the client to perform tree search. """
        return False

    @staticmethod
    def serialize_state(state: object) -> str:
        """ Serialize the state and convert it to a string to be sent between the clients. """
        raise NotImplementedError

    @staticmethod
    def unserialize_state(serialized_state: str) -> object:
        """ Convert the serialized string back into a proper state. """
        raise NotImplementedError
