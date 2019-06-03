""" Abstract definition of a game environment. Whenever you wish to make a new environment, make sure to subclass
this to have all of the correct functions. """

import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Dict


class BaseEnvironment(ABC):
    """ Base class for all environments that can be used by colosseum. """

    def __init__(self, config: str = ""):
        """ The base server environment that all implementations should follow.

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
        """ Property holding the number of players present required to play game.

        Returns
        -------
        int
            Minimum number of players for game to start.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def max_players(self) -> int:
        """ Property holding the max number of players present for a game.

        Currently, this should be the same as min_players. Future additions will allow for dynamic games.

        Returns
        -------
        int
            Maximum number of players allowed in game.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def observation_names() -> List[str]:
        """ Static method for returning the names of the observation objects.

        This needs to be static to setup spacetime dataframes.

        Returns
        -------
        List[str]
            The keys of the observation dictionary.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_shape(self) -> Dict[str, tuple]:
        """ Describe the fixed numpy shapes of each observation.

        Returns
        -------
        Dict[str, Tuple[int]]
            The shape, as a tuple, of each numpy array by their name.
        """
        raise NotImplementedError

    @abstractmethod
    def new_state(self, num_players: int = None) -> Tuple[object, List[int]]:
        """ Create a fresh state. This could return a fixed object or randomly initialized on, depending on the game.

        Note that player numbers must be numbers in the set {0, 1, ..., n-1} for an n player game.

        Parameters
        ----------
        num_players : int
            Total number of players in this game.

        Returns
        -------
        new_state : object
            The initial state for the game. This can be any python object you wish.
        new_players: List[int]
            List of players who's turn it is now.
        """
        raise NotImplementedError

    def add_player(self, state: object) -> object:
        """ OPTIONAL Add a new player to an already existing game.

        If your game cannot dynamically change, then you can leave these methods alone.

        Notes
        -----
        Currently not used in any environment and support for dynamic games is still pending.

        Parameters
        ----------
        state : object
            The current state of the game

        Returns
        -------
        new_state : object
            The state of the game after adding a new player.

        """
        raise RuntimeError("Cannot add new players to an existing game.")

    def remove_player(self, state: object, player: int) -> object:
        """ OPTIONAL Remove a player from the current game if they disconnect somehow.

        Notes
        -----
        Currently not used in any environment and support for dynamic games is still pending.

        Parameters
        ----------
        state : object
            The current state of the game.
        player : int
            The player number of remove.

        Returns
        -------
        new_state : object
            The state of the game after remove the player.
        """
        raise RuntimeError("Cannot remove players from an existing game.")

    @abstractmethod
    def next_state(self, state: object, players: [int], actions: [str]) \
            -> Tuple[object, List[int], List[float], bool, Union[List[int], None]]:
        """
        Compute a single step in the game.

        Notes
        -----
        Player numbers must be numbers in the set {0, 1, ..., n-1} for an n player game.

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
        new_players: List[int]
            List of players who's turn it is in the new state now.
        rewards : List[float]
            The reward for each player that acted.
        terminal : bool
            Whether or not the game has ended.
        winners: List[int]
            If the game has ended, who are the winners.
        """
        raise NotImplementedError

    def compute_ranking(self, state: object, players: [int], winners: [int]) -> Dict[int, int]:
        """ OPTIONAL

        Compute the final ranking of all of the players in the game. The state object will be a terminal object.
        By default, this will simply give a list of players that won with a ranking 0 and losers with ranking 1.

        Parameters
        ----------
        state: object
            Terminal state of the game, right after the final move.
        players: List[int]
            A list of all players in the game
        winners: List[int]
            A list of final winners in the game.

        Returns
        -------
        Dict[int, int]
            A Dictionary mapping player number to of rankings for each player.
            Lower rankings indicating better placement.
        """
        winner_set = set(winners)
        return {player: (0 if player in winner_set else 1) for player in players}

    @abstractmethod
    def valid_actions(self, state: object, player: int) -> [str]:
        """ Valid actions for a specific state.

        Parameters
        ----------
        state : object
            The current state of the game.
        player : int
            The player who is executing this action.

        Returns
        -------
        List[str]
            All possible actions for the game.
        """
        raise NotImplementedError

    @abstractmethod
    def is_valid_action(self, state: object, player: int, action: str) -> bool:
        """ Whether or not an action is valid for a specific state.

        Parameters
        ----------
        state : object
            The current state of the game.
        player : int
            The player who is executing this action.
        action : str
            The action the player is executing.

        Returns
        -------
        bool
            Whether or not this is a valid action in the current state.
        """
        raise NotImplementedError

    @abstractmethod
    def state_to_observation(self, state: object, player: int) -> Dict[str, np.ndarray]:
        """ Convert the raw game state to the observation for the agent. Maps each observation name into an observation.

        This can return different values for the different players. Default implementation is just the identity."""
        raise NotImplementedError

    # Serialization Methods
    @staticmethod
    def serializable() -> bool:
        """ Some environments may allow for the full game state to be serializable and transferable to the agents.

        Returns
        -------
        bool
            Whether or not this class supports serialization of the state."""
        return False

    @staticmethod
    def serialize_state(state: object) -> bytearray:
        """ Serialize a game state and convert it to a bytearray to be saved or sent over a network.

        Parameters
        ----------
        state : object
            The current game state.

        Returns
        -------
        bytearray
            Serialized byte-string for the state.
        """
        raise NotImplementedError

    @staticmethod
    def deserialize_state(serialized_state: bytearray) -> object:
        """ Convert a serialized bytearray back into a game state.

        Parameters
        ----------
        serialized_state : bytearray
            Serialized byte-string for the state.

        Returns
        -------
        object
            The current game state.
        """
        raise NotImplementedError
