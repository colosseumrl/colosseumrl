from spacetimerl.base_environment import BaseEnvironment

from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Dict, Type
import numpy as np


class TurnBasedEnvironment(ABC):
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
    def new_state(self, num_players: int = 1) -> object:
        """ Create a fresh state. This could return a fixed object or randomly initialized on, depending on the game.

        Returns
        -------
        new_state : np.ndarray
            A state for the new game.
        new_players: [int]
            List of players whos turn it is now.
        """
        raise NotImplementedError

    @abstractmethod
    def next_state(self, state: object, player_num: int, action: str) \
            -> Tuple[object, float, bool, Union[List[int], None]]:
        raise NotImplementedError

    @abstractmethod
    def valid_actions(self, state: object, player_num: int) -> [str]:
        """ Valid actions for a specific state. """
        raise NotImplementedError

    @abstractmethod
    def is_valid_action(self, state: object, player_num: int, action: str) -> bool:
        """ Valid actions for a specific state. """
        raise NotImplementedError

    def state_to_observation(self, state: object, player_num: int) -> Dict[str, np.ndarray]:
        """ Convert the raw game state to the observation for the agent. Maps each observation name into an observation.

        This can return different values for the different players. Default implementation is just the identity."""
        return {'state': np.array(state)}


    # Serialization Methods
    @staticmethod
    def serializable() -> bool:
        """ Whether or not this class supports serialization of the current state.
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


def turn_based_environment(cls):
    class TurnBasedWrapper(BaseEnvironment):
        def __init__(self, *args, **kwargs):
            self.base = cls()

        @property
        def min_players(self) -> int:
            return self.base.min_players

        @property
        def max_players(self) -> int:
            return self.base.max_players

        @staticmethod
        def observation_names() -> List[str]:
            """ Static method for returning the names of the observation objects """
            return cls.observation_names()

        @property
        def observation_shape(self) -> Dict[str, tuple]:
            """ Maps each observation name to a numpy shape"""
            return self.base.observation_shape

        def new_state(self, num_players: int = 1) -> Tuple[object, List[int]]:
            return (num_players, self.base.new_state(num_players)), [0]

        def next_state(self, state: object, players: [int], actions: [str]) \
                -> Tuple[object, List[int], List[float], bool, Union[List[int], None]]:
            num_players, state = state
            player: int = players[0]
            action: str = actions[0]

            state, reward, terminal, winners = self.base.next_state(state, player, action)

            state = (num_players, state)
            new_players = [(player + 1) % num_players]
            rewards = [reward]
            winners = winners

            return state, new_players, rewards, terminal, winners

        def valid_actions(self, state: object, player: int) -> [str]:
            num_players, state = state
            return self.base.valid_actions(state, player)

        def is_valid_action(self, state: object, player: int, action: str) -> bool:
            num_players, state = state
            return self.base.is_valid_action(state, player, action)

        def state_to_observation(self, state: object, player: int) -> Dict[str, np.ndarray]:
            num_players, state = state
            return self.base.state_to_observation(state, player)

        @staticmethod
        def serializable():
            return cls.serializable()

        @staticmethod
        def serialize_state(state: object):
            return cls.serialize_state(state)

        @staticmethod
        def deserialize_state(serialized_state: str):
            return cls.deserialize_state(serialized_state)

    return TurnBasedWrapper


