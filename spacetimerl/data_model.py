from rtypes import pcc_set, merge
from rtypes import dimension, primarykey
from typing import List, Dict
import numpy as np
import random, sys


def Player(observation_names: List[str]):
    """ Creates a proper player class with the attributes necessary to transfer the observations. """
    class Player(_Player):
        def __init__(self, name: str):
            super().__init__(name)

        def finalize_player(self, number: int, observations: Dict[str, np.ndarray]):
            self.number = number
            self.set_observation(observations)

        def set_observation(self, observations: Dict[str, np.ndarray]):
            for key, value in observations.items():
                self.__setattr__(key, value)

    for name in observation_names:
        setattr(Player, name, dimension(np.array))

    return pcc_set(Player)


class _Player(object):
    pid = primarykey(int)

    number = dimension(int)
    name = dimension(str)
    action = dimension(str)
    ready_for_action_to_be_taken = dimension(bool)
    turn = dimension(bool)
    reward_from_last_turn = dimension(float)
    acknowledges_game_over = dimension(bool)
    winner = dimension(bool)

    def __init__(self, name):
        self.pid = random.randint(0, sys.maxsize)
        self.name = name
        self.number = -1
        self.action = ""
        self.turn = False  # server is waiting for player to make their action
        self.ready_for_action_to_be_taken = False  # player is ready for their current action to executed, unset when server executes action
        self.reward_from_last_turn = -1.0
        self.acknowledges_game_over = False  # So the server can exit once it knows players got their final pull in.
        self.winner = False

    def finalize_player(self, number: int, observations: Dict[str, np.ndarray]):
        raise NotImplementedError

    def set_observation(self, observations: Dict[str, np.ndarray]):
        raise NotImplementedError


@pcc_set
class ServerState(object):
    oid = primarykey(int)
    env_class_name = dimension(str)
    env_config = dimension(str)
    env_dimensions = dimension(tuple)
    terminal = dimension(bool)
    winners = dimension(str)
    serialized_state = dimension(str)

    def __init__(self, env_class_name, env_config, env_dimensions):
        self.oid = random.randint(0, sys.maxsize)
        self.env_class_name = env_class_name
        self.env_config = env_config
        self.env_dimensions = tuple(env_dimensions)
        self.terminal = False
        self.winners = ""
        self.serialized_state = ""
