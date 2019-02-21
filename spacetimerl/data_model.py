from rtypes import pcc_set, merge
from rtypes import dimension, primarykey
from typing import List, Dict
import numpy as np
import random, sys


def Observation(observation_names: List[str]):
    """ Creates a proper player class with the attributes necessary to transfer the observations. """
    class Observation(_Observation):
        pass

    for name in observation_names:
        setattr(Player, name, dimension(np.array))

    return pcc_set(Observation)


class _Observation:
    pid = primarykey(int)

    def __init__(self, pid: int):
        self.pid = pid

    def set_observation(self, observations: Dict[str, np.ndarray]):
        for key, value in observations.items():
            self.__setattr__(key, value)

@pcc_set
class Player(object):
    pid = primarykey(int)

    number = dimension(int)
    name = dimension(str)
    observation_port = dimension(int)

    action = dimension(str)
    ready_for_start = dimension(bool)
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

    def finalize_player(self, number: int, observation_port: int):
        self.number = number
        self.observation_port = observation_port


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
