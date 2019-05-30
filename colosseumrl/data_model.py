""" Data types that will be used by the Spacetime backend. """

import sys
import random
import numpy as np

from rtypes import pcc_set
from rtypes import dimension, primarykey
from typing import List, Dict


def Observation(observation_names: List[str]):
    """ Creates a proper player class with the attributes necessary to transfer the observations. """

    class Observation(_Observation):
        pass

    for name in observation_names:
        setattr(Observation, name, dimension(np.array))

    return pcc_set(Observation)


class _Observation:
    """ Base observation class that specific observations will be created from. """
    pid = primarykey(int)

    def __init__(self, pid: int):
        self.pid = pid

    def set_observation(self, observations: Dict[str, np.ndarray]):
        for key, value in observations.items():
            self.__setattr__(key, value)

@pcc_set
class Player(object):
    pid = primarykey(int)
    authentication_key = dimension(str)

    name = dimension(str)
    number = dimension(int)
    observation_port = dimension(int)

    action = dimension(str)
    reward_from_last_turn = dimension(float)

    turn = dimension(bool)
    ready_for_start = dimension(bool)
    ready_for_action_to_be_taken = dimension(bool)

    winner = dimension(bool)
    acknowledges_game_over = dimension(bool)

    def __init__(self, name, auth_key: str = ""):
        self.pid = random.randint(0, sys.maxsize)
        self.authentication_key = auth_key
        self.name = name
        self.number = -1
        self.action = ""
        self.turn = False  # server is waiting for player to make their action
        self.ready_for_action_to_be_taken = False  # player is ready for their current action to executed, unset when server executes action
        self.reward_from_last_turn = -1.0
        self.acknowledges_game_over = False  # So the server can exit once it knows players got their final pull in.
        self.winner = False
        self.ready_for_start = False
        self.observation_port = -1

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
    server_no_longer_joinable = dimension(bool)
    winners = dimension(str)
    serialized_state = dimension(bytes)

    def __init__(self, env_class_name, env_config, env_dimensions):
        self.oid = random.randint(0, sys.maxsize)
        self.env_class_name = env_class_name
        self.env_config = env_config
        self.env_dimensions = tuple(env_dimensions)
        self.terminal = False
        self.server_no_longer_joinable = False
        self.winners = ""
        self.serialized_state = b""
