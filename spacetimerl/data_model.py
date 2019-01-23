from rtypes import pcc_set, merge
from rtypes import dimension, primarykey
import numpy as np
import random, sys


@pcc_set
class Player(object):
    pid = primarykey(int)

    number = dimension(int)
    name = dimension(str)
    action = dimension(str)
    ready_for_action_to_be_taken = dimension(bool)
    turn = dimension(bool)
    reward_from_last_turn = dimension(float)
    observation = dimension(str)
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
        self.observation = ""
        self.acknowledges_game_over = False  # So the server can exit once it knows players got their final pull in.
        self.winner = False


    def finalize_player(self, number: int, observation: str):
        self.number = number
        self.observation = observation


@pcc_set
class ServerState(object):
    oid = primarykey(int)
    env_class_name = dimension(str)
    terminal = dimension(bool)
    winners = dimension(str)

    def __init__(self, env_class_name):
        self.oid = random.randint(0, sys.maxsize)
        self.env_class_name = env_class_name
        self.terminal = False
        self.winners = ""
