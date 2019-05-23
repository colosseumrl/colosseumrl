""" Central configuration file, primarily used for listing the available environments. """

from rlcompetition.envs.blokus.blokus_env import BlokusEnv
from rlcompetition.envs.tron.TronGridEnvironment import TronGridEnvironment
from rlcompetition.test_game import TestGame

ENVIRONMENT_CLASSES = {
    'blokus': BlokusEnv,
    'tron': TronGridEnvironment,
    'test': TestGame
}
