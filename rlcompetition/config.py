""" Central configuration file, primarily used for listing the available environments. """

from rlcompetition.envs.blokus.blokus_env import BlokusEnv
from rlcompetition.envs.tron.TronGridEnvironment import TronGridEnvironment
from rlcompetition.envs.tictactoe.tictactoe_2p_env import TicTacToe2PlayerEnv
from rlcompetition.envs.tictactoe.tictactoe_3p_env import TicTacToe3PlayerEnv
from rlcompetition.envs.tictactoe.tictactoe_4p_env import TicTacToe4PlayerEnv

from rlcompetition.test_game import TestGame

ENVIRONMENT_CLASSES = {
    'blokus': BlokusEnv,
    'tron': TronGridEnvironment,
    'tictactoe' : TicTacToe2PlayerEnv,
    'tictactoe_3p' : TicTacToe3PlayerEnv,
    'tictactoe_4p' : TicTacToe4PlayerEnv,
    'test': TestGame,
}
