""" Central configuration file, primarily used for listing the available environments. """
from typing import Type, Callable, Dict, List
from .BaseEnvironment import BaseEnvironment


def blokus() -> Type[BaseEnvironment]:
    from colosseumrl.envs.blokus import BlokusEnvironment
    return BlokusEnvironment


def tron() -> Type[BaseEnvironment]:
    from colosseumrl.envs.tron.TronGridEnvironment import TronGridEnvironment
    return TronGridEnvironment


def test_game() -> Type[BaseEnvironment]:
    from colosseumrl.envs.testgame.TestGame import TestGame
    return TestGame


def tic_tac_toe(n) -> Callable[[], Type[BaseEnvironment]]:
    def ttt() -> Type[BaseEnvironment]:
        if n == 2:
            from colosseumrl.envs.tictactoe.tictactoe_2p_env import TicTacToe2PlayerEnv
            return TicTacToe2PlayerEnv
        if n == 3:
            from colosseumrl.envs.tictactoe.tictactoe_3p_env import TicTacToe3PlayerEnv
            return TicTacToe3PlayerEnv
        if n == 4:
            from colosseumrl.envs.tictactoe.tictactoe_4p_env import TicTacToe4PlayerEnv
            return TicTacToe4PlayerEnv
        else:
            raise ValueError("No Tic Tac Toe with {} players".format(n))
    return ttt


ENVIRONMENT_CLASSES: Dict[str, Callable[[], Type[BaseEnvironment]]] = {
    'blokus': blokus,
    'tron': tron,
    'test': test_game,
    'tictactoe': tic_tac_toe(2),
    'tictactoe_3p': tic_tac_toe(3),
    'tictactoe_4p': tic_tac_toe(4)
}


def get_environment(environment: str) -> Type[BaseEnvironment]:
    """ Get an environment class by name.

    Run available_environments() to see all currently implemented games.

    Parameters
    ----------
    environment : str
        The environment name.

    Returns
    -------
    Type[BaseEnvironment]
        The environment type
    """
    return ENVIRONMENT_CLASSES[environment]()


def available_environments() -> List[str]:
    """ See all currently available environments.

    Returns
    -------
    List[str]
        A list of environment names that can be accessed with get_environment.

    """
    return list(ENVIRONMENT_CLASSES.keys())



