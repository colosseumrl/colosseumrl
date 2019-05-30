from typing import Tuple, List, Union, Dict

import dill
import numpy as np
import scipy.signal

from colosseumrl.BaseEnvironment import BaseEnvironment

State = object

# Match these patterns to win
WINNING_SHAPES = [
    np.full((3, 1), 1, np.int8),
    np.full((1, 3), 1, np.int8),
    np.identity(3, np.int8),
    np.rot90(np.identity(3, np.int8), 1)
]

PLAYER_NUM_TO_STRING = {
    -1: " ",
    0: "X",
    1: "O",
}


def _relative_player_id(current_player: int, absolute_player_num: np.ndarray) -> np.ndarray:
    return np.where(absolute_player_num < 0, absolute_player_num, (absolute_player_num - current_player) % 2)


def action_to_string(index: Tuple[int, int]) -> str:
    """Convert an action index into a formatted action string.

    Parameters
    ----------
    index : Tuple[int, int]
        The location where the piece will be placed in the action.

    Returns
    -------
    action_string : str
    """
    return str(index)


def string_to_action(action_str: str) -> Union[Tuple[int, int], None]:
    """Convert a formatted action string into an index.

    Parameters
    ----------
    action_str : str
        The action in string format

    Returns
    -------
    index : Tuple[int, int]
        The location where the piece will be placed in the action.
    """

    if action_str == '':
        return None

    index = tuple(map(int, action_str.replace('(', '').replace(')', '').split(',')))
    return index


def print_board(state: object):
    """Print board to console

    Parameters
    ----------

    state : object
        The state to render

    Notes
    -----
    X marks player 0.
    O marks player 1.
    """

    board, winner = state

    board = board.tolist()

    if winner is not None:
        print("\nWinner: {}".format(winner))
    else:
        print("")

    for i, row in enumerate(board):
        for j, player_num in enumerate(row):
            print(' {} '.format(PLAYER_NUM_TO_STRING[player_num]), end='')
            if j+1 < len(row):
                print('|', end='')
        print('')
        if i+1 < len(board):
            print('-' * (3*3 + 2))
    print("")


class TicTacToe2PlayerEnv(BaseEnvironment):
    r"""
    Full TicTacToe 2Player environment class with access to the actual game state.
    """

    @property
    def min_players(self) -> int:
        r""" Property holding the number of players present required to play the game.

        (Always 2)
        """
        return 2

    @property
    def max_players(self) -> int:
        r""" Property holding the max number of players present for a game.

        (Always 2)
        """
        return 2

    @property
    def observation_shape(self) -> Dict[str, Tuple[int, ...]]:
        """ Property holding the numpy array shapes for each value in an observation dictionary."""

        return {"board": (3, 3)}

    @staticmethod
    def observation_names():
        """ Get the names for each key in an observation dictionary.

        Returns
        -------
        observation_names : List[int]
        """

        return ["board"]

    def new_state(self, num_players: int = 2) -> Tuple[State, List[int]]:
        r"""Create a fresh TicTacToe 2Player board state for a new game.

        Returns
        -------
        new_state : object
            A state for the new game.
        new_players : List[int]
            List of players who's turn it is in this new state.

        Notes
        -----

        States are arbitrary internal game logic types. In a normal use case,
        there is no need to access or modifying individual data in a state.

        States are not in a format intended to be consumable for a reinforcement learning agent.
        Reinforcement leaning agents are intended to take observations as input,
        and :py:func:`state_to_observation` can be used to convert states into observations.

        """
        if num_players is None:
            num_players = 2

        assert num_players == 2

        board = np.full((3, 3), -1, np.int8)

        winner = None

        return (board, winner), [0]

    # Serialization Methods
    @staticmethod
    def serializable() -> bool:
        """ Whether or not this class supports state serialization.

        (This always returns True for TicTacToe)

        Returns
        -------
        is_serializable : bool
            True
        """
        return True

    @staticmethod
    def serialize_state(state: object) -> bytearray:
        """ Serialize a game state and convert it to a bytearray to be saved or sent over a network.

        Parameters
        ----------
        state : object
            state to be serialized

        Returns
        -------
        serialized_state : bytearray
            serialized state

        """
        return dill.dumps(state)

    @staticmethod
    def deserialize_state(serialized_state: bytearray) -> State:
        """ Convert a serialized bytearray back into a game state.

        Parameters
        ----------
        serialized_state : bytearray
            state bytearray to be deserialized

        Returns
        -------
        deserialized_state : object
            deserialized state

        """
        return dill.loads(serialized_state)

    def current_rewards(self, state: object) -> List[float]:
        """Returns current reward for each player (in absolute order, not relative to any specific player

        Parameters
        ----------
        state : object
            The current state to calculate rewards from

        Returns
        -------
        rewards : List[float]
            A vector containing the current rewards for each player

        """
        board, winner = state

        if winner is not None:
            return [1 if p == winner else -1 for p in range(self.max_players)]
        else:
            return [0 for _ in range(self.max_players)]

    def next_state(self, state: object, players: List[int], actions: List[str]) \
            -> Tuple[State, List[int], List[float], bool, Union[List[int], None]]:
        """Perform a game step from a given state.


        Parameters
        ----------
        state : object
            The current state to execute a game step from.
        players : List[int]
            The players who's turn it is and are executing actions.
            For TicTacToe, only one player should ever be passed in this list at a time.
        actions : List[str],
            The actions to be executed by the players who's turn it is.
            For TicTacToe, only one action should ever be passed in this list at a time.

        Returns
        -------
        next_state : object
            The new state resulting after the game step.
        next_players : List[int]
            The new players who's turn it is after the game step.
            For TicTacToe, this will always only be one player.
        rewards : List[float]
            Rewards for the players who's turn it was.
            For TicTacToe, this will always only be one reward for the single player that execute the action.
        terminal : bool
            Whether the game is now over.
        winners : Union[List[int], None]
            The players that won the game if it is over, else None.


        Notes
        -----

        States are arbitrary internal game logic types. In a normal use case,
        there is no need to access or modifying individual data in a state.

        States are not in a format intended to be consumable for a reinforcement learning agent.
        Reinforcement leaning agents are intended to take observations as input,
        and state_to_observation can be used to convert states into observations.

        """
        board, winner = state
        new_board = board.copy()

        action = actions[0]
        player_num = players[0]

        winners = None
        reward = 0
        terminal = False

        if len(action) > 0 and self.is_valid_action(state, player_num, action) and winner is None:
            index = string_to_action(action)
            new_board[index] = player_num
            player_mask = (new_board == player_num)
            for pattern in WINNING_SHAPES:
                if 3 in scipy.signal.correlate2d(player_mask, pattern, 'valid'):
                    winner = player_num
                    break

        if winner is not None:
            if winner == player_num:
                reward = 1
            else:
                reward = -1
            winners = [winner]
            terminal = True

        if self.valid_actions(state=(new_board, winner), player=player_num) == ['']:
                terminal = True

        new_player_num = (player_num + 1) % 2

        return (new_board, winner), [new_player_num], [reward], terminal, winners

    def valid_actions(self, state: object, player: int) -> List[str]:
        """ Valid actions for a specific state and player.
        If there are no valid actions, empty string is given to represent a no-op

        Parameters
        ----------
        state : object
            The current state to execute a game step from.
        player : int
            The player for which valid actions will be returned.

        Returns
        -------
        valid_actions : list[int]
            A list of valid action strings which the player may execute.

        Notes
        -----
        Players must always choose actions included in this list.
        If no actions are valid for a player, this function returns an empty string.
        When it is a player's turn, if the player has no valid actions,
        it must pass an empty string as its action for :py:func:`next_state`
        for the game to continue.

        This method does not keep track of who's turn it is. That is up to the user.
        If the specified player can physically place a piece at a location, it will be returned as a valid action.
        """
        board, winners = state
        valid_actions = list(map(lambda x: str(x), zip(*np.where(board == -1))))
        if len(valid_actions) == 0:
            valid_actions.append("")
        return valid_actions

    def is_valid_action(self, state: object, player_num: int, action: str) -> bool:
        """ Returns True if an action is valid for a specific player and state.

        Parameters
        ----------
        state : object
            The current state to execute a game step from.
        player_num : int
            The player that would be executing the action.
        action : str
            The action in question

        Returns
        -------
        is_action_valid : bool
            whether this action is valid

        Notes
        -----
        This method does not keep track of who's turn it is. That is up to the user.
        If a piece may be physically placed at the location suggest by the action,
        this method returns true, regardless of who just executed their turn or who should be going now.
        """

        if len(action) == 0:
            return False

        board, winners = state
        index = string_to_action(action)

        return board[index] == -1

    def state_to_observation(self, state: object, player: int) -> Dict[str, np.ndarray]:
        """ Convert the raw game state to a consumable observation for a specific player agent.

        Parameters
        ----------
        state : object
            The state to create an observation for
        player : int
            The player who is intended to view the observation

        Returns
        -------
        observation : Dict[str, np.ndarray]
            The observation for the player RL agent to view

        Notes
        -----
        Observations are specific to individual players.
        Every observation is presented as if the player intended to receive it were actually player 0.
        This is done so that an RL agent only has to learn to perform moves that make player 0 win
        and other players lose.
        """
        board, winners = state
        board = _relative_player_id(current_player=player, absolute_player_num=board)

        return {'board': board}
