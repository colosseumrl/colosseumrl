from copy import deepcopy
from typing import Tuple, List, Union, Dict

import dill
import numpy as np
from . import gui
from .ai import AI
from .board import Board, PIECE_TYPES, ORIENTATIONS, BOARD_TO_PLAYER_OBSERVATION_ROTATION_MATRICES, PLAYER_OBSERVATION_TO_BOARD_ROTATION_MATRICES
from colosseumrl.BaseEnvironment import BaseEnvironment

PLAYER_TO_COLOR = {
    0: 1,
    1: 2,
    2: 3,
    3: 4
}

COLOR_TO_PLAYER = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    0: -1
}

PIECE_NAME_TO_INDEX = {piece_name: i for i, piece_name in enumerate(PIECE_TYPES.keys())}

State = object


def _rotate_board_for_player_perspective(board, player):
    return np.rot90(board, k=-player)


def _separate_offset_from_orientation(orientation_string):
    orientation = []
    offset = []

    for c in orientation_string:
        if c.isdigit():
            offset.append(c)
        else:
            orientation.append(c)

    return ''.join(orientation), ''.join(offset)


def _relative_player_id(current_player: int, absolute_player_num) -> int:
    if absolute_player_num < 0:
        return absolute_player_num

    return (absolute_player_num - current_player) % 4


def action_to_string(piece_type: str, index: Tuple[int, int], orientation: str) -> str:
    """Convert a piece_type, index, and orientation into a formatted action string.

    Parameters
    ----------
    piece_type : str
        The type of blokus piece to be placed in the action.
    index : Tuple[int, int]
        The location where the piece will be placed in the action.
    orientation : str
        The orientation in which the piece will be placed in the action.

    Returns
    -------
    action_string : str

    See Also
    --------
    blokus.blokus_env.BlokusEnv.all_piece_types
        Get list of all possible piece types
    blokus.blokus_env.BlokusEnv.all_orientations
         Get list of all possible orientations
    blokus.blokus_env.BlokusEnv.is_valid_action
        Check if action string is valid
    """
    return "{};{};{}".format(piece_type, index, orientation)


def string_to_action(action_str: str) -> Union[Tuple[str, Tuple[int, int], str], None]:
    """Convert a formatted action string into piece_type, index, and orientation.

    Parameters
    ----------
    action_str : str
        The action in string format

    Returns
    -------
    piece_type : str
        The type of blokus piece to be placed in the action.
    index : Tuple[int, int]
        The location where the piece will be placed in the action.
    orientation : str
        The orientation in which the piece will be placed in the action.
    """

    if action_str == '':
        return None

    piece_type, index, orientation = action_str.split(";")
    index = tuple(map(int, index.replace('(', '').replace(')', '').split(',')))
    return piece_type, index, orientation

def start_gui():
    """Initialize graphical interface in order to render board.

    You must first call this function once before making calls to :py:func:`blokus.blokus_env.display_board`.

    See Also
    --------
    blokus.blokus_env.display_board
    blokus.blokus_env.terminate_gui

    """
    gui.start_gui()


def terminate_gui():
    """Terminate the graphical interface.

    See Also
    --------
    blokus.blokus_env.start_gui
    blokus.blokus_env.display_board

    """
    gui.terminate_gui()


def display_board(state: object, player_num: int, winners: Union[List[int], None] = None):
    """Render Board with graphical interface

    Parameters
    ----------

    state : object
        The state to render
    player_num : int
        The current player whose turn it is.
    winners : List[int] (optional)
        The winners of the game

    Notes
    -----
    :py:func:`blokus.blokus_env.start_gui` must first be called once before calling this function.

    See Also
    --------
    blokus.blokus_env.start_gui
    blokus.blokus_env.terminate_gui
    blokus.blokus_env.print_board

    """

    board, round_count, players = state
    current_player = players[player_num]
    gui.display_board(board_contents=board.board_contents, current_player=current_player, players=players,
                      round_count=round_count, winners=winners)

def print_board(state: object):
    """Print board to console

    Parameters
    ----------

    state : object
        The state to render

    Notes
    -----
    Pieces are marked by the player number that placed them.
    -1 marks empty spaces.

    See Also
    --------
    blokus.blokus_env.start_gui
    blokus.blokus_env.terminate_gui
    blokus.blokus_env.display_board
    """

    print(state[0].board_contents-1)


class BlokusEnvironment(BaseEnvironment):
    r"""
    Full Blokus environment class with access to the actual game state.
    """

    @property
    def min_players(self) -> int:
        r""" Property holding the number of players present required to play the game.

        (Always 4 for Blokus)
        """
        return 4

    @property
    def max_players(self) -> int:
        r""" Property holding the max number of players present for a game.

        (Always 4 for Blokus)
        """
        return 4

    @property
    def observation_shape(self) -> Dict[str, Tuple[int, ...]]:
        """ Property holding the numpy array shapes for each value in an observation dictionary."""

        return {"board": (20, 20), "pieces": (4, 21), "score": (4,), "player": (1,)}

    @staticmethod
    def observation_names():
        """ Get the names for each key in an observation dictionary.

        Returns
        -------
        observation_names : List[int]
        """

        return ["board", "pieces", "score", "player"]

    @staticmethod
    def all_piece_types() -> List[str]:
        """ Get the names every possible piece type in a game of Blokus.

        Returns
        -------
        piece_types : List[str]
        """


        return PIECE_TYPES.keys()

    @staticmethod
    def all_orientations() -> List[str]:
        """ Get the names every possible piece orientation in a game of Blokus.

        Returns
        -------
        orientations : List[str]
        """
        return ORIENTATIONS

    def new_state(self, num_players: int = 4) -> State:
        r"""new_state(self) -> object
        Create a fresh Blokus board state for a new game.

        Returns
        -------
        new_state : object
            A state for the new game.
        new_players : List[int]
            List of players who's turn it is in this new state.


        See Also
        --------

        blokus.blokus_env.BlokusEnv.state_to_observation : Convert states to player specific observations with this method.
        blokus.blokus_env.BlokusEnv.next_state : Pass states to this method to perform a game step.


        Notes
        -----

        States are arbitrary internal Blokus logic types. In a normal use case,
        there is no need to access or modifying individual data in a state.

        States are not in a format intended to be consumable for a reinforcement learning agent.
        Reinforcement leaning agents are intended to take observations as input,
        and :py:func:`blokus.blokus_env.BlokusEnv.state_to_observation` can be used to convert states into observations.

        """
        if num_players is None:
            num_players = 4

        assert num_players == 4

        board = Board()
        red = AI(board, 1)
        blue = AI(board, 2)
        green = AI(board, 3)
        yellow = AI(board, 4)

        return (board, 0, [red, blue, green, yellow]), [0]

    # Serialization Methods
    @staticmethod
    def serializable() -> bool:
        """ Whether or not this class supports state serialization.

        (This always returns True for Blokus)

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
        """Returns current reward for each player (in absolute order, not reltive to any specific player

        Parameters
        ----------
        state : object
            The current state to calculate rewards from

        Returns
        -------
        rewards : List[float]
            A vector containing the current rewards for each player

        """
        board, round_count, players = state

        return [p.player_score for p in players]

    def next_state(self, state: object, players: int, actions: str) \
            -> Tuple[State, List[int], List[float], bool, Union[List[int], None]]:
        """ Perform a game step from a given state.


        Parameters
        ----------
        state : object
            The current state to execute a game step from.
        players : List[int]
            The players who's turn it is and are executing actions.
            For Blokus, only one player should ever be passed in this list at a time.
        actions : List[str],
            The actions to be executed by the players who's turn it is.
            For Blokus, only one action should ever be passed in this list at a time.

        Returns
        -------
        next_state : object
            The new state resulting after the game step.
        next_players : List[int]
            The new players who's turn it is after the game step.
            For Blokus, this will always only be one player.
        rewards : List[float]
            Rewards for the players who's turn it was.
            For Blokus, this will always only be one reward for the single player that execute the action.
        terminal : bool
            Whether the game is now over.
        winners : Union[List[int], None]
            The players that won the game if it is over, else None.


        See Also
        --------

        blokus.blokus_env.BlokusEnv.state_to_observation : Convert states to player specific observations with this method.
        blokus.blokus_env.BlokusEnv.new_state : Create new game states with this method.


        Notes
        -----

        States are arbitrary internal Blokus logic types. In a normal use case,
        there is no need to access or modifying individual data in a state.

        States are not in a format intended to be consumable for a reinforcement learning agent.
        Reinforcement leaning agents are intended to take observations as input,
        and blokus.blokus_env.state_to_observation can be used to convert states into observations.

        """
        player_num = players[0]
        action = actions[0]

        board, round_count, players = state

        players = deepcopy(players)
        new_board = Board(board)

        color = PLAYER_TO_COLOR[player_num]

        current_player = players[player_num]

        if len(action) > 0:
            piece_type, index, orientation = string_to_action(action)
            new_board.update_board(color, piece_type, index, orientation, round_count, True)
            current_player.update_player(piece_type)

        if not any(p.check_moves(board, round_count) for p in players):
            terminal = True
            max_score = 0
            scores = []
            winners = []

            for p in players:
                scores.append((p.player_color, p.player_score))
                if p.player_score > max_score:
                    max_score = p.player_score

            for player_color, score in scores:
                if score == max_score:  # Prints all scores equal to the max score (accounts for ties)
                    winners.append(COLOR_TO_PLAYER[player_color])

            sorted_scores = sorted(scores, key=lambda x: x[1])
            reward = sorted_scores.index((current_player.player_color, current_player.player_score))
        else:
            winners = None
            reward = 0
            terminal = False

        if player_num == 3:
            round_count += 1

        new_player_num = (player_num + 1) % 4

        return (new_board, round_count, players), [new_player_num], [reward], terminal, winners

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


        See Also
        --------
        blokus.blokus_env.action_to_string
        blokus.blokus_env.string_to_action
        blokus.blokus_env.is_valid_action
        blokus.blokus_env.BlokusEnv.next_state
            If an action is valid, you can pass it to this method.

        Notes
        -----
        Players must always choose actions included in this list.
        If no actions are valid for a player, this function returns an empty string.
        When it is a player's turn, if the player has no valid actions,
        it must pass an empty string as its action for :py:func:`blokus.blokus_env.BlokusEnv.next_state`
        for the game to continue.

        This method does not keep track of who's turn it is. That is up to the user.
        If the specified player can physically place a piece at a location, it will be returned as a valid action.
        """
        actions_dict = self.valid_actions_dict(state=state, player=player)

        valid_moves = []
        for piece_type, index_orientation_dict in actions_dict.items():
            for index, orientation_list in index_orientation_dict.items():
                for orientation in orientation_list:
                    valid_moves.append(action_to_string(piece_type=piece_type, index=index, orientation=orientation))

        if len(valid_moves) == 0:
            valid_moves.append("")

        return valid_moves

    def player_perspective_valid_actions(self, state: object, player: int) -> List[str]:
        """ Valid actions for a specific state and player from the player's perspective
        in coordinance with the player's rotated observation of the board.
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
            These actions are rotated to match this player's perspective of the board,


        See Also
        --------
        blokus.blokus_env.action_to_string
        blokus.blokus_env.string_to_action
        blokus.blokus_env.convert_player_perspective_action_to_real_action
            You have to convert a player-perspective action to a real action before passing it to the environment

        Notes
        -----
        This method does not keep track of who's turn it is. That is up to the user.
        If the specified player can physically place a piece at a location (from the player's perspective),
        it will be returned as a valid action.
        """

        actions_dict = self.valid_actions_dict(state=state, player=player)

        valid_moves = []
        for piece_type, index_orientation_dict in actions_dict.items():
            for index, orientation_list in index_orientation_dict.items():
                for orientation in orientation_list:

                    player_action = self.convert_real_action_to_player_perspective_action(
                        action_to_string(piece_type=piece_type, index=index, orientation=orientation), player=player
                    )

                    valid_moves.append(player_action)

        if len(valid_moves) == 0:
            valid_moves.append("")

        return valid_moves

    def convert_real_action_to_player_perspective_action(self, action: str, player: int) -> str:
        """ Converts a real action consumable by the actual environment to the corresponding player-perspective action
        that is in coordinance with the player's rotated observation of the board


        Parameters
        ----------
        action: str
            The real action.
        player : int
            The player to view this action from.

        Returns
        -------
        player_action : str
            The player-perspective action.

        See Also
        --------
        blokus.blokus_env.convert_player_perspective_action_to_real_action
            You have to convert a player-perspective action to a real action before passing it to the environment
        """

        if not action:
            return ""

        piece_type, index, orientation = string_to_action(action)

        index = tuple((np.matmul(BOARD_TO_PLAYER_OBSERVATION_ROTATION_MATRICES[player],
                                 (np.asarray(index) - 9.5)) + 9.5).astype(np.int32))

        orientation, offset = _separate_offset_from_orientation(orientation)
        orientation = ORIENTATIONS[(ORIENTATIONS.index(orientation) + player * 2) % len(ORIENTATIONS)]
        orientation += offset

        return action_to_string(piece_type, index, orientation)


    def convert_player_perspective_action_to_real_action(self, player_action: str, player: int) -> str:
        """ Converts a player-perspective action in coordinance with the player's rotated observation of the board
        to the corresponding real action consumable by the actual environment.


        Parameters
        ----------
        player_action: str
            The player-perspective action.
        player : int
            The player to view this action from.

        Returns
        -------
        action : str
            The real action corresponding to the player-perspective action

        See Also
        --------
        blokus.blokus_env.convert_player_perspective_action_to_real_action
        blokus.blokus_env.BlokusEnv.next_state
            You can pass the result of this method to this.

        """

        if not player_action:
            return ""

        piece_type, index, orientation = string_to_action(player_action)

        index = tuple((np.matmul(PLAYER_OBSERVATION_TO_BOARD_ROTATION_MATRICES[player],
                                 (np.asarray(index) - 9.5)) + 9.5).astype(np.int32))

        orientation, offset = _separate_offset_from_orientation(orientation)
        orientation = ORIENTATIONS[(ORIENTATIONS.index(orientation) - player * 2) % len(ORIENTATIONS)]
        orientation += offset

        return action_to_string(piece_type, index, orientation)

    def valid_actions_dict(self, state: object, player: int) -> Dict[str, Dict[Tuple[int, int], List[str]]]:
        """ Valid actions for a specific state and player in the dictionary form {piece_type: {index: [orientation,]}}

        Parameters
        ----------
        state : object
            The current state to execute a game step from.
        player : int
            The player for which valid actions will be returned.

        Returns
        -------
        valid_actions_dict : Dict[str, Dict[Tuple[int], List[str]]]
            A dictionary of valid actions broken down into piece_type, index, and orientations.


        See Also
        --------
        blokus.blokus_env.action_to_string
        blokus.blokus_env.string_to_action
        blokus.blokus_env.valid_actions
        blokus.blokus_env.is_valid_action
        blokus.blokus_env.BlokusEnv.next_state
            If an action is valid, you can pass it to this method.

        Notes
        -----
        This method does not keep track of who's turn it is. That is up to the user.
        If the specified player can physically place a piece at a location, it will be returned as a valid action.

        """
        board, round_count, players = state
        current_player_object = players[player]
        return board.get_all_valid_moves(round_count=round_count,
                                         player_color=PLAYER_TO_COLOR[player],
                                         player_pieces=current_player_object.current_pieces)

    def is_valid_action(self, state: object, player: int, action: str) -> bool:
        """ Returns True if an action is valid for a specific player and state.

        (Does not validate rotated player-perspective actions)

        Parameters
        ----------
        state : object
            The current state to execute a game step from.
        player : int
            The player that would be executing the action.
        action : str
            The action in question

        Returns
        -------
        is_action_valid : bool
            whether this action is valid

        See Also
        --------
        blokus.blokus_env.action_to_string
        blokus.blokus_env.string_to_action
        blokus.blokus_env.valid_actions
        blokus.blokus_env.is_valid_action
        blokus.blokus_env.BlokusEnv.next_state
            If an action is valid, you can pass it to this method.


        Notes
        -----
        This method does not keep track of who's turn it is. That is up to the user.
        If a piece may be physically placed at the location suggest by the action,
        this method returns true, regardless of who just executed their turn or who should be going now.
        """

        if len(action) == 0:
            return False

        board, round_count, players = state
        piece_type, index, orientation = string_to_action(action)
        current_player = players[player]
        all_valid_moves = current_player.collect_moves(board, round_count)

        is_valid_move = False
        try:
            # print(all_valid_moves[piece_type][index])
            if orientation in all_valid_moves[piece_type][index]:
                is_valid_move = True
        except KeyError:
            pass

        return is_valid_move

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

        See Also
        --------
        blokus.blokus_client_env.BlokusClientEnv.wait_for_turn : also returns observations
        blokus.blokus_client_env.BlokusClientEnv.step : also returns observations

        Notes
        -----
        Observations returned here are of the same format as those given
        by :py:func:`blokus.blokus_client_env.BlokusClientEnv.step`.

        Observations are specific to individual players.
        Every observation is presented as if the player intended to receive it were actually player 0.
        This is done so that an RL agent only has to learn to perform moves that make player 0 win
        and other players lose.
        """

        pieces = np.zeros((4, 21), dtype=np.uint8)
        board, round_count, players = state
        board = [[_relative_player_id(player, COLOR_TO_PLAYER[pos]) for pos in row] for row in board.board_contents]
        rotated_board = _rotate_board_for_player_perspective(board=np.asarray(board), player=player)

        for p in players:

            color = p.player_color
            rel_player_id = _relative_player_id(current_player=player,
                                                absolute_player_num=COLOR_TO_PLAYER[color])

            for piece in p.current_pieces:
                pieces[rel_player_id, PIECE_NAME_TO_INDEX[piece]] = 1

        score = np.roll(np.array([p.player_score for p in players]), -player)

        return {'board': rotated_board, 'pieces': pieces, 'score': score, 'player': np.array([player])}
