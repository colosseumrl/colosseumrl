from typing import Tuple, List, Dict

from .ai import AI
from .BlokusEnvironment import start_gui, terminate_gui, display_board
from .board import Board
from colosseumrl.ClientEnvironment import ClientEnvironment


class BlokusClientEnvironment(ClientEnvironment):

    def __init__(self, *args, **kwargs):
        self._gui_is_active = False
        super().__init__(*args, **kwargs)

    def __del__(self):
        if self._gui_is_active:
            terminate_gui()

    def render(self, state: Tuple[int, Tuple[Board, int, List[AI]]], player_num: int, winners: List[int]):
        r"""A one-line summary that does not use variable names or the
        function name.

        """
        if not self._gui_is_active:
            start_gui()
            display_board(state, player_num, winners)
        display_board(state, player_num, winners)

    # TODO: random_valid_action_strings will be deleted because it isn't random and doesn't conform the standardized api
    # def random_valid_action_string(self, state: Tuple[int, Tuple[Board, int, List[AI]]], player_num: int) -> str:
    #     _, inner_state = state
    #     board, round_count, players = inner_state
    #     player = players[player_num]
    #
    #     all_valid_moves = player.collect_moves(board, round_count)
    #     # print("All possible moves for you: {}".format(all_valid_moves))
    #     # input('press enter to play random move:')
    #
    #     if len(all_valid_moves.keys()) != 0:
    #         random_indexes = random.sample(all_valid_moves.items(), 1)
    #         piece_type = random_indexes[0][0]
    #         index = list(all_valid_moves[piece_type].keys())[0]
    #         orientation = all_valid_moves[piece_type][index][0]
    #
    #         string_action = action_to_string(piece_type, index, orientation)
    #     else:
    #         string_action = ""
    #
    #     return string_action

    def valid_actions_dict(self) -> Dict[str, Dict[Tuple[int], List[str]]]:
        """ Valid actions for a specific state in the dictionary form {piece_type: {index: [orientation]}}"""
        return self._server_environment.valid_actions_dict(state=self.full_state, player=self._player.number)