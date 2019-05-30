''' 
Author: Caleb Pitts
Date: 3/15/19
 
Summary:
Handles state of hte board, piece placements, and valid move driver methods.

COLOR ENCODINGS
0 - Blank
1 - Red
2 - Blue
3 - Green
4 - Yellow
'''

from collections import defaultdict
from copy import deepcopy
import numpy as np
from . import computation as comp

# Stores structure of all playable pieces
# key: piece name:
# val: default offsets from index
PIECE_TYPES = {"monomino1": np.array([(0, 0)]),
               "domino1": np.array([(0, 0), (1, 0)]),
               "trominoe1": np.array([(0, 0), (1, 0), (1, 1)]),
               "trominoe2": np.array([(0, 0), (1, 0), (2, 0)]),
               "tetrominoes1": np.array([(0, 0), (1, 0), (0, 1), (1, 1)]),
               "tetrominoes2": np.array([(0, 0), (1, -1), (1, 0), (2, 0)]),
               "tetrominoes3": np.array([(0, 0), (1, 0), (2, 0), (3, 0)]),
               "tetrominoes4": np.array([(0, 0), (1, 0), (2, 0), (2, -1)]),
               "tetrominoes5": np.array([(0, 0), (1, 0), (1, -1), (2, -1)]),
               "pentominoe1": np.array([(0, 0), (0, -1), (1, 0), (2, 0), (3, 0)]),
               "pentominoe2": np.array([(0, 0), (0, -1), (0, 1), (1, 0), (2, 0)]),
               "pentominoe3": np.array([(0, 0), (0, -1), (0, -2), (1, -2), (2, -2)]),
               "pentominoe4": np.array([(0, 0), (1, 0), (1, -1), (2, -1), (3, -1)]),
               "pentominoe5": np.array([(0, 0), (0, 1), (1, 0), (2, 0), (2, -1)]),
               "pentominoe6": np.array([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]),
               "pentominoe7": np.array([(0, 0), (1, 0), (2, 0), (1, -1), (2, -1)]),
               "pentominoe8": np.array([(0, 0), (0, 1), (1, 0), (1, -1), (2, -1)]),
               "pentominoe9": np.array([(0, 0), (1, 0), (0, 1), (0, 2), (1, 2)]),
               "pentominoe10": np.array([(0, 0), (1, 0), (1, -1), (1, 1), (2, -1)]),
               "pentominoe11": np.array([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]),
               "pentominoe12": np.array([(0, 0), (1, 0), (1, -1), (2, 0), (3, 0)])}

# All possible piece orientations, listed in clockwise order
ORIENTATIONS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]

# Default starting corners for each player (0 to 3)
PLAYER_DEFAULT_CORNERS = [(0, 0), (19, 0), (0, 19), (19, 19)]

BOARD_TO_PLAYER_OBSERVATION_ROTATION_MATRICES = np.array([
    [[1, 0],
     [0, 1]],
    [[0, -1],
     [1, 0]],
    [[-1, 0],
     [0, -1]],
    [[0, 1],
     [-1, 0]]
], dtype=np.int32)


PLAYER_OBSERVATION_TO_BOARD_ROTATION_MATRICES = np.array([
    [[1, 0],
     [0, 1]],
    [[0, 1],
     [-1, 0]],
    [[-1, 0],
     [0, -1]],
    [[0, -1],
     [1, 0]]
], dtype=np.int32)

class Board:
    def __init__(self, copy_from_board=None):
        self.reset_board(copy_from_board)

    def reset_board(self, copy_from_board=None):
        ''' Creates empty 2-dimensional 20 by 20 numpy zeros array that represents a clean board
        '''
        if copy_from_board is not None:
            self.board_contents = deepcopy(copy_from_board.board_contents)
        else:
            self.board_contents = np.zeros((20, 20), dtype=int)

    def update_board(self, player_color, piece_type, index, piece_orientation, round_count, ai_game):
        ''' Takes index point and places piece_type on board
            index[0] = x coord
            index[1] = y coord
        '''
        self.player_color = player_color
        for offset in comp.shift_offsets(PIECE_TYPES[piece_type], int(piece_orientation[-1])):  # NEW: Shifts offset orientation according to last character in piece orientation
            if offset[0] == 0 and offset[1] == 0:
                self.place_piece(index[0] + offset[0], index[1] + offset[1])  # Orientation doesn't matter since (0, 0) is the reference point
            else:
                new_x, new_y = comp.rotate_piece(index, offset[0], offset[1], piece_orientation)
                self.place_piece(new_x, new_y)

    def place_piece(self, x, y):
        ''' Places piece on board by filling board_contents with the current player color
        '''
        self.board_contents[y][x] = self.player_color

    # def gather_empty_board_corners(self, corners_coords):
    #     ''' Checks what corners are still available to play in the first round of the game
    #     '''
    #     empty_corners = []
    #     for corner in corners_coords:
    #         if self.board_contents[corner[1]][corner[0]] == 0:
    #             empty_corners.append((corner[0], corner[1]))
    #     return empty_corners

    def gather_empty_corner_indexes(self, player_color):
        ''' Returns a list of tuples with the indexes of empty corner cells that connect to the player's color.
            The corner_index is not adjecent/touching any same color tiles on its sides beside its corners.
        '''
        empty_corner_indexes = []
        for row_num, row in enumerate(self.board_contents):
            for col_num, cell in enumerate(row):
                if cell == 0:  # Check if cell is empty
                    if self.check_valid_corner(self.board_contents, player_color, row_num, col_num):  # If cell lines up with any adjacent piece that is the same color, its an invalid move, otherwise valid
                        empty_corner_indexes.append((col_num, row_num))

        return empty_corner_indexes

    def check_valid_corner(self, board_contents, player_color, row_num, col_num):
        ''' Checks whether all adjacent pieces are a different color to the current player's color.
            Checks if any corner piece is the same color as the current player.
        '''
        if not comp.is_valid_adjacents(board_contents, row_num, col_num, player_color):  # Not a valid corner if adjacents are same color
            return False

        # Exclude top and right edge cases
        if row_num != 0 and col_num != 19:
            if board_contents[row_num - 1][col_num + 1] == player_color:
                return True

        # Exclude top and left cases
        if row_num != 0 and col_num != 0:
            if board_contents[row_num - 1][col_num - 1] == player_color:
                return True

        # Exclude bottom and right edge cases
        if row_num != 19 and col_num != 19:
            if board_contents[row_num + 1][col_num + 1] == player_color:
                return True

        # Exclude bottom and left cases
        if row_num != 19 and col_num != 0:
            if board_contents[row_num + 1][col_num - 1] == player_color:
                return True

        return False

    def check_orientation_shifts(self, player_color, piece_type, index, orientation):
        ''' DESCRIPTION: Shifts piece N times where N is how large the piece is. Each piece shift
                         is then checked to see whether it is a valid move.
            PARAMETERS: player_color: int indicating current player color
                        piece_type: string mapping to set of default offsets (defined in global space)
                        index: tuple representing index coords on board
                        orientation: string specifying the current orientation being checked
            RETURNS: List of all offset lists where a shift at that index and orientation is possible
        '''
        shifted_offsets = comp.get_all_shifted_offsets(PIECE_TYPES[piece_type], orientation)
        valid_shift_offsets = comp.check_shifted(self.board_contents, player_color, index, orientation, shifted_offsets)

        return valid_shift_offsets

    def get_all_valid_moves(self, round_count, player_color, player_pieces):
        ''' Gathers all valid moves on the board that meet the following criteria:
            - Index of selected piece touches same-colored corner of a piece
            - Player piece does not fall outside of the board
            - Player piece does not overlap any of their pieces or other opponent pieces
            - May lay adjacent to another piece as long as its another color
        '''
        if round_count == 0:  # If still first round of game..
            # empty_corner_indexes = self.gather_empty_board_corners([(0, 0), (19, 0), (0, 19), (19, 19)])
            empty_corner_indexes = [PLAYER_DEFAULT_CORNERS[player_color-1]]
        else:
            empty_corner_indexes = self.gather_empty_corner_indexes(player_color)

        all_valid_moves = {}
        for piece_type in player_pieces:                # Loop through all pieces the player currently has
            all_index_orientations = defaultdict(list)  # Valid indexes with their valid orientations dict created for every piece
            for index in empty_corner_indexes:          # Loop through all indexes where a piece can be placed
                for orientation in ORIENTATIONS:        # Loop through all possible orientations at an unshifted index
                    for shifted_id in self.check_orientation_shifts(player_color, piece_type, index, orientation):  # NEW: Loop through all shifted offsets of a particular orientation, if none valid, nothing gets added
                        all_index_orientations[index].append(orientation + str(shifted_id))                         # NEW: shifted_id is the cell(s) in piece where a shift is possible
            if len(list(all_index_orientations.keys())) > 0:  # If there are valid indexes for the piece type..
                all_valid_moves[piece_type] = all_index_orientations

        return all_valid_moves

    def decode_color(self, player_color):
        ''' Converts int representatio of player color to string representation.
        '''
        player_color_codes = {1: "R",
                              2: "B",
                              3: "G",
                              4: "Y"}

        return player_color_codes[player_color]

    def calculate_winner(self, players, round_count):
        ''' Returns the winner player color. 
        '''
        scores = []
        max_score = 0
        winner = "NONE"

        for current_player in players:
            scores.append((current_player.player_color, current_player.player_score))
            if current_player.player_score > max_score:
                max_score = current_player.player_score

        # print("FINAL SCORES:")
        for player_color, score in sorted(scores, key=lambda x: x[1]):
            # print(self.decode_color(player_color), score)
            if score == max_score:  # Prints all scores equal to the max score (accounts for ties)
                winner = self.decode_color(player_color)

        return winner
