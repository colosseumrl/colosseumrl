''' 
Author: Caleb Pitts
Date: 3/15/19

Summary:
- Contains computation methods that board.py uses to
  manage valid move seeks and piece placement.
- Methods use Numba with jit decorator that precompiles
  types and makes runtime faster than normal python.
'''
from numba import jit
import numpy as np
import math

# def dummy_jit(*args, **kwargs):
#     def dumdum(f):
#         return f
#     return dumdum
#
# jit = dummy_jit


#### METHODS FOR check_shifted() ####
@jit("UniTuple(int64, 2)(UniTuple(int64, 2), UniTuple(int64, 2), double)", nopython=True)  # "int(int64, ...)"
def rotate_by_deg(index, offset_point, angle):
    ''' Rotates each point on piece around the index by the given angle
    '''
    ox, oy = index
    px, py = offset_point

    new_x = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    new_y = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return int(round(new_x, 1)), int(round(new_y, 1))


@jit("UniTuple(int64, 2)(UniTuple(int64, 2), int64, int64)", nopython=True)
def flip_piece_x(index, x, y):
    ''' Takes the difference between index x and point x, then applies reverse
        difference to the index point. y stays the same
    '''
    return index[0] - (index[0] - x) * -1, y


@jit("UniTuple(int64, 2)(UniTuple(int64, 2), int64, int64)", nopython=True)
def flip_piece_y(index, x, y):
    ''' Takes the difference between index y and point y, then applies reverse
        difference to the index point. x stays the same
    '''
    return x, index[1] + (y - index[1]) * -1


@jit("UniTuple(int64, 2)(UniTuple(int64, 2), int64, int64, unicode_type)", nopython=True)
def rotate_piece(index, x_offset, y_offset, piece_orientation):
    ''' Description: Orients piece around the index point
        Parameters:
            index: int tuple that specifies the index coordinate on the board (the coord the piece will rotate around)
            offset: int tuple that specifies the offset from the index coord for the current cell
            piece_orientation: string specifying what new orientation you want the point at
        Returns:
            2 ints x and y that are the new rotated piece coords
    '''
    piece_orientation = piece_orientation[:-1]  # Takes out last character specifying the shift id (not needed in this method)
    x_offset += index[0]  # calculates the actual x coord on board
    y_offset += index[1]  # calculates the actual y coord on board

    if piece_orientation == "north":
        return rotate_by_deg(index, (x_offset, y_offset), math.radians(270))
    elif piece_orientation == "northwest":
        new_x, new_y = rotate_by_deg(index, (x_offset, y_offset), math.radians(270))
        return flip_piece_x(index, new_x, new_y)
    elif piece_orientation == "south":
        return rotate_by_deg(index, (x_offset, y_offset), math.radians(90))
    elif piece_orientation == "southeast":
        new_x, new_y = rotate_by_deg(index, (x_offset, y_offset), math.radians(90))
        return flip_piece_x(index, new_x, new_y)
    elif piece_orientation == "west":
        return rotate_by_deg(index, (x_offset, y_offset), math.radians(180))
    elif piece_orientation == "southwest":
        new_x, new_y = rotate_by_deg(index, (x_offset, y_offset), math.radians(180))
        return flip_piece_y(index, new_x, new_y)
    elif piece_orientation == "northeast":
        new_x, new_y = rotate_by_deg(index, (x_offset, y_offset), math.radians(0))
        return flip_piece_y(index, new_x, new_y)
    else:  # Default orientation (East)
        return rotate_by_deg(index, (x_offset, y_offset), math.radians(0))


@jit("boolean(int64[:, ::1], int64, int64, int64)", nopython=True)
def is_valid_adjacents(board_contents, y, x, player_color):
    ''' Description: Invalid coord if left, right, bottom, or top cell is the same color as the current player.
        Parameters:
            board_contents: 20 by 20 numpy matrix representing the current state of the board
            x: int x coord of the cell
            y: int y coord of the cell
            player_color: int representing current player color
        Returns:
            bool indicating whether the cell is a valid adjacent 
    '''
    valid_adjacent = True

    # Excludes top board edge from top cell check
    if y != 0:
        if board_contents[y - 1][x] == player_color:
            valid_adjacent = False
    # Excludes left board edge from left cell check
    if x != 0:
        if board_contents[y][x - 1] == player_color:
            valid_adjacent = False
    # Excludes bottom board edge from bottom cell check
    if y != 19:
        if board_contents[y + 1][x] == player_color:
            valid_adjacent = False
    # Excludes right board edge from right cell check
    if x != 19:
        if board_contents[y][x + 1] == player_color:
            valid_adjacent = False

    return valid_adjacent


@jit("boolean(int64[:, ::1], int64, int64, int64)", nopython=True)
def is_valid_cell(board_contents, x, y, player_color):
    ''' Description: If the cell x, y is empty, has no adjacent cells that are the same color,
                     and is not out of bounds of the 20x20 board, then the cell is valid 
                     to put a part of a piece on it.
        Parameters:
            board_contents: 20 by 20 numpy matrix representing the current state of the board
            x: int x coord of the cell
            y: int y coord of the cell
            player_color: int representing current player color
        Returns:
            bool indicating whether the cell is a valid cell
    '''
    # Out of bounds check
    if x < 0 or x >= 20 or y < 0 or y >= 20:
        return False
    # Checks if cell is empty and a valid adjacent
    if (board_contents[y][x] == 0 and is_valid_adjacents(board_contents, y, x, player_color)):
        return True
    else:
        return False


@jit("int64[:](int64[:, ::1], int64, UniTuple(int64, 2), unicode_type, int64[:, :, ::1])", nopython=True)
def check_shifted(board_contents, player_color, index, orientation, shifted_offsets):
    ''' Description: Shifts entire piece N times were N is how many cells the piece takes up. 
                     All shifted offsets are checked for the current orientation to see whether
                     the shifted set of offsets is a valid move. 
        Parameters:
            board_contents: 20 by 20 numpy matrix representing the current state of the board
            played_color: int representing current player color
            index: int tuple that specifies the index coordinate on the board (the coord the piece will rotate around)
            orientation: string specifying which orientation is being checked
            shifted_offsets: list of a list of tuples where each element in the main list represents a different set of coords
                             for a shifted piece.
        Returns:
            Returns the list of ints representing the shifted offsets ids where the piece can be
            placed at that set of shifted offsets
    '''
    shifted_ids = np.zeros(shifted_offsets.shape[0], np.int64)
    num_items = 0
    for shifted_id in range(shifted_offsets.shape[0]):  # Shift piece N times where N is the number of cells in the piece
        valid_placement = True
        for offset_id in range(shifted_offsets.shape[1]):
            offset = shifted_offsets[shifted_id, offset_id, :]
            if offset[0] == 0 and offset[1] == 0:  # No need to rotate coord since its the index and there is no offset
                if not is_valid_cell(board_contents, index[0], index[1], player_color):
                    valid_placement = False
            else:
                new_piece = rotate_piece(index, offset[0], offset[1], orientation)
                new_x = new_piece[0]
                new_y = new_piece[1]
                if not is_valid_cell(board_contents, new_x, new_y, player_color):
                    valid_placement = False
        if valid_placement:
            shifted_ids[num_items] = shifted_id
            num_items += 1

    return shifted_ids[:num_items]


#### METHODS FOR get_all_shifted_offsets() ####
@jit("int64[:, ::1](int64[:, ::1], unicode_type)", nopython=True)
def rotate_default_piece(offsets, orientation):
    ''' Description: Rotates the initial default piece orientation for shifting.
        Parameters:
            offsets: numpy array of tuples indicated all corresponding offsets to a specific piece type
            orientation: string indicating the orientation to rotate the offset pieces
        Returns:
            numpy list of all offsets for given orientation
    '''
    orientation_offsets_to_shift = np.zeros((len(offsets), 2), np.int64)

    for index in range(len(offsets)):
        if offsets[index][0] == 0 and offsets[index][1] == 0:
            orientation_offsets_to_shift[index, :] = (0, 0)
        else:
            new_coord = rotate_piece((0, 0), offsets[index][0], offsets[index][1], orientation + "!")  # adding dummy character to end since rotate ignores last character of orientation
            orientation_offsets_to_shift[index, :] = (new_coord[0], new_coord[1])

    return orientation_offsets_to_shift


@jit("int64[:, ::1](int64[:, ::1], int64)", nopython=True)
def shift_offsets(offsets, offset_id):
    ''' Description: Shifts the offsets so that the offset that corresponds to the offset_id is the new index
        Parameters:
            offsets: numpy array of tuples containing the coords of offsets needing do be shifted
            offset_id: identifies which coord becomes the new index in the list of offsets
        Returns:
            numpy array of tuples containing the newly shifted offsets
    '''
    shifted_offsets = np.zeros((len(offsets), 2), np.int64)

    if offset_id == 0:  # No need to shift the offsets for the default piece shape defined in the global space
        return offsets

    new_origin_y_diff = offsets[offset_id][0]
    new_origin_x_diff = offsets[offset_id][1]

    for index in range(len(offsets)):
        shifted_offsets[index, :] = (offsets[index][0] - new_origin_y_diff, offsets[index][1] - new_origin_x_diff)

    return shifted_offsets


@jit("int64[:, :, ::1](int64[:, ::1], unicode_type)", nopython=True)
def get_all_shifted_offsets(offsets, orientation):
    ''' Description: Compiles a list of all shifted offsets for a piece at a specific orientation.
                     Returns a numpy array, which is a list of a list of tuples which each contain 
                     a shifted offset. 
        Parameters:  
            offsets: numpy array of tuples containing the coords of offsets needing do be shifted
            orientation: string specifying the orientation that the shifts should take place
        Returns:
            list of all shifted orientations a piece can make at a given orientation

    '''
    orientation_offsets_to_shift = rotate_default_piece(offsets, orientation)
    shifted_offsets = np.zeros((len(offsets), len(orientation_offsets_to_shift), 2), np.int64)

    for offset_id in range(len(orientation_offsets_to_shift)):
        shifted_offsets[offset_id] = shift_offsets(orientation_offsets_to_shift, offset_id)

    return shifted_offsets
