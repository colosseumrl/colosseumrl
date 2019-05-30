''' 
Author: Caleb Pitts
Date: 3/15/19

Summary:
Keeps track of player score, inventory, and returns valid moves for that specific player.
'''

# Stores structure of all playable pieces
# key: piece name
# val: number of points associated with piece
GAME_PIECE_VALUES = {"monomino1": 1, "domino1": 2,
                     "trominoe1": 3, "trominoe2": 3,
                     "tetrominoes1": 4, "tetrominoes2": 4,
                     "tetrominoes3": 4, "tetrominoes4": 4,
                     "tetrominoes5": 4,
                     "pentominoe1": 5, "pentominoe2": 5,
                     "pentominoe3": 5, "pentominoe4": 5,
                     "pentominoe5": 5, "pentominoe6": 5,
                     "pentominoe7": 5, "pentominoe8": 5,
                     "pentominoe9": 5, "pentominoe10": 5,
                     "pentominoe11": 5, "pentominoe12": 5}


class AI:
    def __init__(self, board_state, color):
        self.player_score = 0
        self.player_color = color
        self.current_pieces = list(GAME_PIECE_VALUES.keys())  # Gives all piece names to player when game starts

    def collect_moves(self, board, round_count):
        ''' Collects all valid moves for this player from the current state of the board
        '''
        return board.get_all_valid_moves(round_count, self.player_color, self.current_pieces)

    def check_moves(self, board, round_count):
        ''' Checks whether player has at least one valid move before prompting player for a move.
        '''
        self.all_valid_moves = board.get_all_valid_moves(round_count, self.player_color, self.current_pieces)
        if len(list(self.all_valid_moves.keys())) == 0:  # If no valid moves available for this player, return FALSE
            return False
        return True

    def update_player(self, piece_type):
        ''' Keeps track of player's inventory and score as piece type has been played
        '''
        self.current_pieces.remove(piece_type)  # Remove played piece from player's inventory

        if len(self.current_pieces) == 0 and piece_type == "monomino1":  # Additional 20 points if last piece played is a monomino
            self.player_score += 20
        elif len(self.current_pieces) == 0 and piece_type != "monomino1":  # Additional 15 points if all pieces have been played
            self.player_score += 15

        self.player_score += GAME_PIECE_VALUES[piece_type]  # Continue assignment of corresponding points for played piece regardless of the additional points cases
