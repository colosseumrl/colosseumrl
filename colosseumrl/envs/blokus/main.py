''' 
Author: Caleb Pitts
Date: 3/15/19
'''

from . import board
from . import ai
from . import gui
import itertools
import random
import sys
import time


def welcome():
    ''' Displays welcome message to console
    '''
    print("=======================")
    print("= Welcome to Blokus! =")
    print("=======================")
    print()


def get_show_command():
    ''' Collects sys arg to display board or not while a game cycle is running
    '''
    try:
        show_board = sys.argv[1]
        if show_board == "show":
            return True
        else:
            print("MSG: Bad show board input. Use 'show' arg to display board while a game cycle is running.")
            print("MSG: Running cycle without showing board...\n")
            return False
    except IndexError:
        return False


def initialize_players(current_board):
    red = ai.AI(current_board, 1)
    blue = ai.AI(current_board, 2)
    green = ai.AI(current_board, 3)
    yellow = ai.AI(current_board, 4)
    all_players = [red, blue, green, yellow]

    return all_players


def run_game_cycle(current_board, show_board, all_players):
    ''' Runs one complete game cycle where ai picks random move from list of available moves
    '''
    total_start = time.time()
    round_count = 0
    player_count = 0
    num_players = len(all_players)
    players_with_no_moves = 0
    round_time = 0

    if show_board:
        gui.start_gui()

    for current_player in itertools.cycle(all_players):
        all_valid_moves = current_player.collect_moves(current_board, round_count)  # change: passed in current_board

        if show_board:
            gui.display_board(current_board.board_contents, current_player, all_players, round_count)  # I don't know why i need to put this here

        if len(list(all_valid_moves.keys())) > 0:  # If no valid moves available for this player.
            players_with_no_moves = 0  # Reset to zero if at least one player can make a move

            if show_board:
                gui.display_board(current_board.board_contents, current_player, all_players, round_count)

            # Get all valid moves and ai decides on which one to make
            random_indexes = random.sample(all_valid_moves.items(), 1)
            piece_type = random_indexes[0][0]
            index = list(all_valid_moves[piece_type].keys())[0]
            orientation = all_valid_moves[piece_type][index][0]

            # ai chooses move
            current_board.update_board(current_player.player_color, piece_type, index, orientation, round_count, True)
            current_player.update_player(piece_type)  # Updates ai

            if player_count == 0:  # Stop ai game and look mechanism
                end = time.time()
                print("Time For AI Round ", round_count, ": ", round(round_time, 2), " seconds.", sep="")
                # x = input()  # Stops each round to observe ai game visually. Disable by commenting out this line
                start = time.time()
        else:
            players_with_no_moves += 1

        player_count += 1

        if player_count == num_players:  # Increment round count each time last player's turn is done.
            round_count += 1
            end = time.time()
            round_time = end - start
            player_count = 0

        if players_with_no_moves == 4:  # If 4 players with no moves, end game
            break

    winner = current_board.calculate_winner(all_players, round_count)
    print("Winner(s):", winner)
    total_end = time.time()
    print("Game took:", round(total_end - total_start, 2), "seconds.")


def main():
    show_board = get_show_command()
    welcome()

    while True:
        current_board = board.Board()
        all_players = initialize_players(current_board)
        run_game_cycle(current_board, show_board, all_players)

        quit = input("\nWould you like to run another game cycle? ([y]/n): ").upper().strip()
        if show_board:
            gui.terminate_gui()
        if quit == "N":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
