''' 
Author: Caleb Pitts
Date: 3/15/19

Summary:
Utlizes Pygame to show pieces getting placed on board as a visual cue. 
When enabled, the gui slows computation time signifigantly. We recomend you 
disable the gui for agent training. 
'''

import sys
import contextlib
with contextlib.redirect_stdout(None):  # disables pygame welcome message
    import pygame as pg

clock = pg.time.Clock()

COLORS = {1: (220, 20, 60),
          2: (30, 144, 255),
          3: (50, 205, 50),
          4: (255, 255, 102),
          0: (220, 220, 220)}


def decode_color_rgb(x, y, board_contents):
    ''' Returns rgb code for coded cell color at coord (x, y) in board contents
    '''
    return COLORS[board_contents[y][x]]


def decode_color(player_color):
    ''' Converts int representatio of player color to string representation.
    '''
    player_color_codes = {1: "R",
                          2: "B",
                          3: "G",
                          4: "Y"}

    return player_color_codes[player_color]


def setup_texts(current_player, players, round_count):
    ''' Sets up texts to be displayed on the gui board.
    '''
    player_text = "Player: " + decode_color(current_player.player_color)
    round_text = "Round: " + str(round_count)
    scores_text = "SCORES:  R = " + str(players[0].player_score) + " | B = " + str(players[1].player_score) + " | G = " + str(players[2].player_score) + " | Y = " + str(players[3].player_score)

    titlefont = pg.font.SysFont('Helvetica', 30)
    gamefont = pg.font.SysFont('Helvetica', 20)

    titlesurface = titlefont.render('Blokus', True, (255, 255, 255))
    player_indicator = gamefont.render(player_text, True, (255, 255, 255))
    round_indicator = gamefont.render(round_text, True, (255, 255, 255))
    scores_indicator = gamefont.render(scores_text, True, (255, 255, 255))

    return titlesurface, player_indicator, round_indicator, scores_indicator


def prep_cells(board_contents):
    ''' Preps cells to be drawn on gui board corresponding to the board contents. 
    '''
    rectangles = []
    height = 20
    width = 20
    size = 40

    for y in range(height):
        for x in range(width):
            rect = pg.Rect(x * (size + 1), y * (size + 1) + 55, size, size)
            color = decode_color_rgb(x, y, board_contents)
            rectangles.append((rect, color))

    return rectangles


def display_board(board_contents, current_player, players, round_count, winners=None):
    ''' Displays board contents and other game-related stats to a gui using pygame.
    '''
    screen = pg.display.set_mode((819, 960))

    titlesurface, player_indicator, round_indicator, scores_indicator = setup_texts(current_player, players, round_count)

    screen.blit(titlesurface, (355, 13))
    screen.blit(player_indicator, (10, 20))
    screen.blit(round_indicator, (700, 20))
    screen.blit(scores_indicator, (10, 895))

    rectangles = prep_cells(board_contents)
    for rect, color in rectangles:
        pg.draw.rect(screen, color, rect)

    winner_indicator = None

    # Displays winner in final state of board.
    # if winner is not None:
    #     print("FOUND WINNER", winner)
    #     scores_text = "WINNER: " + winner
    #     winner_font = pg.font.SysFont('Helvetica', 20)
    #     winner_indicator = winner_font.render(scores_text, True, (255, 255, 255))
    #     screen.blit(winner_indicator, (10, 920))  # Pop-up block appears when winner determined
    #     pg.display.update()

    pg.display.flip()
    clock.tick(60)


def start_gui():
    ''' Initializies pygame gui window
    '''
    pg.display.init()
    pg.display.set_caption('Blokus Game - Visual Cue')
    pg.font.init()


def terminate_gui():
    ''' Closes pygame gui window
    '''
    pg.display.quit()
    pg.quit()
