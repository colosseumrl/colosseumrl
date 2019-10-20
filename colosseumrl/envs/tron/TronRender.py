import numpy as np
from typing import Tuple
from matplotlib import cm

class TronRender:
    BACKGROUND_COLOR = (0.14, 0.14, 0.14)
    BLANK_COLOR = (0.85, 0.85, 0.85)
    LOSS_COLOR = (0.3, 0.1, 0.1)
    WIN_COLOR = (0.1, 0.3, 0.1)

    def __init__(self, board_size: int, num_players: int,
                 window_size: Tuple[int, int] = (600, 600),
                 outside_border: int = 25,
                 grid_space_ratio: float = 6,
                 winner_player: int = None):
        self.board_size = board_size
        self.window_size = window_size
        self.outside_border = outside_border
        self.grid_space_ratio = grid_space_ratio

        self.winner_player = winner_player
        if winner_player is not None:
            self.other_players = (np.arange(num_players - 1) + winner_player + 1) % num_players

        self.colors = cm.plasma(np.linspace(0.1, 0.9, num_players))
        self.colors = np.minimum(self.colors * 1.3, 1.0)

        # Rendering Objects
        self.viewer = None
        self.grid = None
        self.flat_grid = None
        self.background = None

    def start(self):
        from gym.envs.classic_control import rendering
        
        screen_width, screen_height = self.window_size
        border_space = self.outside_border
        grid_space_ratio = self.grid_space_ratio
        board_size = self.board_size

        self.viewer = rendering.Viewer(screen_width, screen_height)

        # Create Background polygon for controlling background color
        self.background = rendering.FilledPolygon([(0, 0), (screen_width, 0), (screen_width, screen_height), (0, screen_height)])
        self.background.set_color(*self.BACKGROUND_COLOR)

        # Create parameters for the grid of board cells
        gridx = (border_space, screen_width - border_space)
        gridy = (border_space, screen_height - border_space)
        grid_size_x = (gridx[1] - gridx[0]) / board_size
        grid_size_y = (gridy[1] - gridy[0]) / board_size
        grid_space = min(grid_size_x, grid_size_y) / grid_space_ratio

        # Create grid of polygons that will consist of our board
        self.grid = []
        self.flat_grid = []

        for y in range(board_size):
            row = []
            for x in range(board_size):
                startx = gridx[0] + x * grid_size_x + grid_space
                endx = startx + grid_size_x - 2 * grid_space

                starty = gridy[0] + y * grid_size_y + grid_space
                endy = starty + grid_size_y - 2 * grid_space

                box = rendering.FilledPolygon([(startx, starty), (endx, starty), (endx, endy), (startx, endy)])
                box.set_color(*self.BLANK_COLOR)
                row.append(box)
                self.flat_grid.append(box)

            self.grid.append(row[::-1])

        self.viewer.add_geom(self.background)
        for box in self.flat_grid:
            self.viewer.add_geom(box)

    def close(self):
        if self.viewer:
            self.viewer.close()

            self.viewer = None
            self.background = None
            self.grid = None
            self.flat_grid = None

    def render(self, state, mode='human'):
        if self.viewer is None:
            self.start()

        flat_board = state[0].ravel()
        heads = set(state[1])
        deaths = state[3]

        for idx in np.where(flat_board > 0)[0]:
            x = idx % self.board_size
            y = idx // self.board_size
            val = flat_board[idx] - 1
            factor = 1.0 if idx in heads else 0.6

            self.grid[y][x].set_color(*(factor * self.colors[val, :-1]))

        if self.winner_player is not None:
            if deaths[self.winner_player] > 0:
                self.background.set_color(*self.LOSS_COLOR)
            else:
                if all(deaths[player] > 0 for player in self.other_players):
                    self.background.set_color(*self.WIN_COLOR)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_observation(self, observation, mode='human'):
        state = [observation['board'], observation['heads'], None, observation['deaths']]
        return self.render(state, mode='human')
