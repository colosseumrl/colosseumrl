import numpy as np
import random

from matplotlib import cm
from . import TronGridEnvironment

import gym
from gym.spaces import Dict, Discrete, Box

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class TronRayEnvironment(MultiAgentEnv):
    action_space = Discrete(3)

    def __init__(self, board_size=15, num_players=4):
        self.env = TronGridEnvironment.create(board_size=board_size, num_players=num_players)
        self.state = None
        self.players = None

        self.observation_space = Dict({
            'board': Box(0, num_players, shape=(board_size, board_size)),
            'heads': Box(0, np.infty, shape=(num_players,)),
            'directions': Box(0, 4, shape=(num_players,)),
            'deaths': Box(0, num_players, shape=(num_players,))
        })

    def reset(self):
        self.state, self.players = self.env.new_state()
        return {str(i): self.env.state_to_observation(self.state, i) for i in range(self.env.num_players)}

    def step(self, action_dict):
        action_to_string = {
            0: 'forward',
            1: 'right',
            2: 'left'
        }

        actions = []

        for player in self.players:
            action = action_dict.get(str(player), 0)
            actions.append(action_to_string[action])

        self.state, self.players, rewards, terminal, winners = self.env.next_state(self.state, self.players, actions)

        num_players = self.env.num_players
        alive_players = set(self.players)

        observations = {str(i): self.env.state_to_observation(self.state, i) for i in range(num_players)}
        rewards = {str(i): rewards[i] for i in range(num_players)}
        dones = {str(i): i not in alive_players for i in range(num_players)}
        dones['__all__'] = terminal

        return observations, rewards, dones, {}


class SimpleAvoidAgent:
    def __init__(self, noise=0.1):
        self.noise = noise

    def __call__(self, env, observation):
        if random.random() <= self.noise:
            return random.choice(['forward', 'right', 'left'])

        board = observation['board']
        head = observation['heads'][0]
        direction = observation['directions'][0]

        board_size = board.shape[0]
        x, y = head % board_size, head // board_size

        # Check ahead. If it's clear, then take a step forward.
        nx, ny = env.next_cell(x, y, direction, board_size)
        if board[ny, nx] == 0:
            return 'forward'

        # Check a random direction. If it's clear, then go there.
        offset, action, backup = random.choice([(1, 'right', 'left'), (-1, 'left', 'right')])
        nx, ny = env.next_cell(x, y, (direction + offset) % 4, board_size)
        if board[ny, nx] == 0:
            return action

        # Otherwise, turn the opposite direction.
        return backup


class TronRaySinglePlayerEnvironment(gym.Env):
    action_space = Discrete(3)

    def __init__(self, board_size=15, num_players=4, spawn_offset=2, agent=SimpleAvoidAgent()):
        self.env = TronGridEnvironment.create(board_size=board_size, num_players=num_players)
        self.state = None
        self.players = None
        self.human_player = None
        self.spawn_offset = spawn_offset
        self.agent = agent

        self.render_viewer = None
        self.render_colors = cm.plasma(np.linspace(0.1, 0.9, num_players))
        self.render_colors = np.minimum(self.render_colors * 1.3, 1.0)

        self.observation_space = Dict({
            'board': Box(0, num_players, shape=(board_size, board_size)),
            'heads': Box(0, np.infty, shape=(num_players,)),
            'directions': Box(0, 4, shape=(num_players,)),
            'deaths': Box(0, num_players, shape=(num_players,))
        })

    def reset(self):
        self.state, self.players = self.env.new_state(spawn_offset=self.spawn_offset)
        self.human_player = self.players[0]

        return self._get_observation(self.human_player)

    def _get_observation(self, player):
        return self.env.state_to_observation(self.state, player)

    def step(self, action: int):
        human_player = self.human_player

        action_to_string = {
            0: 'forward',
            1: 'right',
            2: 'left'
        }

        actions = []
        for player in self.players:
            if player == human_player:
                actions.append(action_to_string[action])
            else:
                actions.append(self.agent(self.env, self._get_observation(player)))

        self.state, self.players, rewards, terminal, winners = self.env.next_state(self.state, self.players, actions)

        observation = self._get_observation(human_player)
        reward = rewards[human_player]
        done = human_player not in self.players

        return observation, reward, done, {}

    def render(self, mode='human'):
        # Constants
        screen_width = 500
        screen_height = 500
        border_space = 25
        grid_space_ratio = 6

        background_color = (0.14, 0.14, 0.14)
        blank_color = (0.85, 0.85, 0.85)
        board_size = self.env.N

        if self.render_viewer is None:
            from gym.envs.classic_control import rendering
            self.render_viewer = rendering.Viewer(screen_width, screen_height)
            gridx = (border_space, screen_width - border_space)
            gridy = (border_space, screen_height - border_space)

            background = rendering.FilledPolygon(
                [(0, 0), (screen_width, 0), (screen_width, screen_height), (0, screen_height)])
            background.set_color(*background_color)

            grid_size_x = (gridx[1] - gridx[0]) / board_size
            grid_size_y = (gridy[1] - gridy[0]) / board_size
            grid_space = min(grid_size_x, grid_size_y) / grid_space_ratio

            self.render_grid = []
            self.render_flatgrid = []

            for y in range(board_size):
                row = []
                for x in range(board_size):
                    startx = gridx[0] + x * grid_size_x + grid_space
                    endx = startx + grid_size_x - 2 * grid_space

                    starty = gridy[0] + y * grid_size_y + grid_space
                    endy = starty + grid_size_y - 2 * grid_space

                    box = rendering.FilledPolygon([(startx, starty), (endx, starty), (endx, endy), (startx, endy)])
                    box.set_color(*blank_color)
                    row.append(box)
                    self.render_flatgrid.append(box)

                self.render_grid.append(row)

            self.render_viewer.add_geom(background)
            for box in self.render_flatgrid:
                self.render_viewer.add_geom(box)

        if self.state is None: return None

        flat_board = self.state[0].ravel()
        heads = set(self.state[1])
        for idx in np.where(flat_board > 0)[0]:
            x = idx % board_size
            y = idx // board_size
            val = flat_board[idx] - 1
            factor = 1.0 if idx in heads else 0.6

            self.render_grid[y][x].set_color(*(factor * self.render_colors[val, :-1]))

        return self.render_viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.render_viewer:
            self.render_viewer.close()
            self.render_viewer = None

