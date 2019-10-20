import numpy as np
import random

from .TronGridEnvironment import TronGridEnvironment
from .TronRender import TronRender

import gym
from gym.spaces import Dict, Discrete, Box

# from ray.rllib.env.multi_agent_env import MultiAgentEnv


class TronRayEnvironment():
    action_space = Discrete(3)

    def __init__(self, board_size=15, num_players=4):
        self.env = TronGridEnvironment.create(board_size=board_size, num_players=num_players)
        self.state = None
        self.players = None

        self.renderer = TronRender(board_size, num_players)

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

        observations = {str(i): self.env.state_to_observation(self.state, i) for i in map(int, action_dict.keys())}
        rewards = {str(i): rewards[i] for i in map(int, action_dict.keys())}
        dones = {str(i): i not in alive_players for i in map(int, action_dict.keys())}
        dones['__all__'] = terminal

        return observations, rewards, dones, {}

    def render(self, mode='human'):
        if self.state is None:
            return None

        return self.renderer.render(self.state, mode)

    def close(self):
        self.renderer.close()


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

        self.renderer = TronRender(board_size, num_players)

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
        if self.state is None:
            return None

        return self.renderer.render(self.state, mode)

    def close(self):
        self.renderer.close()

