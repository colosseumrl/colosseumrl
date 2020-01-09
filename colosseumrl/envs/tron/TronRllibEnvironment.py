import numpy as np

from gym.spaces import Dict, Discrete, Box, Space

from colosseumrl import BaseEnvironment
from colosseumrl.envs.wrappers import RllibWrapper
from . import TronGridEnvironment


class TronRllibEnvironment(RllibWrapper):
    def create_env(self, *args, **kwargs) -> BaseEnvironment:
        return TronGridEnvironment.create(*args, **kwargs)

    def create_observation_space(self, *args, **kwargs) -> Space:
        num_players = self.env.num_players
        board_size = self.env.N

        return Dict({
            'board': Box(0, num_players, shape=(board_size, board_size)),
            'heads': Box(0, np.infty, shape=(num_players,)),
            'directions': Box(0, 4, shape=(num_players,)),
            'deaths': Box(0, num_players, shape=(num_players,))
        })

    def create_action_space(self, *args, **kwargs) -> Space:
        return Discrete(3)

    def create_done_dict(self, state, players, rewards, terminal, action_dict):
        alive_players = set(map(str, players))
        return {player: (player not in alive_players) or terminal for player in action_dict}

    def action_map(self, action):
        if action == 0:
            return 'forward'
        elif action == 1:
            return 'right'
        else:
            return 'left'
