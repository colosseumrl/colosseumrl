import numpy as np
from typing import Dict, Tuple, List, Union
from colosseumrl.BaseEnvironment import BaseEnvironment, SimpleConfigParser


class RPSTwoPlayerEnvironment(BaseEnvironment):
    parser = SimpleConfigParser((float, 1), (float, -1), (float, 0))

    action_map = {
        "R": 0,
        "P": 1,
        "S": 2
    }

    def __init__(self, config: str = None):
        super().__init__(config)
        self.win_reward, self.loss_reward, self.tie_reward = self.parser.parse(config)
        rw, rl, rt = self.win_reward, self.loss_reward, self.tie_reward

        # Create the empirical game matrix of the game
        self.game_matrix = np.array([
            [[rt, rt], [rl, rw], [rw, rl]],
            [[rw, rl], [rt, rt], [rl, rw]],
            [[rl, rw], [rw, rl], [rt, rt]]
        ])

        self.players = list(range(2))

    @classmethod
    def create(cls, win_reward: float = 1.0, loss_reward: float = -1.0, tie_reward: float = 0.0):
        return cls(cls.parser.store(win_reward, loss_reward, tie_reward))

    @property
    def min_players(self) -> int:
        return 2

    @property
    def max_players(self) -> int:
        return 2

    @staticmethod
    def observation_names() -> List[str]:
        return ['game_matrix']

    @property
    def observation_shape(self) -> Dict[str, tuple]:
        return {'game_matrix': (3, 3, 2)}

    def new_state(self, num_players: int = None) -> Tuple[object, List[int]]:
        return np.copy(self.game_matrix), self.players.copy()

    def next_state(self, state: object, players: [int], actions: [str]) -> Tuple[
        object, List[int], List[float], bool, Union[List[int], None]]:
        action_indices = tuple(self.action_map[action] for action in actions)
        rewards = state[action_indices]
        winners = np.where(rewards == np.max(rewards))[0]

        return state, players, rewards, True, list(winners)

    def valid_actions(self, state: object, player: int) -> [str]:
        return ["R", "P", "S"]

    def is_valid_action(self, state: object, player: int, action: str) -> bool:
        return action in ["R", "P", "S"]

    def state_to_observation(self, state: object, player: int) -> Dict[str, np.ndarray]:
        return {'game_matrix': np.copy(state)}