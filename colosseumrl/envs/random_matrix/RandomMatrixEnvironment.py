import numpy as np
from typing import Dict, Tuple, List, Union
from colosseumrl.BaseEnvironment import BaseEnvironment, SimpleConfigParser


class RandomMatrixEnvironment(BaseEnvironment):
    parser = SimpleConfigParser((int, 2), (int, 2), (int, None))

    def __init__(self, config: str = None):
        super(BaseEnvironment, self).__init__()

        self.num_players, self.num_strategies, seed = self.parser.parse(config)

        self.random_state = np.random.RandomState(seed)
        self.game_matrix_shape = tuple([self.num_strategies for _ in range(self.num_players)] + [self.num_players])
        self.game_matrix = self.random_state.rand(*self.game_matrix_shape)

    @classmethod
    def create(cls, num_players: int = 2, num_strategies: int = 2, seed: int = None):
        return cls(cls.parser.store(num_players, num_strategies, seed))

    @property
    def min_players(self) -> int:
        return self.num_players

    @property
    def max_players(self) -> int:
        return self.num_players

    @staticmethod
    def observation_names() -> List[str]:
        return ["game_matrix"]

    @property
    def observation_shape(self) -> Dict[str, tuple]:
        return {"game_matrix": self.game_matrix_shape}

    def new_state(self, num_players: int = None) -> Tuple[np.ndarray, np.ndarray]:
        return self.game_matrix.copy(), np.arange(self.num_players)

    def next_state(self, state: np.ndarray, players: np.ndarray, actions: [str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Union[List[int], None]]:
        actions = tuple(int(action) for action in actions)
        rewards = state[actions]

        return state, players, rewards, True, [int(np.argmax(rewards))]

    def valid_actions(self, state: object, player: int) -> [str]:
        return [str(i) for i in range(self.num_strategies)]

    def is_valid_action(self, state: object, player: int, action: str) -> bool:
        return int(action) < self.num_strategies

    def state_to_observation(self, state: np.ndarray, player: int) -> Dict[str, np.ndarray]:
        return {"game_matrix": state}
