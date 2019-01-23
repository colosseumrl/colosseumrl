from typing import Tuple
import numpy as np
from base_environment import BaseEnvironment


class TestGame(BaseEnvironment):

    @property
    def min_players(self) -> int:
        return 2

    @property
    def max_players(self) -> int:
        return 2

    @property
    def state_shape(self) -> tuple:
        return 1,

    @property
    def observation_shape(self):
        return 1,

    def new_state(self, num_players = 1) -> np.ndarray:
        return np.random.randint(0, 100, (1,))

    def add_player(self, state: np.ndarray):
        return state

    def next_state(self, state: np.ndarray, player: int, action: str) -> Tuple[np.ndarray, float, bool, int]:
        action = int(action)
        distance = -np.abs(action - state[0])
        print("action: {} state: {} distance: {}".format(action, state, distance))
        if action == state[0]:
            return state, distance, True, player
        return state, distance, False, -1

    def valid_actions(self, state: np.ndarray) -> [str]:
        return map(lambda x: str(x), np.arange(100))

    def is_valid_action(self, state: np.ndarray, action: str) -> bool:
        return action in self.valid_actions(state)

    def state_to_observation(self, state: np.ndarray, player: int) -> np.ndarray:
        return state
