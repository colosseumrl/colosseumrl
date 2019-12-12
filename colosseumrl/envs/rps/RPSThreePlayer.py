import numpy as np
from typing import Dict, Tuple, List, Union
from .RPSTwoPlayer import RPSTwoPlayerEnvironment


class RPSThreePlayerEnvironment(RPSTwoPlayerEnvironment):
    def __init__(self, config: str = None):
        super().__init__(config)
        rw, rl, rt = self.win_reward, self.loss_reward, self.tie_reward

        self.players = list(range(3))

        # Create the empirical game matrix of the game
        self.game_matrix = np.array([[
            # Player 1 Rock
            # ----------------------------------------------------------------
            # P3 Rock       P3 Paper      P3 Scissors
            [[rt, rt, rt], [rl, rl, rw], [rl, rl, rw]],  # Player 2 Rock
            [[rl, rw, rl], [rw, rl, rl], [rt, rt, rt]],  # Player 2 Paper
            [[rl, rw, rl], [rt, rt, rt], [rw, rl, rl]]   # Player 2 Scissors

            ], [
            # Player 1 Paper
            # ----------------------------------------------------------------
            # P3 Rock       P3 Paper      P3 Scissors
            [[rw, rl, rl], [rl, rw, rl], [rt, rt, rt]],  # Player 2 Rock
            [[rl, rl, rw], [rt, rt, rt], [rl, rl, rw]],  # Player 2 Paper
            [[rt, rt, rt], [rl, rw, rl], [rw, rl, rl]]   # Player 2 Scissors

            ], [
            # Player 1 Scissors
            # ----------------------------------------------------------------
            # P3 Rock       P3 Paper      P3 Scissors
            [[rw, rl, rl], [rt, rt, rt], [rl, rw, rl]],  # Player 2 Rock
            [[rt, rt, rt], [rw, rl, rl], [rl, rw, rl]],  # Player 2 Paper
            [[rl, rl, rw], [rl, rl, rw], [rt, rt, rt]]   # Player 2 Scissors
        ]])

    @property
    def min_players(self) -> int:
        return 3

    @property
    def max_players(self) -> int:
        return 3

    @property
    def observation_shape(self) -> Dict[str, tuple]:
        return {'game_matrix': (3, 3, 3, 3)}