import numpy as np
from typing import Dict, Tuple, List, Union
from dill import dumps, loads
from time import time

from rlcompetition.BaseEnvironment import BaseEnvironment
from .CyTronGrid import next_state_inplace, relative_player_inplace


def CreateTronGridConfig(*args) -> str:
    raw_string = "{};" * len(args)
    return raw_string[:-1].format(*args)


def ParseTronGridConfig(config: str):
    if len(config) == 0:
        return 20, 4, -1, False

    def parse(inp: str):
        try:
            return int(inp)
        except ValueError:
            return inp.lower() == "true"

    options = list(map(parse, config.split(";")))
    if len(options) == 1:
        options.append(4)
    if len(options) == 2:
        options.append(-1)
    if len(options) == 3:
        options.append(False)
    return options


class TronGridEnvironment(BaseEnvironment):
    STRING_TO_ACTION = {
        "": 0,
        "forward": 0,
        "right": 1,
        "left": -1,
    }

    @staticmethod
    def create(board_size: int = 20,
               num_players: int = 4,
               observation_window: int = -1,
               remove_on_death: bool = False) -> "TronGridEnvironment":
        """ Secondary constructor with explicit options for creating the environment
        """
        return TronGridEnvironment(CreateTronGridConfig(board_size,
                                                        num_players,
                                                        observation_window,
                                                        remove_on_death))

    def __init__(self, config: str = ""):
        super().__init__(config)
        board_size, num_players, observation_window, remove_on_death = ParseTronGridConfig(config)

        self.N = board_size
        self.num_players = num_players
        self.observation_window = observation_window
        self.fully_observable = observation_window < 0
        self.remove_on_death = remove_on_death

        self.player_array = np.arange(num_players)
        self.move_array = ['forward', 'right', 'left']

    def __repr__(self):
        print("Tron Finite Grid Environment")
        print("="*50)
        print("\tSize: {}x{}".format(self.N, self.N))
        print("\tNumber of players: {}".format(self.num_players))
        print("\tFully Observable: {}".format("Yes" if self.fully_observable else "No"))
        print("\tRemove old players: {}".format("Yes" if self.remove_on_death else "No"))
        print("-"*50)

    @property
    def min_players(self) -> int:
        return self.num_players

    @property
    def max_players(self) -> int:
        return self.num_players

    @staticmethod
    def observation_names() -> List[str]:
        return ["board", "heads", "directions", "deaths"]

    @property
    def observation_shape(self) -> Dict[str, tuple]:
        return {
            "board": (self.N, self.N),
            "heads": (self.num_players, ),
            "directions": (self.num_players, ),
            "deaths": (self.num_players, )
        }

    def new_state(self, num_players: int = None) -> Tuple[object, List[int]]:
        num_players = self.num_players if num_players is None else num_players
        assert num_players == self.num_players, "Do not change the number of players from the game configuration."

        # Generate the Starting configuration
        # TODO Make the starting points fair and spread out
        np.random.seed(int(time()))
        board = np.zeros((self.N, self.N), dtype=np.int64)
        heads = np.random.choice(self.N * self.N, size=self.num_players, replace=False)
        directions = np.random.randint(0, 4, size=num_players, dtype=np.int64)
        deaths = np.zeros(self.num_players, dtype=np.int64)

        # Set up the initial board
        board.ravel()[heads] = self.player_array + 1

        return (board, heads, directions, deaths), self.player_array

    def next_state(self, state: object, players: [int], actions: [str]):
        board, heads, directions, deaths = state

        # Convert the move strings to move indices for c++
        moves = np.fromiter((self.STRING_TO_ACTION[a] for a in actions), dtype=np.int64, count=len(actions))

        # Make a copy of the state since we operate in-place
        new_board = np.copy(board)
        new_heads = np.copy(heads)
        new_directions = np.copy(directions)
        new_deaths = np.copy(deaths)

        # Execute the move
        next_state_inplace(new_board, new_heads, new_directions, new_deaths, moves)

        # Reduce players to the ones still alive
        new_players = np.where(new_deaths == 0)[0]

        # Make rewards be whether or not you lived or died
        rewards = -2 * (new_deaths > 0) + 1

        # Terminal is if everyone or everyone except one has died
        terminal = (new_deaths > 0).sum() >= self.num_players - 1

        # Winner is the final player or nobody if tie
        winners = new_players if terminal else None

        return (new_board, new_heads, new_directions, new_deaths), new_players, rewards, terminal, winners

    def valid_actions(self, state: object, player: int) -> [str]:
        return self.move_array

    def is_valid_action(self, state: object, player: int, action: str) -> bool:
        return True

    def state_to_observation(self, state: object, player: int) -> Dict[str, np.ndarray]:
        board, heads, directions, deaths = state

        # Adjust board to reflect relative player number
        # i.e. observing player always sees themselves as player 1
        observation = board.copy()
        relative_player_inplace(observation, self.num_players, player + 1)

        rolled_idx = (np.arange(self.num_players) + player) % self.num_players

        heads = heads[rolled_idx]
        deaths = deaths[rolled_idx]
        directions = directions[rolled_idx]

        # Fully observable
        if self.fully_observable:
            return {
                "board": observation,
                "heads": heads,
                "directions": directions,
                "deaths": deaths
            }

        # Partially Observable
        # TODO Make this work
        # TODO Make it so that you can see far ahead but only a bit to the side and back
        else:
            head = heads[0]
            headx = head % self.N
            heady = head // self.N
            delta = self.observation_window

            return {
                "board": observation[heady - delta:heady + delta, headx - delta:headx + delta],
                "heads": heads,
                "deaths": deaths
            }

    @staticmethod
    def serializable() -> bool:
        """ Whether or not this class supports serialization of the state."""
        return False

    @staticmethod
    def serialize_state(state: object) -> bytearray:
        return dumps(state)

    @staticmethod
    def deserialize_state(serialized_state: bytearray) -> object:
        return loads(serialized_state)

