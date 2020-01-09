import numpy as np
from typing import Dict, Tuple, List
from dill import dumps, loads
from time import time
from itertools import starmap
from collections import Counter

from colosseumrl.BaseEnvironment import BaseEnvironment
from .CyTronGrid import next_state_inplace, relative_player_inplace


def create_tron_config(*args) -> str:
    """ Convert a list of parameters into a serialized tron grid string.
    Parameters
    ----------
    args
        All of the arguments into tron grid as star args.

    Returns
    -------
    str
        Serialized string
    """
    raw_string = "{};" * len(args)
    return raw_string[:-1].format(*args)


def parse_tron_config(config: str) -> Tuple:
    """ Convert a serialized configuration string into the list of options to tron grid.

    Parameters
    ----------
    config : str
        Config string in the form "{};{};...;{}"

    Returns
    -------
    List
        A list of options into tron grid environment.

    """
    if len(config) == 0:
        return 19, 4, -1, False

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
    def create(board_size: int = 19,
               num_players: int = 4,
               observation_window: int = -1,
               remove_on_death: bool = False) -> "TronGridEnvironment":
        """ Secondary constructor with explicit options for creating the environment

        Parameters
        ----------
        board_size : int
            This will specify the square size of the playing grid.
        num_players : int
            Number of active players in the game.
        observation_window : -1
            Current not used
        remove_on_death : bool
            Whether or not to remove the player and their associated walls when they are eliminated.
        """
        return TronGridEnvironment(create_tron_config(board_size,
                                                      num_players,
                                                      observation_window,
                                                      remove_on_death))

    def __init__(self, config: str = ""):
        """ Create the discrete tron environment.

        Parameters
        ----------
        config : str
            Serialized config string for specifying options for the environment.
            Use TronGridEnvironment.create for a more programming friendly way of initializing
            the environment.

        See Also
        --------
        colosseumrl.envs.tron.TronGridEnvironment.create
            A better constructor for the tron environment.
        """
        super().__init__(config)
        board_size, num_players, observation_window, remove_on_death = parse_tron_config(config)

        self.N = board_size
        self.num_players = num_players
        self.observation_window = observation_window
        self.fully_observable = observation_window < 0
        self.remove_on_death = remove_on_death

        self.player_array = np.arange(num_players)
        self.move_array = ['forward', 'right', 'left']
        self._moves = np.zeros(num_players, dtype=np.int64)

    def __repr__(self):
        output = ""
        output += "Tron Finite Grid Environment\n"
        output += "="*50 + "\n"
        output += "\tSize: {}x{}\n".format(self.N, self.N)
        output += "\tNumber of players: {}\n".format(self.num_players)
        output += "\tFully Observable: {}\n".format("Yes" if self.fully_observable else "No")
        output += "\tRemove old players: {}\n".format("Yes" if self.remove_on_death else "No")
        output += "-"*50 + "\n"
        return output

    def __str__(self):
        return self.__repr__()

    @property
    def min_players(self) -> int:
        """ Property holding the number of players present required to play game.

        Returns
        -------
        int
            The specified number of players in this game.
        """
        return self.num_players

    @property
    def max_players(self) -> int:
        """ Property holding the number of players present required to play game.

        Returns
        -------
        int
            The specified number of players in this game.
        """
        return self.num_players

    @staticmethod
    def observation_names() -> List[str]:
        """ Static method for returning the names of the observation objects.

        Returns
        -------
        List[str]
            The keys of the observation dictionary.
        """
        return ["board", "heads", "directions", "deaths"]

    @property
    def observation_shape(self) -> Dict[str, tuple]:
        """ Describe the fixed numpy shapes of each observation.

        Returns
        -------
        Dict[str, Tuple[int]]
            The shape, as a tuple, of each numpy array by their name.
        """
        return {
            "board": (self.N, self.N),
            "heads": (self.num_players, ),
            "directions": (self.num_players, ),
            "deaths": (self.num_players, )
        }

    def generate_start_positions(self, ring_offset: int = 1, spawn_offset: int = 0):
        N = self.N

        # Central ring parameters
        size = N // 2
        offset = N % 2
        center = -0.5 * (offset - 1)
        r1, r2 = size - ring_offset - 1, size - ring_offset
        side_length = 2 * (r1 + 1)

        # Create the outer ring
        y, x = np.ogrid[-size + center:size + offset + center, -size + center:size + offset + center]
        mask1 = (np.abs(x) <= r1) & (np.abs(y) <= r1)
        mask2 = (np.abs(x) <= r2) & (np.abs(y) <= r2)
        mask = mask2 ^ mask1

        yy, xx = np.where(mask)
        indices = yy * N + xx

        # Get each section of the ring independently
        top = indices[:side_length]
        right = indices[side_length:3 * side_length:2]
        bottom = indices[3 * side_length:]
        left = indices[side_length + 1:3 * side_length + 1:2]

        # Assigned each section a direction facing away from the wall
        directions = np.arange(side_length * 4) // side_length
        directions = (directions + 2) % 4

        # Extract the start locations and directions from the ordered ring
        sections = np.array_split(np.concatenate([top, right, bottom[::-1], left[::-1]]), self.num_players)
        directions = np.array_split(directions, self.num_players)

        def get_centers(array_list, offsets):
            clamp = lambda val, lower, upper: max(min(val, upper), lower)
            indices = (clamp(arr.size // 2 + offset, 0, arr.size - 1) for arr, offset in zip(array_list, offsets))
            centers = starmap(lambda arr, index: arr[index], zip(array_list, indices))
            return np.fromiter(centers, np.int64, len(array_list))

        if isinstance(spawn_offset, int):
            spawn_offset = (spawn_offset, spawn_offset + 1)
        offsets = [np.random.randint(*spawn_offset) for _ in range(self.num_players)]

        return get_centers(sections, offsets), get_centers(directions, offsets)

    def new_state(self, num_players: int = None, ring_offset: int = 1, spawn_offset: int = 2) -> Tuple[object, List[int]]:
        """ Create an initial tron state.

        Parameters
        ----------
        num_players : int, optional.
            The number of players for the game.
            Note, this option gets ignored here in favor of the global player configuration when creating
            the environment.
        ring_offset: int
            How into the arena the spawn ring is created. For example, if ring offset is 1, then the players
            will spawn 1 black away from the wall.
        spawn_offset: int or tuple
            The offset given to all of the players spawn positions along their spawn region.
            If it is a tuple, randomly chosen offset from the range.

        Returns
        -------
        State : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The full state of the new tron environment.
        player_list : List[int]
            Which players are currently acting.
        """
        num_players = self.num_players if num_players is None else num_players
        assert num_players == self.num_players, "Do not change the number of players from the game configuration."

        # Generate the Starting configuration
        np.random.seed(int(time()))
        board = np.zeros((self.N, self.N), dtype=np.int64)
        heads, directions = self.generate_start_positions(ring_offset, spawn_offset)
        deaths = np.zeros(self.num_players, dtype=np.int64)

        # Set up the initial board
        board.ravel()[heads] = self.player_array + 1

        return (board, heads, directions, deaths), self.player_array

    def next_state(self, state: object, players: [int], actions: [str]):
        """ Compute a single step in the game.

        Notes
        -----
        Player numbers must be numbers in the set {0, 1, ..., n-1} for an n player game.

        Parameters
        ----------
        state : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The current state of the game.
        players: [int]
            The players which are taking the given actions.
        actions : [str]
            The actions of each player.

        Returns
        -------
        new_state : object
            The new state of the game.
        new_players: List[int]
            List of players who's turn it is in the new state now.
        rewards : List[float]
            The reward for each player that acted.
        terminal : bool
            Whether or not the game has ended.
        winners: List[int]
            If the game has ended, who are the winners.
        """
        board, heads, directions, deaths = state

        # Convert the move strings to move indices for c++
        for player, action in zip(players, actions):
            self._moves[player] = self.STRING_TO_ACTION[action]

        # Make a copy of the state since we operate in-place
        new_board = np.copy(board)
        new_heads = np.copy(heads)
        new_directions = np.copy(directions)
        new_deaths = np.copy(deaths)

        # Execute the move
        next_state_inplace(new_board, new_heads, new_directions, new_deaths, self._moves)

        # Reduce players to the ones still alive
        new_players = np.where(new_deaths == 0)[0]

        # Make rewards be whether or not you lived or died
        rewards = -2 * (new_deaths > 0) + 1

        # Terminal is if everyone or everyone except one has died
        terminal = new_players.size <= 1

        # Winner is the final player or nobody if tie
        winners = new_players if terminal else None
        if winners is not None and len(winners) > 0:
            rewards[winners] += 9

        return (new_board, new_heads, new_directions, new_deaths), new_players, rewards, terminal, winners

    def valid_actions(self, state: object, player: int) -> [str]:
        """ Valid actions for a specific state.

        Parameters
        ----------
        state : object
            The current state of the game.
        player : int
            The player who is executing this action.

        Returns
        -------
        List[str]
            All possible actions for the game.
            For tron, this will always be ['forward', 'left', 'right']
        """
        return self.move_array

    def is_valid_action(self, state: object, player: int, action: str) -> bool:
        """ Whether or not an action is valid for a specific state.

        Parameters
        ----------
        state : object
            The current state of the game.
        player : int
            The player who is executing this action.
        action : str
            The action the player is executing.

        Returns
        -------
        bool
            Whether or not this is a valid action in the current state.
            This is always true for tron as every action is valid.
        """
        return True

    def state_to_observation(self, state: object, player: int) -> Dict[str, np.ndarray]:
        """ Convert the raw game state to the observation for the agent. Maps each observation name into an observation.

        Parameters
        ----------
        state : object
            The full server state of the game.
        player : int
            Which player is getting the observation.

        Returns
        -------
        Dict[str, np.ndarray]
            The observation dictionary with keys equal to the observation_names above.

        See Also
        --------
        colosseumrl.envs.tron.TronGridEnvironment.observation_names
            The list of observation keys.
        colosseumrl.envs.tron.TronGridEnvironment.observation_shapes
            The sizes of each observation.
        """
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
        """ Whether or not this class supports serialization of the state.

        Returns
        -------
        bool
            False, Tron doesnt need to be serializable as the state is current fully observable.
        """
        return False

    @staticmethod
    def serialize_state(state: object) -> bytearray:
        """ Serialize a game state and convert it to a bytearray to be saved or sent over a network.

        Parameters
        ----------
        state : object
            The current game state.

        Returns
        -------
        bytearray
            Serialized byte-string for the state.
        """
        return dumps(state)

    @staticmethod
    def deserialize_state(serialized_state: bytearray) -> object:
        """ Convert a serialized bytearray back into a game state.

        Parameters
        ----------
        serialized_state : bytearray
            Serialized byte-string for the state.

        Returns
        -------
        object
            The current game state.
        """
        return loads(serialized_state)

    # Customer Helpers
    @staticmethod
    def next_cell(x, y, direction, board_size: int = None):
        if direction == 0:  # North
            y = y - 1
        elif direction == 1:  # East
            x = x + 1
        elif direction == 2:  # South
            y = y + 1
        elif direction == 3:  # West
            x = x - 1

        if board_size:
            x = max(min(x, board_size - 1), 0)
            y = max(min(y, board_size - 1), 0)

        return x, y

    def compute_ranking(self, state: object, players: [int], winners: [int]) -> Dict[int, int]:
        board, _, _, deaths = state
        num_players = deaths.shape[0]

        # Find scores for each player to be the length of their tail
        scores = Counter(state[0].ravel() - 1)
        del scores[-1]

        # Rebalance ties
        tie_locations = np.where(deaths[deaths - 1] - np.arange(num_players) - 1 == 0)[0]
        for tie in tie_locations:
            killer = deaths[tie] - 1
            scores[tie] = min(scores[tie], scores[killer])

        # Compute Rankings
        rankings = {}
        previous_score = np.inf

        for next_rank, (player, score) in enumerate(scores.most_common()):
            if score < previous_score:
                current_rank = next_rank

            rankings[player] = current_rank
            previous_score = score

        return rankings


