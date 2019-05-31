from colosseumrl.ClientEnvironment import ClientEnvironment

from .TronGridEnvironment import parse_tron_config, TronGridEnvironment

from typing import Tuple


# Stub for later
class TronGridClientEnvironment(ClientEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parse out config for easy access to game parameters
        config = parse_tron_config(self._server_state.env_config)
        self.board_size = config[0]
        self.num_players = config[1]
        self.observation_window = config[2]
        self.remove_on_death = config[3]

        # Force the server environment to be the tron game
        if self._server_environment is None:
            self._server_environment = TronGridEnvironment(config)

    @staticmethod
    def direction_to_delta(direction: int) -> Tuple[int, int]:
        """ Convert an integer direction into an x, y delta

        Parameters
        ----------
        direction : int
            The base direction provided by the directions dictionary.
        Returns
        -------
        Tuple[int, int]
            The offset required in the x and y direction.
        """
        if direction == 0:
            return 0, -1
        elif direction == 1:
            return 1, 0
        elif direction == 2:
            return 0, 1
        elif direction == 3:
            return -1, 0
        else:
            return 0, 0

    def board(self, x: int = None, y: int = None):
        """ Helper method to get a location on the board

        Parameters
        ----------
        x : int
            The x coordinate of the board; or, if y is none, the absolute location
        y : int
            The y coordinate of the board

        Returns
        -------
        If both parameters are None, returns the whole board
        If x is given and y is not, then it gets an absolute location at the board
        If both x and y are given, then it gets the location (x, y) on the board
        """
        if x is None:
            return self.observation['board']

        if y is None:
            x = x % self.board_size
            y = x // self.board_size

        if (y < 0) or (y >= self.board_size) or (x < 0) or (x >= self.board_size):
            return -1

        return self.observation['board'][y, x]

    def next_location(self):
        """ Get the location and object in front of your current head

        Returns
        -------
        location: int
            Location that you will be in the next move if you move forward
        object: int
            Object located at that location
        """
        observation = self.observation

        head = observation['heads'][0]
        direction = observation['directions'][0]

        headx, heady = head % self.board_size, head // self.board_size
        delta = self.direction_to_delta(direction)
        headx += delta[0]
        heady += delta[1]

        new_head = heady * self.board_size + headx
        return new_head, self.board(headx, heady)
