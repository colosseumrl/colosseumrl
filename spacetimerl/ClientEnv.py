from multiprocessing import Pipe
import numpy as np
from typing import Tuple
import sys

import spacetime
from spacetime import Application
from spacetimerl.Environment import BaseEnvironment
from spacetimerl.Datamodel import Player


def client_app(dataframe, remote, parent_remote, player_name):
    # parent_remote.close() # we are in a separate thread, not process

    dataframe.pull()
    dataframe.checkout()

    player = Player(name=player_name)

    dataframe.add_one(Player, player)
    dataframe.commit()
    dataframe.push()

    print("remote: {} parent_remote: {}, player_name: {}".format(remote, parent_remote, player_name))

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'state_shape':
                raise NotImplementedError
                state_shape = None
                remote.send(state_shape)
            elif cmd == 'observation_shape':
                raise NotImplementedError
                observation_shape = None
                remote.send(observation_shape)
            elif cmd == 'next_state':
                state, player, actions = data
                raise NotImplementedError
                new_state, reward, terminal, winner = None
                remote.send((new_state, reward, terminal, winner))
            elif cmd == 'valid_actions':
                state = data
                raise NotImplementedError
                valid_actioms = None
                remote.send(valid_actions)
            elif cmd == 'print':
                print(data)
                remote.send("printed")
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('client_app: got KeyboardInterrupt')
    finally:
        "close connection to server?"


class ClientEnv(BaseEnvironment):

    def __init__(self, server_hostname, port, player_name):

        self.remote, app_remote = Pipe()

        self.player_client = Application(client_app, dataframe=(server_hostname, port), Types=[Player],
                                    version_by=spacetime.utils.enums.VersionBy.FULLSTATE)
        self.player_client.start_async(remote=app_remote, parent_remote=self.remote, player_name=player_name)
        # app_remote.close() # we are in a separate thread, not process

        self.remote.send(("print", "hey"))
        print(self.remote.recv())

    def close(self):
        self.remote.send(("close", None))
        self.player_client.join()

    @property
    def state_shape(self) -> tuple:
        """ Property holding the numpy shape of a single state. """
        raise NotImplementedError

    @property
    def observation_shape(self) -> tuple:
        """ Property holding the numpy shape of a transformed observation state. """
        raise NotImplementedError

    def new_state(self, num_players: int = 1) -> np.ndarray:
        """ Create a fresh state. This could return a fixed object or randomly initialized on, depending on the game. """
        raise NotImplementedError

    def add_player(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def next_state(self, state: np.ndarray, player: int, action: int) -> Tuple[np.ndarray, float, bool, int]:
        """
        Compute a single step in the game.

        Parameters
        ----------
        state : np.ndarray
        player: int
        action : int

        Returns
        -------
        new_state : np.ndarray
        reward : float
        terminal : bool
        winner: int - Only matters if terminal = True
        """
        raise NotImplementedError

    def valid_actions(self, state: np.ndarray) -> [int]:
        """ Valid actions for a specific state. """
        raise NotImplementedError

    def state_to_observation(self, state: np.ndarray, player: int) -> np.ndarray:
        """ Convert the raw game state to the observation for the agent.

        This can return different values for the different players. Default implementation is just the identity."""
        return state


if __name__ == '__main__':
    ce = ClientEnv(server_hostname="localhost", port=7777, player_name="this_is_a_player_name")
    ce.close()
