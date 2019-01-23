from multiprocessing import Pipe
import numpy as np
from typing import Tuple
import pickle
import logging

import spacetime
from spacetime import Application
from spacetimerl.data_model import Player, ServerState
from spacetimerl.frame_rate_keeper import FrameRateKeeper

CLIENT_TICK_RATE = 60

logger = logging.getLogger(__name__)


def game_is_terminal(dataframe):
    return dataframe.read_all(ServerState)[0].terminal


def client_app(dataframe, remote, parent_remote, player_name):
    # parent_remote.close() # we are in a separate thread, not process

    dataframe.pull()
    dataframe.checkout()
    player = Player(name=player_name)
    dataframe.add_one(Player, player)
    dataframe.commit()
    dataframe.push()

    dataframe.pull()
    dataframe.checkout()
    if dataframe.read_one(Player, player.pid) is not None:
        logger.info("Connected to server, waiting for game to start...")
    else:
        logger.info("Server rejected adding your player, perhaps the max player limit has been reached.")
        exit()

    fr = FrameRateKeeper(CLIENT_TICK_RATE)

    while player.number == -1:
        fr.tick()
        dataframe.pull()
        dataframe.checkout()
        player = dataframe.read_one(Player, player.pid)

    logger.info("Game has started, acting as player number {}".format(player.number))

    while not player.turn:
        fr.tick()
        dataframe.pull()
        dataframe.checkout()
        player = dataframe.read_one(Player, player.pid)
    remote.send(pickle.loads(player.observation))
    logger.debug("First turn for player {} started".format(player.number))

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == 'step':

                if not game_is_terminal(dataframe):
                    action = data
                    player.action = action
                    player.ready_for_action_to_be_taken = True
                    dataframe.commit()
                    dataframe.push()

                    print("sent action")

                    while not player.turn or player.ready_for_action_to_be_taken:
                        fr.tick()
                        dataframe.pull()
                        dataframe.checkout()
                        player = dataframe.read_one(Player, player.pid)

                new_observation = pickle.loads(player.observation)
                reward = player.reward_from_last_turn
                terminal = game_is_terminal(dataframe)

                if terminal:
                    winners = pickle.loads(dataframe.read_all(ServerState)[0].winners)
                    player.acknowledges_game_over = True
                    dataframe.commit()
                    dataframe.push()
                else:
                    winners = None

                remote.send((new_observation, reward, terminal, winners))

            elif cmd == 'close':
                dataframe.delete_one(Player, player)
                remote.close()
                break

            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('client_app: got KeyboardInterrupt')


class ClientNetworkEnv:

    def __init__(self, server_hostname, port, player_name):
        self.remote, app_remote = Pipe()
        self.player_client = Application(client_app, dataframe=(server_hostname, port), Types=[Player, ServerState],
                                    version_by=spacetime.utils.enums.VersionBy.FULLSTATE)
        self.player_client.start_async(remote=app_remote, parent_remote=self.remote, player_name=player_name)
        self.first_observation = self.remote.recv()

    def close(self):
        self.remote.send(("close", None))
        self.player_client.join()

    def get_first_observation(self):
        return self.first_observation

    def step(self, action: str) -> Tuple[np.ndarray, float, bool, int]:
        """
        Compute a single step in the game.

        Parameters
        ----------
        action : int

        Returns
        -------
        new_observation : np.ndarray
        reward : float
        terminal : bool
        winners: list - Only matters if terminal = True
        """
        self.remote.send(('step', action))
        return self.remote.recv()

