from __future__ import print_function
import logging

import grpc

from .grpc_gen.server_pb2 import QuickMatchRequest
from .grpc_gen.server_pb2_grpc import MatchmakerStub

import random

from ..envs.blokus.blokus_env import BlokusEnv
from ..envs.blokus.blokus_client_env import BlokusClientEnv
from rlcompetition.rl_logging import init_logging
from rlcompetition.client_environment import RLApp
import numpy as np3
from random import choice

from collections import namedtuple
from typing import NamedTuple


class GameResponse(NamedTuple):
    host: str
    port: int
    username: str
    token: str


def request_game(hostname: str, port: int, username: str, password: str = "") -> GameResponse:
    """ Contact a matchmaking server and ask for a new game.

    This function will block until enough players connect to create a server.

    Parameters
    ----------
    hostname: str
    port: int
        Hostname and port of the remote matchmaking server
    username: str
        Username that will identify you in the game.
    password: str
        NOT USED CURRENT: Password to confirm your username, this will be used for tournament and ranking.

    Returns
    -------
    GameResponse NamedTuple with the following fields
        host: str
        port: int
            Hostname and port of the game server that was created for you
        username: str
            Your username again to verify.
        token: str
            Authentication string you will need to provide to connect to the match server
    """
    with grpc.insecure_channel('{}:{}'.format(hostname, port)) as channel:
        response = MatchmakerStub(channel).GetMatch(QuickMatchRequest(username=username))

    host, port = response.server.split(":")
    return GameResponse(host, int(port), username, response.auth_key)

def client(ce: BlokusClientEnv):
    logger = init_logging()

    logger.debug("Connecting to server and waiting for game to start...")
    player_num = ce.connect("player_{}".format(np.random.randint(0, 1024)))
    logger.debug("First observation: {}".format(ce.wait_for_turn()))
    logger.info("Game started...")

    while True:
        # ce.render(ce.full_state, player_num, winners)

        valid_actions = ce.valid_actions()
        action = choice(valid_actions)

        new_obs, reward, terminal, winners = ce.step(str(action))

        logger.debug("Took step with action {}, got: {}".format(action, (new_obs, reward, terminal, winners)))
        if terminal:
            logger.info("Game is over. Players {} won".format(winners))
            break


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = MatchmakerStub(channel)
        username = 'noobmaster{}'.format(random.randint(1,100))
        print("Acting as {}".format(username))
        response = stub.GetMatch(QuickMatchRequest(username=username))
    print("Got Match: " + response.server + " " + response.auth_key + " " + response.username)
    host, port = response.server.split(":")
    app = RLApp(host, int(port), response.auth_key, BlokusClientEnv, BlokusEnv)(client)
    app()

if __name__ == '__main__':
    logging.basicConfig()
    run()