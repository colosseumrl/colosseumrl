""" This script defines an example automated tron client that will avoid walls if it's about to crash into one.

This is meant to be an example of how to implement a basic matchmaking agent.

"""

import argparse

from random import choice, randint

from colosseumrl.matchmaking import request_game, GameResponse
from colosseumrl.RLApp import create_rl_agent
from colosseumrl.envs.tron import TronGridClientEnvironment
from colosseumrl.envs.tron import TronGridEnvironment
from colosseumrl.rl_logging import init_logging

logger = init_logging()


def tron_client(env: TronGridClientEnvironment, username: str):
    """ Our client function for the random tron client.

    Parameters
    ----------
    env : TronGridClientEnvironment
        The client environment that we will interact with for this agent.
    username : str
        Our desired username.
    """

    # Connect to the game server and wait for the game to begin.
    # We run env.connect once we have initialized ourselves and we are ready to join the game.
    player_num = env.connect(username)
    logger.debug("Player number: {}".format(player_num))

    # Next we run env.wait_for_turn() to wait for our first real observation
    env.wait_for_turn()
    logger.info("Game started...")

    # Keep executing moves until the game is over
    terminal = False
    while not terminal:
        # See if there is a wall in front of us, if there is, then we will turn in a random direction.
        n_loc, n_obj = env.next_location()
        if n_obj == 0:
            action = 'forward'
        else:
            action = choice(['left', 'right'])

        # We use env.step in order to execute an action and wait until it is our turn again.
        # This function will block while the action is executed and will return the next observation that belongs to us
        new_obs, reward, terminal, winners = env.step(action)
        logger.debug("Took step with action {}, got: {}".format(action, (new_obs, reward, terminal, winners)))

    # Once the game is over, we print out the results and close the agent.
    logger.info("Game is over. Players {} won".format(winners))
    logger.info("Final observation: {}".format(new_obs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", "-s", type=str, default="localhost",
                        help="Hostname of the matchmaking server.")
    parser.add_argument("--port", "-p", type=int, default=50051,
                        help="Port the matchmaking server is running on.")
    parser.add_argument("--username", "-u", type=str, default="",
                        help="Desired username to use for your connection. By default it will generate a random one.")

    logger.debug("Connecting to matchmaking server. Waiting for a game to be created.")

    args = parser.parse_args()

    if args.username == "":
        username = "Tester_{}".format(randint(0, 1000))
    else:
        username = args.username

    # We use request game to connect to the matchmaking server and await a game assigment.
    game: GameResponse = request_game(args.host, args.port, username)
    logger.debug("Game has been created. Playing as {}".format(username))
    logger.debug("Current Ranking: {}".format(game.ranking))

    # Once we have been assigned a game server, we launch an RLApp agent and begin our computation
    agent = create_rl_agent(agent_fn=tron_client,
                            host=game.host,
                            port=game.port,
                            auth_key=game.token,
                            client_environment=TronGridClientEnvironment,
                            server_environment=TronGridEnvironment)
    agent(username)
