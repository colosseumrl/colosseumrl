from rlcompetition.matchmaking import request_game
from rlcompetition.client_environment import create_rl_agent
from rlcompetition.envs.tron import TronGridClientEnvironment
from rlcompetition.envs.tron import TronGridEnvironment

from rlcompetition.rl_logging import init_logging

from random import choice, randint


logger = init_logging()


def tron_client(env: TronGridClientEnvironment, username: str):
    logger.debug("Connecting to game server and waiting for game to start")
    player_num = env.connect(username)
    logger.debug("Player number: {}".format(player_num))
    logger.debug("First observation: {}".format(env.wait_for_turn()))
    logger.info("Game started...")

    while True:
        # ce.render(ce.full_state, player_num, winners)

        n_loc, n_obj = env.next_location()
        if n_obj == 0:
            action = 'forward'
        else:
            action = choice(['left', 'right'])

        new_obs, reward, terminal, winners = env.step(action)

        logger.debug("Took step with action {}, got: {}".format(action, (new_obs, reward, terminal, winners)))
        if terminal:
            logger.info("Game is over. Players {} won".format(winners))
            logger.info("Final observation: {}".format(new_obs))
            break


if __name__ == '__main__':
    logger.debug("Connecting to matchmaking server. Waiting for a game to be created.")
    username = "SwagMaster_{}".format(randint(0, 1024))
    game = request_game("localhost", 50051, username)
    logger.debug("Game has been created. Playing as {}".format(username))

    agent = create_rl_agent(tron_client, game.host, game.port, game.token, TronGridClientEnvironment, TronGridEnvironment)
    agent(username)