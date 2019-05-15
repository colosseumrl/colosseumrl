from ..envs.blokus.blokus_env import BlokusEnv
from ..envs.blokus.blokus_client_env import BlokusClientEnv
from rlcompetition.rl_logging import init_logging
from rlcompetition.client_environment import RLApp
import numpy as np
from random import choice


@RLApp("localhost", 7777, client_environment=BlokusClientEnv, server_environment=BlokusEnv, time_out=5)
def main(ce: BlokusClientEnv):
    logger = init_logging()

    logger.debug("Connecting to server and waiting for game to start...")
    player_num = ce.connect("player_{}".format(np.random.randint(0, 1024)))
    logger.debug("First observation: {}".format(ce.wait_for_turn()))
    winners = None

    while True:
        # ce.render(ce.full_state, player_num, winners)

        valid_actions = ce.valid_actions()
        action = choice(valid_actions)

        new_obs, reward, terminal, winners = ce.step(str(action))

        logger.debug("Took step with action {}, got: {}".format(action, (new_obs, reward, terminal, winners)))
        if terminal:
            logger.info("Game is over. Players {} won".format(winners))
            break


if __name__ == '__main__':
    main()
