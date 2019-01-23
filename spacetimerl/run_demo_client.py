from spacetimerl.client_network_env import ClientNetworkEnv
from spacetimerl.rl_logging import init_logging

import numpy as np

if __name__ == '__main__':
    logger = init_logging()

    ce = ClientNetworkEnv(server_hostname="localhost", port=7777,
                          player_name="Xxsome_player_{}xX".format(np.random.randint(0, 500)))

    logger.debug("First observation: {}".format(ce.get_first_observation()))

    last_reward = -100
    action_delta = -1
    action = np.random.randint(0, 100)
    while True:
        new_obs, reward, terminal, winner = ce.step(str(action))

        logger.debug("Took step with action {}, got: {}".format(action, (new_obs, reward, terminal, winner)))
        if terminal:
            logger.info("Game is over. Player {} won".format(winner))
            break

        # Simple action policy for test game
        if reward <= last_reward:
            action_delta = action_delta * -1
        action += action_delta
        last_reward = reward

    ce.close()
