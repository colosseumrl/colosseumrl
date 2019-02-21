from spacetimerl.rl_logging import init_logging
from spacetimerl.client_environment import RLApp, ClientEnv

import numpy as np

@RLApp("localhost", 7777)
def main(ce: ClientEnv):
    logger = init_logging()
    ce.connect("player_{}".format(np.random.randint(0, 1024)))
    logger.debug("First observation: {}".format(ce.wait_for_turn()))

    last_reward = -100
    action_delta = -1
    action = np.random.randint(0, 100)
    while True:
        new_obs, reward, terminal, winner = ce.step(str(action))

        logger.debug("Took step with action {}, got: {}".format(action, (new_obs, reward, terminal, winner)))
        if terminal:
            logger.info("Game is over. Players {} won".format(winner))
            break

        # Simple action policy for test game
        if reward <= last_reward:
            action_delta = action_delta * -1
        action += action_delta
        last_reward = reward

if __name__ == '__main__':
    main()
