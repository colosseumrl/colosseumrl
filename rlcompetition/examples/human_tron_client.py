from rlcompetition.matchmaking import request_game, GameResponse
from rlcompetition.RLApp import create_rl_agent
from rlcompetition.envs.tron import TronGridClientEnvironment
from rlcompetition.envs.tron import TronGridEnvironment

from rlcompetition.rl_logging import init_logging

from random import choice, randint
from matplotlib import pyplot as plt
import argparse
from time import time
from threading import Thread, Lock
from time import sleep

from pynput.keyboard import Key, Listener


logger = init_logging()


class Action:
    def __init__(self):
        self._action = "forward"

    def action(self):
        return self._action

    def __call__(self):
        return self._action

    def turn_right(self):
        self._action = "right"

    def turn_left(self):
        self._action = "left"

    def reset(self):
        self._action = "forward"


class ControlThread(Thread):
    def __init__(self, action: Action):
        super().__init__()
        self.action: Action = action

    def run(self) -> None:
        def on_press(key):
            if key == Key.left:
                self.action.turn_left()
                print("LEFT")
            elif key == Key.right:
                self.action.turn_right()
                print("RIGHT")
            elif key == Key.up:
                self.action.reset()
                print("UP")

        with Listener(on_press=on_press) as listener:
            listener.join()

def tron_client(env: TronGridClientEnvironment, username: str):
    logger.debug("Connecting to game server and waiting for game to start")
    player_num = env.connect(username)
    logger.debug("Player number: {}".format(player_num))
    logger.debug("First observation: {}".format(env.wait_for_turn()))
    logger.info("Game started...")

    current_action = Action()
    control_thread = ControlThread(current_action)
    control_thread.start()

    plt.figure(figsize=(8, 8))
    plt.ion()

    while True:
        sleep(0.1)
        plt.imshow(env.board())
        plt.draw()
        plt.pause(0.01)
        sleep(0.4)

        new_obs, reward, terminal, winners = env.step(current_action())
        current_action.reset()

        logger.debug("Got: {}".format((new_obs, reward, terminal, winners)))
        if terminal:
            logger.info("Game is over. Players {} won".format(winners))
            logger.info("Final observation: {}".format(new_obs))
            break


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
        username = "SwagMaster_{}".format(randint(0, 1024))
    else:
        username = args.username

    game: GameResponse = request_game(args.host, args.port, username)
    logger.debug("Game has been created. Playing as {}".format(username))
    logger.debug("Current Ranking: {}".format(game.ranking))

    agent = create_rl_agent(tron_client, game.host, game.port, game.token, TronGridClientEnvironment, TronGridEnvironment)
    agent(username)
