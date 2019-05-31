""" This script defines an example human operated tron client.

This is meant to be a fun example and a test of the networking and matchmaking system.

NOTE
----------------------------------------------------------------------------------------------------------
This script has EXTRA DEPENDENCIES. In order to run this script you need to install opencv and pynput.
These are left out of the global dependency list in order to simplify installation.
"""

from colosseumrl.matchmaking import request_game, GameResponse
from colosseumrl.RLApp import create_rl_agent
from colosseumrl.envs.tron import TronGridClientEnvironment
from colosseumrl.envs.tron import TronGridEnvironment

from colosseumrl.rl_logging import init_logging

from random import randint
import cv2
import argparse
from threading import Thread
from time import sleep, time
import numpy as np
from pynput.keyboard import Key, Listener


logger = init_logging()

FRAME_MILLISECONDS = 200


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

    frame_start_time = time()
    while True:

        board = env.board()
        im = cv2.applyColorMap(np.uint8(board * 255 // np.max(board)), cv2.COLORMAP_JET)
        im = cv2.resize(im, (420, 420), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Tron", im)
        cv2.waitKey(1)
        frame_delta = time() - frame_start_time
        sleep((FRAME_MILLISECONDS / 1000) - frame_delta)

        new_obs, reward, terminal, winners = env.step(current_action())
        frame_start_time = time()

        current_action.reset()

        if terminal:
            logger.info("Game is over. Players {} won".format(winners))
            logger.info("Final observation: {}".format(new_obs))
            cv2.destroyAllWindows()

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
