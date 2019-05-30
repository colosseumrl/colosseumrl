from ..envs.blokus.BlokusEnvironment import BlokusEnvironment
from ..envs.blokus.BlokusClientEnvironment import BlokusClientEnvironment
from colosseumrl.RLApp import RLApp
from colosseumrl.rl_logging import init_logging
import logging
import numpy as np
from random import choice
from itertools import repeat
import multiprocessing as mp
import time
import os


def start_multiple_clients(server_hostname, server_port):
    ctx = mp.get_context('fork')
    remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(4)])
    ps = [ctx.Process(target=client_subproc_worker, args=(work_remote, remote, hostname, port))
          for (work_remote, remote, hostname, port) in zip(work_remotes, remotes, repeat(server_hostname), repeat(server_port))]
    for p in ps:
        # p.daemon = True  # if the main process crashes, we should not cause things to hang
        p.start()
    for remote in work_remotes:
        remote.close()

    return remotes


def client_subproc_worker(remote, parent_remote, server_hostname, server_port):
    parent_remote.close()

    @RLApp(server_hostname, server_port, client_environment=BlokusClientEnvironment, server_environment=BlokusEnvironment, time_out=10)
    def run_client(ce: BlokusClientEnvironment):

        # Ensure every process has a different seed
        np.random.seed(os.getpid())

        player_num = ce.connect("player_{}".format(np.random.randint(0, 1024)))
        logger.debug("client player number {} connected".format(player_num))
        remote.send(player_num)

        winners = None
        first_observation = ce.wait_for_turn()

        try:
            while True:
                cmd, data = remote.recv()
                if cmd == 'step':
                    action = str(data)
                    new_obs, reward, terminal, winners = ce.step(action)
                    remote.send((new_obs, reward, terminal, winners))
                    if terminal:
                        break
                elif cmd == 'render':
                    ce.render(ce.full_state, player_num, winners)
                elif cmd == 'quit':
                    break
                elif cmd == 'valid_actions_list':
                    remote.send(ce.valid_actions())
                elif cmd == 'valid_actions_dict':
                    remote.send(ce.valid_actions_dict())
                elif cmd == 'player_num':
                    remote.send(player_num)
                elif cmd == 'first_observation':
                    remote.send(first_observation)
                else:
                    raise NotImplementedError
        except KeyboardInterrupt:
            print('client subproc worker: got KeyboardInterrupt')

        remote.close()

    run_client()


def main():

    logger.info("Starting multiple clients...")
    client_remotes = start_multiple_clients(server_hostname="localhost", server_port=7777)

    logger.info("Waiting for game to start...")

    # Get player turn numbers for each client remote
    client_player_nums = []
    for client_remote in client_remotes:
        player_number = client_remote.recv()
        client_player_nums.append(player_number)
    logger.debug("Client player nums: {}".format(client_player_nums))

    # Sort client remotes by player turn number
    client_remotes = [remote for _, remote in sorted(zip(client_player_nums, client_remotes), key=lambda pair: pair[0])]

    # Main game loop
    first_turn = True
    player_actions = [None, None, None, None]
    while True:
        terminal = None
        winners = None

        for i, client_remote in enumerate(client_remotes):
            if first_turn:
                client_remote.send(("first_observation", None))
                logger.debug("Player {} first observation: {}".format(i, client_remote.recv()))
                logger.info("Game started...")
            else:
                new_obs, reward, terminal, winners = client_remote.recv()
                logger.debug("Player i took step with action {}, got: {}".format(i, player_actions[i],
                                                                                 (new_obs, reward, terminal, winners)))
                if terminal:
                    continue

            # Render player's view of game
            # client_remote.send(('render', None))

            # Get available actions for this player
            client_remote.send(("valid_actions_list", None))
            valid_actions_list = client_remote.recv()
            player_actions[i] = choice(valid_actions_list)

            # Perform action
            client_remote.send(("step", player_actions[i]))
        if first_turn:
            first_turn = False

        if terminal:
            logger.info("Game is over. Players {} won".format(winners))
            break

    for client_remote in client_remotes:
        client_remote.close()


if __name__ == '__main__':
    logger = init_logging()
    logger.setLevel(logging.INFO)

    while True:
        main()
        time.sleep(0.1)
