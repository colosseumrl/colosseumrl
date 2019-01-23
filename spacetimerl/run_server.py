import argparse
import sys
import time
import datetime
import pickle

import spacetime
from spacetime import Application
from spacetimerl.data_model import Player, ServerState
from spacetimerl.rl_logging import init_logging
from spacetimerl.frame_rate_keeper import FrameRateKeeper

logger = init_logging(logfile=None, redirect_stdout=True, redirect_stderr=True)

DEFAULT_PARAMS = {
    "port": 7777,
    "env_class_name": "test_game.TestGame"
}

SERVER_TICK_RATE = 60


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def log_params(params):
    params = vars(params)
    for k in sorted(params.keys()):
        logger.info('{}: {}'.format(k, params[k]))


def server_app(dataframe, env_class):
    fr = FrameRateKeeper(max_frame_rate=SERVER_TICK_RATE)

    dataframe.checkout()
    server_state = ServerState(env_class.__name__)
    dataframe.add_one(ServerState, server_state)
    dataframe.commit()
    dataframe.push()

    env = env_class()

    logger.info("Waiting for enough players to join ({} required)...".format(env.min_players))
    last_player_count = 0
    last_joined_player_pids_and_names = {}
    while last_player_count < env.min_players:
        fr.tick()

        dataframe.sync()
        players = dataframe.read_all(Player)
        new_player_count = len(players)

        if new_player_count > last_player_count:
            for player in players:
                if player.pid not in last_joined_player_pids_and_names:
                    last_joined_player_pids_and_names[player.pid] = player.name
                    logger.info("New player joined with name: {}".format(player.name))

        if new_player_count < last_player_count:
            for pid in last_joined_player_pids_and_names.keys():
                if pid not in [player.pid for player in players]:
                    logger.info("Player {} has left.".format(last_joined_player_pids_and_names[pid]))
                    del last_joined_player_pids_and_names[pid]

        last_player_count = new_player_count

    logger.info("Finalizing players and setting up new environment.")
    current_env_state = env.new_state(num_players=last_player_count)
    player_pids_by_turn = {}
    for i, player in enumerate(dataframe.read_all(Player)):
        player.finalize_player(number=i, observation=pickle.dumps(env.state_to_observation(state=current_env_state, player=i)))
        player_pids_by_turn[i] = player.pid
    dataframe.sync()

    env_done = False
    current_player_turn = 0
    turn_count = 0
    while not env_done:
        fr.tick()
        dataframe.pull()
        dataframe.checkout()
        current_player = dataframe.read_one(Player, player_pids_by_turn[current_player_turn])

        server_state = dataframe.read_one(ServerState, server_state.oid)

        if current_player.ready_for_action_to_be_taken:
            if env.is_valid_action(state=current_env_state, action=current_player.action):
                action_to_execute = current_player.action
            else:
                logger.info("Player #{}, {}'s, action of {} was invalid, passing empty string as action"
                            .format(current_player_turn, current_player.name, current_player.action))
                action_to_execute = ""

            next_state, reward, env_done, winner = env.next_state(state=current_env_state, player=current_player_turn, action=action_to_execute)

            current_env_state = next_state
            current_player.reward_from_last_turn = float(reward)

            if env_done:
                server_state.winner = winner
                logger.info("Player {} won the game.".format(winner))

            current_player.ready_for_action_to_be_taken = False
            current_player.turn = False

            turn_count += 1
            current_player_turn = (current_player_turn + 1) % last_player_count
            dataframe.commit()
            dataframe.push()

        else:
            if current_player.turn is False:
                current_player.turn = True
                dataframe.commit()
                dataframe.push()
                logger.debug("Player #{}'s turn...".format(current_player_turn))

    for player in dataframe.read_all(Player):
        player.turn = True
    dataframe.commit()
    dataframe.push()

    # TODO| The code below attempts to ensure that the players have the final state of the game before the server quits.
    # TODO| However, an error is thrown when players disconnect during the checkout. If this snippet was removed,
    # TODO| players would have a similar error when the server would quit while they are pulling.
    # TODO| May need to talk to Rohan about cleanly exiting this kind of situation.
    # TODO| It would also be great if we could instead properly confirm that recipients got a message.
    for player in dataframe.read_all(Player):
        while not player.acknowledges_game_over:
            fr.tick()
            dataframe.checkout()

    logger.info("Game has ended. Player {} is the winner.".format(dataframe.read_one(ServerState, server_state.oid)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for key, value in DEFAULT_PARAMS.items():
        key = '--' + key.replace('_', '-')
        parser.add_argument(key, type=type(value), default=value)

    args = parser.parse_args()
    log_params(args)

    env_class = get_class(args.env_class_name)

    server_app = Application(server_app, server_port=args.port, Types=[Player, ServerState],
                             version_by=spacetime.utils.enums.VersionBy.FULLSTATE)
    server_app.start(env_class)
