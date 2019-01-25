import argparse
import pickle

import spacetime
from spacetime import Application
from spacetimerl.data_model import Player, ServerState
from spacetimerl.rl_logging import init_logging
from spacetimerl.frame_rate_keeper import FrameRateKeeper
from spacetimerl.base_environment import BaseEnvironment

logger = init_logging(logfile=None, redirect_stdout=True, redirect_stderr=True)

DEFAULT_PARAMS = {
    "port": 7777,
    "lenient_mode": True,
    "tick_rate": 60,
    "env_class_name": "test_game.TestGame"
}


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


def server_app(dataframe, env_class, args):
    fr = FrameRateKeeper(max_frame_rate=args['tick_rate'])
    players = {}

    server_state = ServerState(env_class.__name__)
    dataframe.add_one(ServerState, server_state)
    dataframe.commit()

    env: BaseEnvironment = env_class()

    logger.info("Waiting for enough players to join ({} required)...".format(env.min_players))
    while len(players) < env.min_players:
        fr.tick()
        dataframe.sync()

        new_players = dict((p.pid, p) for p in dataframe.read_all(Player))

        for new_id in new_players.keys() - players.keys():
            logger.info("New player joined with name: {}".format(new_players[new_id].name))

        for remove_id in players.keys() - new_players.keys():
            logger.info("Player {} has left.".format(players[remove_id].name))

        players = new_players

    logger.info("Finalizing players and setting up new environment.")
    state, player_turns = env.new_state(num_players=len(players))
    for i, player in enumerate(players.values()):
        player.finalize_player(number=i, observation=pickle.dumps(env.state_to_observation(state=state, player=i)))
        if i in player_turns:
            player.turn = True

    players_by_number = dict((p.number, p) for p in players.values())
    dataframe.sync()

    terminal = False
    turn_count = 0
    while not terminal:
        # Wait for a frame to tick
        fr.tick()

        dataframe.checkout()

        # Get the player dataframes of the players whos turn it is right now
        current_players = [p for p in players.values() if p.number in player_turns]
        current_actions = []
        server_state = dataframe.read_one(ServerState, server_state.oid)

        if args['lenient_mode'] and not all(p.ready_for_action_to_be_taken for p in current_players):
            continue

        # Queue up each players action if it is legal
        # If the player failed to respond in time, we will simply execute the previous action
        # If it is invalid, we will pass in a blank string
        for player in current_players:
            if env.is_valid_action(state=state, player=player.number, action=player.action):
                current_actions.append(player.action)
            else:
                logger.info("Player #{}, {}'s, action of {} was invalid, passing empty string as action"
                            .format(player.number, player.name, player.action))
                current_actions.append('')

        # Execute the current move
        state, player_turns, rewards, terminal, winners = (
            env.next_state(state=state, players=player_turns, actions=current_actions)
        )

        # Update the player data from the previous move.
        for player, reward in zip(current_players, rewards):
            player.reward_from_last_turn = float(reward)
            player.ready_for_action_to_be_taken = False
            player.turn = False

        # Tell the new players that its their turn and provide observation
        for player_number in player_turns:
            player = players_by_number[player_number]
            player.observation = pickle.dumps(env.state_to_observation(state=state, player=player_number))
            player.turn = True

        if terminal:
            server_state.terminal = True
            server_state.winners = pickle.dumps(winners)
            for player_number in winners:
                players_by_number[player_number].winner = True
            logger.info("Player: {} won the game.".format(winners))

        turn_count += 1
        dataframe.commit()

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
    server_app.start(env_class, vars(args))
