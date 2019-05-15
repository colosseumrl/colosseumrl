import argparse
import pickle

import spacetime
from spacetime import Node, Dataframe

from .data_model import ServerState, Player, _Observation, Observation
from .rl_logging import init_logging, get_logger
from .frame_rate_keeper import FrameRateKeeper
from .base_environment import BaseEnvironment
from .config import ENVIRONMENT_CLASSES
from .util import log_params

from importlib import import_module
from typing import Type, Dict, List

logger = get_logger()

def get_class(kls: str):
    """ Dynamically import a python class from a module listing. """
    module, name = kls.rsplit('.', 1)
    module = import_module(module)
    return getattr(module, name)


def server_app(dataframe: spacetime.Dataframe,
               env_class: Type[BaseEnvironment],
               observation_type: Type,
               args: dict,
               whitelist: list = []):
    fr: FrameRateKeeper = FrameRateKeeper(max_frame_rate=args['tick_rate'])

    # Keep track of each player and their associated observations
    observation_dataframes: Dict[int, Dataframe] = {}
    observations: Dict[int, _Observation] = {}
    players: Dict[int, Player] = {}

    # Function to help push all observations
    def push_observations():
        for df in observation_dataframes.values():
            df.commit()

    # Add the server state to the master dataframe
    server_state = ServerState(env_class.__name__, args["config"], env_class.observation_names())
    dataframe.add_one(ServerState, server_state)
    dataframe.commit()

    # Create the environment and start the server
    env: BaseEnvironment = env_class(args["config"])

    logger.info("Waiting for enough players to join ({} required)...".format(env.min_players))

    # Add whitelist support, players will be rejected if their key does not match the expected keys
    whitelist_used = len(whitelist) > 0
    whitelist_connected = {key: False for key in whitelist}
    while len(players) < env.min_players:
        fr.tick()
        dataframe.sync()

        new_players: Dict[int, Player] = dict((p.pid, p) for p in dataframe.read_all(Player))

        for new_id in new_players.keys() - players.keys():
            name = new_players[new_id].name
            key = new_players[new_id].authentication_key

            if whitelist_used and key not in whitelist_connected:
                logger.info("Player tried to join with invalid authentication_key: {}".format(name))
                del new_players[new_id]
                continue

            if whitelist_used and whitelist_connected[key]:
                logger.info("Player tried to join twice with the same authentication_key: {}".format(name))
                del new_players[new_id]
                continue

            logger.info("New player joined with name: {}".format(name))

            # Create new observation dataframe for the new player
            obs_df = Dataframe("{}_observation".format(name), [observation_type])
            obs = observation_type(new_id)
            obs_df.add_one(observation_type, obs)

            # Add the dataframes to the database
            observation_dataframes[new_id] = obs_df
            observations[new_id] = obs

        for remove_id in players.keys() - new_players.keys():
            logger.info("Player {} has left.".format(players[remove_id].name))
            del observations[remove_id]
            del observation_dataframes[remove_id]

        players = new_players

    logger.info("Finalizing players and setting up new environment.")

    # Create the inital state for the environment and push it if enabled
    state, player_turns = env.new_state(num_players=len(players))
    if not args["observations_only"] and env.serializable():
        server_state.serialized_state = env.serialize_state(state)

    # Set up each player
    for i, (pid, player) in enumerate(players.items()):
        # Add the initial observation to each player
        observations[pid].set_observation(env.state_to_observation(state=state, player=i))

        # Finalize each player by giving it a player number and a port for the dataframe
        player.finalize_player(number=i, observation_port=observation_dataframes[pid].details[1])
        if i in player_turns:
            player.turn = True

    # Push all of the results to the player
    players_by_number: Dict[int, Player] = dict((p.number, p) for p in players.values())
    push_observations()
    dataframe.sync()

    # Wait for all players to be ready
    # TODO: Add timeout to this
    while not all(player.ready_for_start for player in players.values()):
        dataframe.checkout()
        fr.tick()

    # Start the game
    terminal = False

    server_state.server_no_longer_joinable = True
    dataframe.commit()

    turn_count = 0
    while not terminal:
        # Wait for a frame to tick
        fr.tick()

        # Get new data
        dataframe.checkout()

        # Get the player dataframes of the players whos turn it is right now
        current_players: List[Player] = [p for p in players.values() if p.number in player_turns]
        current_actions: List[str] = []

        if not args['realtime'] and not all(p.ready_for_action_to_be_taken for p in current_players):
            continue

        # Queue up each players action if it is legal
        # If the player failed to respond in time, we will simply execute the previous action
        # If it is invalid, we will pass in a blank string
        for player in current_players:
            if player.action == '' or env.is_valid_action(state=state, player=player.number, action=player.action):
                current_actions.append(player.action)
            else:
                logger.info("Player #{}, {}'s, action of {} was invalid, passing empty string as action"
                            .format(player.number, player.name, player.action))
                current_actions.append('')

        # Execute the current move
        state, player_turns, rewards, terminal, winners = (
            env.next_state(state=state, players=player_turns, actions=current_actions)
        )

        # Update true state if enabled
        if not args["observations_only"] and env.serializable():
            server_state.serialized_state = env.serialize_state(state)

        # Update the player data from the previous move.
        for player, reward in zip(current_players, rewards):
            player.reward_from_last_turn = float(reward)
            player.ready_for_action_to_be_taken = False
            player.turn = False

        # Tell the new players that its their turn and provide observation
        for player_number in player_turns:
            player = players_by_number[player_number]
            observations[player.pid].set_observation(env.state_to_observation(state=state, player=player_number))
            player.turn = True

        if terminal:
            server_state.terminal = True
            server_state.winners = pickle.dumps(winners)
            for player_number in winners:
                players_by_number[player_number].winner = True
            logger.info("Player: {} won the game.".format(winners))

        turn_count += 1
        push_observations()
        dataframe.commit()

    for player in players:
        player.turn = True

    dataframe.commit()
    dataframe.push()

    # TODO| The code below attempts to ensure that the players have the final state of the game before the server quits.
    # TODO| However, an error is thrown when players disconnect during the checkout. If this snippet was removed,
    # TODO| players would have a similar error when the server would quit while they are pulling.
    # TODO| May need to talk to Rohan about cleanly exiting this kind of situation.
    # TODO| It would also be great if we could instead properly confirm that recipients got a message.
    for player in players:
        while not player.acknowledges_game_over:
            fr.tick()
            dataframe.checkout()

    logger.info("Game has ended. Player {} is the winner.".format(dataframe.read_one(ServerState, server_state.oid)))


if __name__ == '__main__':

    logger = init_logging(logfile=None, redirect_stdout=True, redirect_stderr=True)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("environment", type=str,
                        help="The name of the environment. Choices are: ['blokus']")
    parser.add_argument("--config", '-c', type=str, default="",
                        help="Config string that will be passed into the environment constructor.")
    parser.add_argument("--port", "-p", type=int, default=7777,
                        help="Server Port.")
    parser.add_argument("--tick-rate", "-t", type=int, default=60,
                        help="The max tick rate that the server will run on.")
    parser.add_argument("--realtime", "-r", action="store_true",
                        help="With this flag on, the server will not wait for all of the clients to respond.")
    parser.add_argument("--observations-only", '-f', action='store_true',
                        help="With this flag on, the server will not push the true state of the game to the clients "
                             "along with observations")

    args = parser.parse_args()
    log_params(args)

    # env_class: Type[BaseEnvironment] = get_class(args.environment_class)
    try:
        env_class: Type[BaseEnvironment] = ENVIRONMENT_CLASSES[args.environment]
    except KeyError:
        raise ValueError("The \'environment\' argument must must be chosen from the following list: {}".format(
            ENVIRONMENT_CLASSES.keys()
        ))

    observation_type: Type[_Observation] = Observation(env_class.observation_names())

    while True:
        app = Node(server_app,
                                 server_port=args.port,
                                 Types=[Player, ServerState])
        app.start(env_class, observation_type, vars(args))
        del app

