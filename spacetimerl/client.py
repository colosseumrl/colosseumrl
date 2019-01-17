import argparse
import logging

import spacetime

LOG_FORMAT = "%(asctime)s %(levelname)-6s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__package__)
formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

DEFAULT_PARAMS = {
    "server_hostname": 'localhost',
    "port": 7777,
    "player_name": 'no name'
}


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def visualize(dataframe):
    dataframe.pull()
    world = World()
    for a in dataframe.read_all(Asteroid):
        world.asteroids[a.oid] = a

    vis = Visualizer(world)
    threading.Thread(target=sync, args=[dataframe, world]).start()
    # Run pygame on the main thread or else
    vis.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for key, value in DEFAULT_PARAMS.items():
        key = '--' + key.replace('_', '-')
        parser.add_argument(key, type=type(value), default=value)

    args = parser.parse_args()
    dict_args = vars(args)
    log_params(dict_args)

    player_client = Application(visualize, dataframe=(args.host, args.port), Types=[Asteroid, Ship],
                                version_by=spacetime.utils.enums.VersionBy.FULLSTATE)
    player_client.start()