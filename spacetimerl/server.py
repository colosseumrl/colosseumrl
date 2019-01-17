import argparse
import sys
import time
import datetime

import spacetime
from spacetime import Application
from spacetimerl.Datamodel import Player
from spacetimerl.rl_logging import init_logging

logger = init_logging(logfile=None, redirect_stdout=True, redirect_stderr=True)

DEFAULT_PARAMS = {
    "port": 7777,
}


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def server_app(dataframe):
    while True:
        time.sleep(5)
        print("Players: {}".format(dataframe.read_all(Player)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for key, value in DEFAULT_PARAMS.items():
        key = '--' + key.replace('_', '-')
        parser.add_argument(key, type=type(value), default=value)

    args = parser.parse_args()
    dict_args = vars(args)
    log_params(dict_args)

    server_app = Application(server_app, server_port=dict_args['port'], Types=[Player],
                             version_by=spacetime.utils.enums.VersionBy.FULLSTATE)
    server_app.start()
