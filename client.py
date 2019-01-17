import argparse
import logging

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for key, value in DEFAULT_PARAMS.items():
        key = '--' + key.replace('_', '-')
        parser.add_argument(key, type=type(value), default=value)

    args = parser.parse_args()
    dict_args = vars(args)
    log_params(dict_args)

