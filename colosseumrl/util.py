from .rl_logging import get_logger
import socket

logger = get_logger()


def log_params(params):
    """ Print the current parameters to the log file. """
    params = vars(params)
    for k in sorted(params.keys()):
        logger.info('{}: {}'.format(k, params[k]))


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
