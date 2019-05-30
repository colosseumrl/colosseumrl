import logging as _logging
import sys
import os

use_colorlog = True
try:
    from colorlog import ColoredFormatter
except ImportError:
    use_colorlog = False

COLORED_LOG_FORMAT = "%(asctime)s %(log_color)s%(levelname)-6s%(reset)s | %(log_color)s%(message)s%(reset)s"
LOG_FORMAT = "%(asctime)s %(levelname)-6s | %(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class LogWriter(object):

    def __init__(self, logger, log_level=_logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def read(self):
        pass

    def flush(self):
        pass


def get_logger():
    return _logging.getLogger(__package__)


def handle_exception(exc_type, exc_value, exc_traceback):
    # if issubclass(exc_type, KeyboardInterrupt):
    #     get_logger().warning("Keyboard Interrupt")
    #     sys.__excepthook__(exc_type, exc_value, exc_traceback)
    #     return

    get_logger().error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def init_logging(logfile=None, redirect_stdout=True, redirect_stderr=True, handle_tensorflow=True):
    logger = get_logger()
    logger.setLevel(_logging.DEBUG)

    add_handlers_to_logger(logger, logfile)

    sys.excepthook = handle_exception

    if redirect_stdout:
        sys.stdout = LogWriter(logger, log_level=_logging.DEBUG)
    if redirect_stderr:
        sys.stderr = LogWriter(logger, log_level=_logging.ERROR)
    if handle_tensorflow:
        add_handlers_to_logger(_logging.getLogger('tensorflow'), logfile)

    return logger


def add_handlers_to_logger(logger, logfile=None):
    if use_colorlog:
        formatter = ColoredFormatter(COLORED_LOG_FORMAT, DATE_FORMAT)
    else:
        formatter = _logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    stream_handler = _logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if logfile:
        log_to_file(logger, logfile)


def log_to_file(logger, filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    file_handler = _logging.FileHandler(filename)
    formatter = _logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

