import logging


class CustomFormatter(logging.Formatter):
    DEBUG = '\033[37m'  # white
    INFO = '\033[36m'  # cyan
    WARNING = '\033[33m'  # yellow
    ERROR = '\033[31m'  # red
    CRITICAL = '\033[41m'  # white on red bg
    reset = "\x1b[0m"
    format = "%(asctime)s - [%(levelname)s] - %(message)s"

    FORMATS = {
        logging.DEBUG: DEBUG + format + reset,
        logging.INFO: INFO + format + reset,
        logging.WARNING: WARNING + format + reset,
        logging.ERROR: ERROR + format + reset,
        logging.CRITICAL: CRITICAL + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def load_logger(verbosity: int):
    try:
        log_level = {
            0: logging.ERROR,
            1: logging.WARN,
            2: logging.INFO}[verbosity]
    except KeyError:
        log_level = logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
