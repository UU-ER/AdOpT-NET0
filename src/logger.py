import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    filename="full_log.log",
    encoding="utf-8",
    format="%(levelname)s:%(message)s",
)

# Create logger
logger = logging.getLogger(__name__)


def log_event(message: str, print_it: bool = True, level: str = "info"):
    """
    Logs and prints a message
    :param str message: message to log
    :param int print_it: [0,1] if message should also be printed
    :param str level: ['info', 'warning'] which level to log
    """
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning((message))
    if print_it:
        print(message)
