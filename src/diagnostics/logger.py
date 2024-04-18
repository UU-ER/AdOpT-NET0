import logging
from pathlib import Path

def configure_logging(save_path):
    """
    Defines a logger, that saves the complete log file to the specified path. This function needs to be
    called before deploying diagnostic tools

    :param str save_path: Path to save log file to
    """
    logging.basicConfig(filename=Path(save_path), level=logging.INFO)