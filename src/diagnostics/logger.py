import logging

def configure_logging(save_path):
    logging.basicConfig(filename=save_path, level=logging.INFO)
