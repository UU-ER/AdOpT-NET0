import logging
from .core import ModelHub
from .plugins import PluginManager

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# Stream Handler to control console output
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)
