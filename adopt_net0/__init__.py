import logging
from .core import (
    ModelHub,
    create_optimization_templates,
    create_input_data_folder_template,
    copy_technology_data,
    copy_network_data,
    fill_carrier_data
)
from .plugins import PluginManager

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# Stream Handler to control console output
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)
