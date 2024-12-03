import logging
from logging.handlers import MemoryHandler
from .modelhub import ModelHub as ModelHub
from .result_management import (
    print_h5_tree,
    extract_dataset_from_h5,
    extract_datasets_from_h5group,
)
from .diagnostics import get_infeasible_constraints
from .data_preprocessing import *
from .case_studies import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Stream Handler to control console output
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)
