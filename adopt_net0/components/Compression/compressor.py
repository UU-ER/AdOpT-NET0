from ..component import ModelComponent
from ..utilities import (
    annualize,
    set_discount_rate,
    perform_disjunct_relaxation,
    determine_variable_scaling,
    determine_constraint_scaling,
)

import pandas as pd
import copy
import pyomo.environ as pyo
import pyomo.gdp as gdp

import logging

log = logging.getLogger(__name__)


class Compressor(ModelComponent):
    """
    Class to read and manage compression features

    """

    def __init__(self, comp_data: dict):
        """
        Initializes compression class from compressor data

        :param dict comp_data: compressor data
        """
        super().__init__(comp_data)
