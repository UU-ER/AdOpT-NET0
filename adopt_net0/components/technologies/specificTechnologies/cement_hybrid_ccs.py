import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

from ..utilities import fit_piecewise_function
from ..technology import Technology

import logging

log = logging.getLogger(__name__)


class CementHybridCCS(Technology):
    """
    Cement plant with hybrid CCS

    The plant had an oxyfuel combustion in the calciner and post-combustion capture with MEA afterward. The size
    of the oxyfuel is fixed, while the size and capture rate of the MEA are variables of the optimization
    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)
