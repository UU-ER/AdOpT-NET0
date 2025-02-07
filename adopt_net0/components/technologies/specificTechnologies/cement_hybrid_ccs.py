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

        self.component_options.emissions_based_on = "output"
        self.component_options.size_based_on = "output"
        self.component_options.prod_capacity_clinker = tec_data["prod_capacity_clinker"]
        self.component_options.main_output_carrier = tec_data["Performance"][
            "main_output_carrier"
        ]

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits the technology performance

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        super(CementHybridCCS, self).fit_technology_performance(climate_data, location)

        # fit coefficients
        self.processed_coeff.time_independent["fit"] = (
            self.fitting_class.fit_performance_function(
                self.input_parameters.performance_data["performance"]
            )
        )

        phi = {}
        for car in self.input_parameters.performance_data["input_ratios"]:
            phi[car] = self.input_parameters.performance_data["input_ratios"][car]
        self.processed_coeff.time_independent["phi"] = phi

        def _calculate_bounds(self):
            """
            Calculates the bounds of the variables used
            """
            super(CementHybridCCS, self)._calculate_bounds()

            time_steps = len(self.set_t_performance)

            self.bounds["input"] = self.fitting_class.calculate_input_bounds(
                self.component_options.size_based_on, time_steps
            )
            self.bounds["output"] = self.fitting_class.calculate_output_bounds(
                self.component_options.size_based_on, time_steps
            )

            # Input bounds recalculation
            for car in self.component_options.input_carrier:
                if not car == self.component_options.main_input_carrier:
                    self.bounds["input"][car] = (
                        self.bounds["input"][self.component_options.main_input_carrier]
                        * self.input_parameters.performance_data["input_ratios"][car]
                    )
