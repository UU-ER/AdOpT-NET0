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

    The plant has an oxyfuel combustion in the calciner and post-combustion capture with MEA afterward. The size
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

        performance_data_path = Path(__file__).parent.parent.parent.parent
        performance_data_path = (
            performance_data_path
            / "data/technology_data/Industrial/CementHybridCCS_data/performance_cost_cementHybridCCS.xlsx"
        )

        performance_data = pd.read_excel(
            performance_data_path, sheet_name="performance", index_col=0
        )
        capex_data_oxy = pd.read_excel(
            performance_data_path, sheet_name="cost_oxy", index_col=0
        )
        capex_data_mea = pd.read_excel(
            performance_data_path, sheet_name="cost_mea", index_col=0
        )
        # TODO: make a function that cleans data (cement output either 0 or at full capacity), converts CO2 to clinker and daily to hourly

        prod_capacity_clinker = self.input_parameters.performance_data[
            "prod_capacity_clinker"
        ]

        if prod_capacity_clinker < 100:
            plant_size_type = "small"
        else:
            plant_size_type = "large"

        self.processed_coeff.time_independent["plant_size_type"] = plant_size_type
        self.processed_coeff.time_independent["alpha_oxy"] = performance_data.loc[
            "alpha_oxy", "value"
        ]
        self.processed_coeff.time_independent["alpha_mea"] = performance_data.loc[
            "alpha_mea", "value"
        ]
        self.processed_coeff.time_independent["beta_oxy"] = performance_data.loc[
            "beta_oxy", "value"
        ]
        self.processed_coeff.time_independent["capex_data_oxy"] = capex_data_oxy
        self.processed_coeff.time_independent["capex_data_mea"] = capex_data_mea

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(CementHybridCCS, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)
        prod_capacity_clinker = self.input_parameters.performance_data[
            "prod_capacity_clinker"
        ]
        emissions_clinker = self.input_parameters.performance_data["performance"][
            "tCO2_tclinker"
        ]
        CCR_oxy = self.input_parameters.performance_data["performance"]["CCR_oxy"]
        CCR_mea = self.input_parameters.performance_data["performance"]["CCR_mea"]

        # Output Bounds
        self.bounds["output"]["CO2captured"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * prod_capacity_clinker
                * emissions_clinker
                * (CCR_oxy + (1 - CCR_oxy) * max(CCR_mea)),
            )
        )

        self.bounds["output"]["clinker"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps) * prod_capacity_clinker,
            )
        )

        # Input Bounds
        self.bounds["input"]["electricity"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * (
                    prod_capacity_clinker
                    * emissions_clinker
                    * CCR_oxy
                    * self.processed_coeff.time_independent["alpha_oxy"]
                    + prod_capacity_clinker
                    * emissions_clinker
                    * (1 - CCR_oxy)
                    * max(CCR_mea)
                    * self.processed_coeff.time_independent["alpha_mea"]
                ),
            )
        )
        self.bounds["input"]["heat"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=time_steps)
                * (
                    prod_capacity_clinker
                    * emissions_clinker
                    * CCR_oxy
                    * self.processed_coeff.time_independent["beta_oxy"]
                ),
            )
        )

    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type CementHybridCCS

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(CementHybridCCS, self).construct_tech_model(
            b_tec, data, set_t_full, set_t_clustered
        )

        # Size constraint
        prod_capacity_clinker = self.input_parameters.performance_data[
            "prod_capacity_clinker"
        ]
        emissions_clinker = self.input_parameters.performance_data["performance"][
            "tCO2_tclinker"
        ]
        alpha_oxy = self.processed_coeff.time_independent["alpha_oxy"]
        beta_oxy = self.processed_coeff.time_independent["beta_oxy"]
        alpha_mea = self.processed_coeff.time_independent["alpha_mea"]
        emission_capacity_max = prod_capacity_clinker * emissions_clinker
        CCR_oxy = self.input_parameters.performance_data["performance"]["CCR_oxy"]
        CCR_mea = self.input_parameters.performance_data["performance"]["CCR_mea"]

        b_tec.var_co2_captured_mea = pyo.Var(
            self.set_t_performance,
            within=pyo.NonNegativeReals,
            bounds=[0, emission_capacity_max * (1 - CCR_oxy) * CCR_mea],
        )

        b_tec.var_size_mea = pyo.Var(
            within=pyo.NonNegativeReals,
            bounds=[0, emission_capacity_max * (1 - CCR_oxy)],
        )

        def init_size_constraint_mea(const, t):
            return b_tec.var_co2_captured_mea[t] <= b_tec.var_size_mea * CCR_mea

        b_tec.const_size = pyo.Constraint(
            self.set_t_performance, rule=init_size_constraint_mea
        )

        def init_mea_operation_constraint(const, t):
            return (
                b_tec.var_co2_captured_mea
                <= self.output["clinker"] * (1 - CCR_oxy) * CCR_mea
            )

        b_tec.const_mea_operation = pyo.Constraint(
            self.set_t_performance, rule=init_mea_operation_constraint
        )

        # Input bounds
        # TODO check if this function is actually called (and not the one in superclass technology)
        def init_input_bounds(bounds, t, car):
            return tuple(self.bounds["input"][car][self.sequence[t - 1] - 1, :])

        # output bounds
        # TODO check if this function is actually called (and not the one in superclass technology)
        def init_output_bounds(bounds, t, car):
            return tuple(self.bounds["output"][car][self.sequence[t - 1] - 1, :])

        # input-output correlations
        # TODO define var_co2_mea, to be inserted here as well
        def init_input_output(const, t, car_input):
            if car_input == "heat":
                return (
                    self.input[t, car_input]
                    == self.output[t, "clinker"]
                    * emissions_clinker
                    * CCR_oxy
                    * beta_oxy
                )
            elif car_input == "electricity":
                return self.input[t, car_input] == self.output[
                    t, "clinker"
                ] * emissions_clinker * (
                    CCR_oxy * alpha_oxy + (1 - CCR_oxy) * CCR_mea * alpha_mea
                )

        b_tec.const_input_output = pyo.Constraint(
            self.set_t_performance, b_tec.set_input_carriers, rule=init_input_output
        )

        def init_output_output(const, t):
            return (
                self.output[t, "CO2captured"]
                == self.output[t, "clinker"] * emissions_clinker * CCR_oxy
                + b_tec.var_co2_captured_mea
            )

        b_tec.const_output_output = pyo.Constraint(
            self.set_t_performance, rule=init_output_output
        )

        # define emissions

        #
