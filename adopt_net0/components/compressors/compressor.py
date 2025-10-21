from ..component import ModelComponent
from ..utilities import (
    annualize,
    set_discount_rate,
    perform_disjunct_relaxation,
    determine_variable_scaling,
    determine_constraint_scaling,
    get_attribute_from_dict,
)

import pandas as pd
import math
import copy
import pyomo.environ as pyo
import pyomo.gdp as gdp

import logging

log = logging.getLogger(__name__)


class Compressor(ModelComponent):
    """
    Class to read and manage data for compressors.

    Important
    ---------
    - The component that has the gas as **output** is defined as ``output_component``
      (with respective output pressure, type, existing).
    - The component that has the gas as **input** is defined as ``input_component``
      (with respective input pressure, type, existing).

    **Parameter declarations:**

    The following is a list of declared pyomo parameters.

    - ``para_name``: name of the compressor, formatted as type_component1_component2_(existing)
    - ``para_size_min``: minimal possible size
    - ``para_size_max``: maximal possible size
    - ``para_unit_capex``: investment costs per unit
    - ``para_unit_capex_annual``: Unit CAPEX annualized (annualized from given data on
      up-front CAPEX, lifetime and discount rate)
    - ``para_fix_capex``: fixed costs independent of size
    - ``para_fix_capex_annual``: Fixed CAPEX annualized (annualized from given data on
      up-front CAPEX, lifetime and discount rate)
    - ``para_opex_variable``: operational cost EUR/output or input
    - ``para_opex_fixed``: fixed opex as fraction of up-front capex

    **Variable declarations:**

    - ``var_flow``: flow that is processed by the compressor

    Only for active compressors:

    - ``var_consumption_energy``: energy consumption required by the compressor to raise the pressure (only if active)
    - ``var_size``: size of the compressor (only if active)
    - ``var_capex``: annualized investment of the compressor (only if active)
    - ``var_opex_variable``: variable operation costs
    - ``var_opex_fixed``: fixed operational costs as fraction of up-front CAPEX
    - ``var_capex_aux``: auxiliary variable to calculate the fixed opex of existing compressor

    **Constraint declarations**

    - For new compressors capex are defined linearly:

        .. math::
            capex_{aux} = size * capex_{unit_annual}

    - Variable OPEX: variable opex is defined in terms of the flow:

        .. math::
            opexvar_{t} = Flow_{t} * opex_{var}

    - Energy consumption calculation based on pressure ratio, type of compressor, flow:

        .. math::
             energyconsumption_{t} = Flow_{t} * energyconsumption(MW_{el}/MW_{H2})

    """

    def __init__(self, compr_data: dict):
        """
        Initializes compressor class from compressor data

        :param dict compr_data: compressor data
        """
        super().__init__(compr_data)

        # Modelling attributes
        self.input = None
        self.output = None
        self.set_t_full = None
        self.set_t_performance = None
        self.set_t_global = None
        self.sequence = None
        self.compression_active = None

        self.output_component = compr_data["connection_info"]["components"][0]
        self.input_component = compr_data["connection_info"]["components"][1]
        self.output_pressure = compr_data["connection_info"]["pressure"][0]
        self.input_pressure = compr_data["connection_info"]["pressure"][1]

        self.carrier = compr_data["carrier"]
        self.output_type = compr_data["connection_info"]["type"][0]
        self.input_type = compr_data["connection_info"]["type"][1]
        self.output_existing = compr_data["connection_info"]["existing"][0]
        self.input_existing = compr_data["connection_info"]["existing"][1]
        self.name_compressor = (
            f"{self.carrier}_Compressor_{self.output_component}_{self.input_component}"
        )

        if (self.output_existing == 1) and (self.input_existing == 1):
            self.name_compressor = self.name_compressor + "_existing"

    def fit_compressor_performance(self):
        """
        Fits compressor performance (bounds and coefficients).
        """

        time_independent = {}

        # Size
        time_independent["size_min"] = self.size_min
        time_independent["size_max"] = self.size_max

        if self.output_pressure >= self.input_pressure:
            self.compression_active = 0

        else:
            self.compression_active = 1

            # Energy consumption parameters

            time_independent["consumption"] = self.performance_data["energyconsumption"]
            time_independent["pressure_per_stage"] = self.performance_data[
                "max_pressure_per_stage"
            ]
            time_independent["isentropic_efficiency"] = self.performance_data[
                "isentropic_efficiency"
            ]
            time_independent["heat_coefficient"] = self.performance_data[
                "heat_coefficient"
            ]
            time_independent["mean_compressibility_factor"] = self.performance_data[
                "mean_compressibility_factor"
            ]

            # Energy
            time_independent["n_stages"] = math.ceil(
                math.log(self.input_pressure / self.output_pressure)
                / math.log(time_independent["pressure_per_stage"])
            )

            R = 8.314  # kJ/kmol/K
            T_in = 298.15  # K

            time_independent["energy_consumption"] = (
                time_independent["mean_compressibility_factor"]
                / 120
                / 2
                * T_in
                * (R / 1000)
                * time_independent["n_stages"]
                * (
                    time_independent["heat_coefficient"]
                    / (time_independent["heat_coefficient"] - 1)
                )
                * (1 / time_independent["isentropic_efficiency"])
                * (
                    (self.input_pressure / self.output_pressure)
                    ** (
                        (time_independent["heat_coefficient"] - 1)
                        / (
                            time_independent["n_stages"]
                            * time_independent["heat_coefficient"]
                        )
                    )
                    - 1
                )
            )  # MW_el/MW_H2

        if (self.output_existing == 1) and (self.input_existing == 1):
            self.existing = 1
        else:
            self.existing = 0

        self.processed_coeff.time_independent = time_independent

    def construct_compressor_model(
        self, b_compr, data: dict, set_t_full, set_t_clustered
    ):
        """
        Construct the compressor model with all required parameters, variable, sets,...

        :param b_compr: pyomo block with compressor model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with compressor model
        """

        # LOG
        log_msg = f"\t - Adding Compressor {self.name}"
        print(log_msg)
        log.info(log_msg)

        # compressor data
        config = data["config"]

        # SET T
        self.set_t_full = set_t_full

        if config["optimization"]["typicaldays"]["N"]["value"] == 0:
            # everything with full resolution
            self.modelled_with_full_res = True
            self.set_t_performance = set_t_full
            self.set_t_global = set_t_full
            self.sequence = list(self.set_t_performance)

        elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
            # everything with reduced resolution
            self.modelled_with_full_res = False
            self.set_t_performance = set_t_clustered
            self.set_t_global = set_t_clustered
            self.sequence = list(self.set_t_performance)

        elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
            # resolution of balances is full, so interactions with them also need to
            # be full resolution
            self.set_t_global = set_t_full

        # GENERAL TECHNOLOGY CONSTRAINTS
        b_compr = self._define_flow(b_compr)
        if self.compression_active == 1:
            b_compr = self._define_energyconsumption_parameters(b_compr)
            b_compr = self._define_energy_consumption(b_compr, data)
            b_compr = self._define_opex_var(b_compr, data)
            b_compr = self._define_size(b_compr)
            b_compr = self._define_capex_parameters(b_compr, data)
            b_compr = self._define_capex_variables(b_compr, data)
            b_compr = self._define_capex_constraints(b_compr, data)
            b_compr = self._define_opex_fixed(b_compr, data)

        return b_compr

    def _define_flow(self, b_compr):
        """
        Defines variable for compressor flow.

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """

        b_compr.var_flow = pyo.Var(
            self.set_t_global,
            within=pyo.NonNegativeReals,
        )

        return b_compr

    def _define_capex_parameters(self, b_compr, data):
        """
        Defines the capex parameters

        :param b_compr: pyomo block with compressor model
        :param dict data: dict containing model information
        :return:
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        b_compr.para_unit_capex = pyo.Param(
            domain=pyo.Reals,
            initialize=economics["unit_capex"],
            mutable=True,
        )
        b_compr.para_fix_capex = pyo.Param(
            domain=pyo.Reals,
            initialize=economics["fix_capex"],
            mutable=True,
        )
        b_compr.para_unit_capex_annual = pyo.Param(
            domain=pyo.Reals,
            initialize=annualization_factor * economics["unit_capex"],
            mutable=True,
        )
        b_compr.para_fix_capex_annual = pyo.Param(
            domain=pyo.Reals,
            initialize=annualization_factor * economics["fix_capex"],
            mutable=True,
        )

        return b_compr

    def _define_capex_variables(self, b_compr, data: dict):
        """
        Defines variables related to compressor capex.

        :param b_compr: pyomo block with compressor model
        :param dict data: dict containing model information
        :return: pyomo block with compressor model
        """
        config = data["config"]

        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        # CAPEX auxilliary (used to calculate theoretical CAPEX)
        # For new compressor, this is equal to actual CAPEX
        # For existing compressor it is used to calculate fixed OPEX
        b_compr.var_capex_aux = pyo.Var()

        b_compr.var_capex = pyo.Var()

        return b_compr

    def _define_capex_constraints(self, b_compr, data: dict):
        """
        Defines constraints related to compressor capex.

        :param b_compr: pyomo block with compressor model
        :param dict data: dict containing model information
        :return: pyomo block with compressor model
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        b_compr.const_capex_aux = pyo.Constraint(
            expr=b_compr.var_size * b_compr.para_unit_capex_annual
            == b_compr.var_capex_aux
        )

        if self.existing == 0:
            b_compr.const_capex = pyo.Constraint(
                expr=b_compr.var_capex == b_compr.var_capex_aux
            )
        elif self.existing == 1:
            b_compr.const_capex = pyo.Constraint(expr=b_compr.var_capex == 0)

        return b_compr

    def _define_energyconsumption_parameters(self, b_compr):
        """
        Constructs constraints for compressor energy consumption

        :param b_compr: pyomo compressor block
        :return: pyomo compressor block
        """
        # Set of consumed carriers
        b_compr.set_consumed_carriers = pyo.Set(
            initialize=list(self.processed_coeff.time_independent["consumption"].keys())
        )

        # Consumption from compressor
        b_compr.var_consumption_energy = pyo.Var(
            self.set_t_global,
            b_compr.set_consumed_carriers,
            domain=pyo.NonNegativeReals,
        )

        return b_compr

    def _define_energy_consumption(self, b_compr, data):
        """
        Defines compressor energy consumption

        :param b_compr: pyomo compressor block
        :return: pyomo compressor block
        """

        def init_compr_energy(b, t, car):
            """
            Define energy for compression in MW
            """
            return (
                b_compr.var_consumption_energy[t, car]
                == b_compr.var_flow[t]
                * self.processed_coeff.time_independent["energy_consumption"]
            )  # MW_el

        b_compr.const_compress_energy = pyo.Constraint(
            self.set_t_global, b_compr.set_consumed_carriers, rule=init_compr_energy
        )

        return b_compr

    def _define_size(self, b_compr):
        """
        Defines variables and parameters related to compressor size.

        :param b_compr: pyomo block with compressor model
        :return: pyomo block with compressor model
        """
        coeff_ti = self.processed_coeff.time_independent

        b_compr.para_size_min = pyo.Param(
            domain=pyo.NonNegativeReals, initialize=coeff_ti["size_min"], mutable=True
        )
        b_compr.para_size_max = pyo.Param(
            domain=pyo.NonNegativeReals, initialize=coeff_ti["size_max"], mutable=True
        )

        b_compr.var_size = pyo.Var(
            within=pyo.NonNegativeReals,
            bounds=(b_compr.para_size_min, b_compr.para_size_max),
        )

        def sizing_rule(b, t, car):
            return b_compr.var_size >= b_compr.var_consumption_energy[t, car]

        if self.existing == 0:
            b_compr.const_size = pyo.Constraint(
                self.set_t_global, b_compr.set_consumed_carriers, rule=sizing_rule
            )

        return b_compr

    def fix_size(self, b_compr, size):
        """
        Fixes the size of existing compressor by constrain using component capacity.

        :param b_compr: pyomo block with compressor model
        :param size: minimum energy capacity siz between input and output component
        :return: pyomo block with compressor model
        """

        if self.existing == 1:

            def sizing_existing_compressor(b):
                return (
                    b_compr.var_size
                    == size
                    * self.processed_coeff.time_independent["energy_consumption"]
                )

            b_compr.const_size_existing = pyo.Constraint(
                rule=sizing_existing_compressor
            )

        return b_compr

    def _define_opex_var(self, b_compr, data: dict):
        """
        Defines variable and fixed OPEX

        :param b_compr: pyomo block with compressor model
        :param dict data: dict containing model information
        :return: pyomo block with compressor model
        """
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        # VARIABLE OPEX
        b_compr.para_opex_variable = pyo.Param(
            domain=pyo.Reals, initialize=economics["opex_variable"], mutable=True
        )
        b_compr.var_opex_variable = pyo.Var()

        hour_factors = data["hour_factors"]
        nr_timesteps_averaged = data["nr_timesteps_averaged"]

        def init_opex_variable(const):
            """opexvar = sum(Input_{t, maincarrier}) * opex_{var}"""

            return (
                sum(
                    (b_compr.var_flow[t] * nr_timesteps_averaged * hour_factors[t - 1])
                    * b_compr.para_opex_variable
                    for t in self.set_t_global
                )
                == b_compr.var_opex_variable
            )

        b_compr.const_opex_variable = pyo.Constraint(rule=init_opex_variable)

        return b_compr

    def _define_opex_fixed(self, b_compr, data: dict):

        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        # FIXED OPEX
        b_compr.para_opex_fixed = pyo.Param(
            domain=pyo.Reals, initialize=economics["opex_fixed"], mutable=True
        )
        b_compr.var_opex_fixed = pyo.Var()

        b_compr.const_opex_fixed = pyo.Constraint(
            expr=(b_compr.var_capex_aux / annualization_factor)
            * b_compr.para_opex_fixed
            == b_compr.var_opex_fixed
        )

        return b_compr

    # def _define_decommissioning_at_once_constraints(self, b_compr):
    #     """
    #     Defines constraints to ensure that a technology can only be decommissioned as a whole.
    #
    #     This function creates a disjunction formulation that enforces
    #     full-plant decommissioning decisions, meaning that either the technology is fully installed
    #     or fully decommissioned, with no partial decommissioning allowed.
    #
    #     :param b_tec: The block representing the technology.
    #
    #     :return: The modified technology block with added decommissioning constraints.
    #     """
    #
    #     # Full plant decommissioned only
    #     self.big_m_transformation_required = 1
    #     s_indicators = range(0, 2)
    #
    #     def init_decommission_full(dis, ind):
    #         if ind == 0:  # compressor not installed
    #             dis.const_decommissioned = pyo.Constraint(expr=b_compr.var_size == 0)
    #         else:  # tech installed
    #             dis.const_installed = pyo.Constraint(
    #                 expr=b_compr.var_size == b_compr.para_size_initial
    #             )
    #
    #     b_compr.dis_decommission_full = gdp.Disjunct(
    #         s_indicators, rule=init_decommission_full
    #     )
    #
    #     def bind_disjunctions(dis):
    #         return [b_compr.dis_decommission_full[i] for i in s_indicators]
    #
    #     b_compr.disjunction_decommission_full = gdp.Disjunction(rule=bind_disjunctions)
    #
    #     return b_compr

    def write_results_compressor_design(self, h5_group, model_block):
        """
        Function to report compressor design

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        if self.compression_active == 1:
            h5_group.create_dataset("size", data=[model_block.var_size.value])
            if self.existing == 0:
                h5_group.create_dataset("capex_tot", data=[model_block.var_capex.value])
            else:
                return
        else:
            return

    def write_results_compressor_operation(self, h5_group, model_block):
        """
        Function to report compressor operation

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """

        h5_group.create_dataset(
            "flow", data=[model_block.var_flow[t].value for t in self.set_t_global]
        )

        for car in model_block.set_consumed_carriers:
            h5_group.create_dataset(
                "energy consumption",
                data=[
                    model_block.var_consumption_energy[t, car].value
                    for t in self.set_t_global
                ],
            )
