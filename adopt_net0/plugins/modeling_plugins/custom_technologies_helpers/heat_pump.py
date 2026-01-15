"""
Resembles a heat pump.
Three different types of heat pumps are possible: air sourced (
'HeatPump_AirSourced'), ground sourced ('HeatPump_GroundSourced') and water
sourced ('HeatPump_WaterSourced'). Additionally, a heating curve is determined for
heating for buildings. Then, the application needs to be set to either
'floor_heating' or 'radiator_heating' in the data file. Otherwise, the output
temperature of the heat pump can also be set to a given temperature. The
coefficient of performance at full load is calculated in the respective fitting
function with the equations provided in Ruhnau, O., Hirth, L., & Praktiknjo,
A. (2019). Time series of heat demand and heat pump efficiency for energy system
modeling. Scientific Data, 6(1). https://doi.org/10.1038/s41597-019-0199-y

The part load behavior is modelled after equation (3) in Xu, Z., Li, H., Xu, W.,
Shao, S., Wang, Z., Gou, X., Zhao, M., & Li, J. (2022). Investigation on the
efficiency degradation characterization of low ambient temperature air source
heat pump under partial load operation. International Journal of Refrigeration,
133, 99–110. https://doi.org/10.1016/J.IJREFRIG.2021.10.002

Essentially, the equations for the heat pump model are the same as for generic
conversion technology type 1 (with time-dependent performance parameter).

"""

import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
import numpy as np
import statsmodels.api as sm

from adopt_net0.core.components.technologies.utilities import fit_piecewise_function, fit_linear_function
from adopt_net0.core.components.technology import Technology

import logging

log = logging.getLogger(__name__)


class HeatPump(Technology):
    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.emissions_based_on = "input"
        self.main_input_carrier = tec_data["Performance"]["main_input_carrier"]

    def fit_performance(self, modelhub: object, component_id: tuple, **kwargs):
        """
        Fits technology performance and writes it to self.

        :param modelhub: model hub
        :param tuple component_id: component id containing (period, node, component name)
        """
        super(HeatPump, self).fit_performance(modelhub, component_id)
        period = component_id[0]
        node = component_id[1]
        technology_time_series = modelhub.data["time_series_data"]["full_resolution"][(period, node, "TechnologyTimeSeries", self.name)]

        # Climate data & Number of timesteps
        time_steps = len(technology_data)

        # Ambient air temperature
        T = copy.deepcopy(technology_data["temp_air"])

        # Determine T_out
        if self.performance_data["application"] == "radiator_heating":
            t_out = 40 - T
        elif self.performance_data["application"] == "floor_heating":
            t_out = 30 - 0.5 * T
        else:
            t_out = self.performance_data["T_out"]

        # Determine delta T
        delta_T = t_out - T

        # Determine COP
        if "AirSourced" in self.name:
            cop = 6.08 - 0.09 * delta_T + 0.0005 * delta_T**2
        elif "GroundSourced" in self.name:
            cop = 10.29 - 0.21 * delta_T + 0.0012 * delta_T**2
        elif "WaterSourced" in self.name:
            cop = 9.97 - 0.20 * delta_T + 0.0012 * delta_T**2

        log.info("Deriving performance data for Heat Pump...")

        if (
            self.performance_function_type == 1 or self.performance_function_type == 2
        ):  # Linear performance function
            size_alpha = 1
        elif self.performance_function_type == 3:
            size_alpha = 2
        else:
            raise Exception(
                "performance_function_type must be an integer between 1 and 3"
            )

        fit = {}
        fit["out"] = {}
        alpha1 = np.empty(shape=(time_steps, size_alpha))
        alpha2 = np.empty(shape=(time_steps, size_alpha))
        bp_x = np.empty(shape=(time_steps, size_alpha + 1))

        for idx, cop_t in enumerate(cop):
            if idx % 100 == 1:
                print("\rComplete: ", round(idx / time_steps, 2) * 100, "%", end="")

            if self.performance_function_type == 1:
                x = np.linspace(self.performance_data["min_part_load"], 1, 9)
                y = (x / (1 - 0.9 * (1 - x))) * cop_t * x
                coeff = fit_linear_function(x, y)
                alpha1[idx, :] = coeff[0]

            elif self.performance_function_type == 2:
                x = np.linspace(self.performance_data["min_part_load"], 1, 9)
                y = (x / (1 - 0.9 * (1 - x))) * cop_t * x
                x = sm.add_constant(x)
                coeff = fit_linear_function(x, y)
                alpha1[idx, :] = coeff[1]
                alpha2[idx, :] = coeff[0]

            elif self.performance_function_type == 3:  # piecewise performance function
                y = {}
                x = np.linspace(self.performance_data["min_part_load"], 1, 9)
                y["out"] = (x / (1 - 0.9 * (1 - x))) * cop_t * x
                time_step_fit = fit_piecewise_function(x, y, 2)
                alpha1[idx, :] = time_step_fit["out"]["alpha1"]
                alpha2[idx, :] = time_step_fit["out"]["alpha2"]
                bp_x[idx, :] = time_step_fit["out"]["bp_x"]
        print("Complete: ", 100, "%")

        # Coefficients
        fit["coeff"] = {}
        if self.performance_function_type == 1:
            fit["coeff"]["alpha1"] = alpha1.round(5)

        elif self.performance_function_type == 2:  # Linear performance function
            fit["coeff"]["alpha1"] = alpha1.round(5)
            fit["coeff"]["alpha2"] = alpha2.round(5)

        elif self.performance_function_type == 3:  # Piecewise performance function
            fit["coeff"]["alpha1"] = alpha1.round(5)
            fit["coeff"]["alpha2"] = alpha2.round(5)
            fit["coeff"]["bp_x"] = bp_x.round(5)

        # Coefficients
        self.processed_coeff.time_dependent_full = fit["coeff"]

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(HeatPump, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        if self.performance_function_type == 1:
            for c in self.output_carrier:
                self.bounds["output"][c] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps))
                        * self.processed_coeff.time_dependent_used["alpha1"][:, 0],
                    )
                )

        elif self.performance_function_type == 2:  # Linear performance function
            for c in self.output_carrier:
                self.bounds["output"][c] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        self.processed_coeff.time_dependent_used["alpha1"][:, -1]
                        + self.processed_coeff.time_dependent_used["alpha2"][:, -1],
                    )
                )

        elif self.performance_function_type == 3:  # Piecewise performance function
            for c in self.output_carrier:
                self.bounds["output"][c] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        self.processed_coeff.time_dependent_used["alpha1"][:, -1]
                        + self.processed_coeff.time_dependent_used["alpha2"][:, -1],
                    )
                )

        # Input Bounds
        for car in self.input_carrier:
            self.bounds["input"][car] = np.column_stack(
                (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
            )

        time_steps = len(self.set_t_performance)

    def construct_model(self, b_tec, modelhub, set_t_full, set_t_clustered, **kwargs):
        """
        Adds constraints to technology blocks for tec_type HP (Heat Pump)

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(HeatPump, self).construct_model(
            b_tec, modelhub, set_t_full, set_t_clustered, **kwargs
        )

        # DATA OF TECHNOLOGY
        coeff_ti = self.processed_coeff.time_independent
        dynamics = self.processed_coeff.dynamics
        rated_capacity = coeff_ti["rated_capacity"]

        if self.performance_function_type == 1:
            self._performance_function_type_1(b_tec)
        elif self.performance_function_type == 2:
            self._performance_function_type_2(b_tec)
            self.big_m_transformation_required = 1
        elif self.performance_function_type == 3:
            self._performance_function_type_3(b_tec)
            self.big_m_transformation_required = 1

        # size constraint based on input
        def init_size_constraint(const, t):
            return self.input[t, "electricity"] <= b_tec.var_size * rated_capacity

        b_tec.const_size = pyo.Constraint(
            self.set_t_performance, rule=init_size_constraint
        )

        # RAMPING RATES
        if "ramping_time" in dynamics:
            if not dynamics["ramping_time"] == -1:
                self._define_ramping_rates(b_tec, data)



    def _performance_function_type_1(self, b_tec):
        """
        Linear, no minimal partload, through origin
        :param b_tec: technology block
        :return: technology block
        """
        # Get performance parameters
        coeff_td = self.processed_coeff.time_dependent_used
        alpha1 = coeff_td["alpha1"]

        def init_input_output(const, t):
            return (
                self.output[t, "heat"] == alpha1[t - 1] * self.input[t, "electricity"]
            )

        b_tec.const_input_output = pyo.Constraint(
            self.set_t_performance, rule=init_input_output
        )



    def _performance_function_type_2(self, b_tec):
        """
        Linear, minimal partload
        :param b_tec: technology block
        :return: technology block
        """
        # Get performance parameters
        coeff_td = self.processed_coeff.time_dependent_used
        coeff_ti = self.processed_coeff.time_independent
        alpha1 = coeff_td["alpha1"]
        alpha2 = coeff_td["alpha2"]
        min_part_load = coeff_ti["min_part_load"]
        rated_capacity = coeff_ti["rated_capacity"]

        # define disjuncts for on/off
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off

                def init_input_off(const):
                    return self.input[t, "electricity"] == 0

                dis.const_input = pyo.Constraint(rule=init_input_off)

                def init_output_off(const):
                    return self.output[t, "heat"] == 0

                dis.const_output_off = pyo.Constraint(rule=init_output_off)
            else:  # technology on
                # input-output relation
                def init_input_output_on(const):
                    return (
                        self.output[t, "heat"]
                        == alpha1[t - 1] * self.input[t, "electricity"]
                        + alpha2[t - 1] * b_tec.var_size * rated_capacity
                    )

                dis.const_input_output_on = pyo.Constraint(rule=init_input_output_on)

                # min part load relation
                def init_min_partload(const):
                    return (
                        self.input[t, "electricity"]
                        >= min_part_load * b_tec.var_size * rated_capacity
                    )

                dis.const_min_partload = pyo.Constraint(rule=init_min_partload)

        b_tec.dis_input_output = gdp.Disjunct(
            self.set_t_performance, s_indicators, rule=init_input_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]

        b_tec.disjunction_input_output = gdp.Disjunction(
            self.set_t_performance, rule=bind_disjunctions
        )



    def _performance_function_type_3(self, b_tec):
        """
        Piece-wise linear, minimal partload
        :param b_tec: technology block
        :return: technology block
        """
        # Get performance parameters
        coeff_td = self.processed_coeff.time_dependent_used
        coeff_ti = self.processed_coeff.time_independent
        alpha1 = coeff_td["alpha1"]
        alpha2 = coeff_td["alpha2"]
        bp_x = coeff_td["bp_x"]
        min_part_load = coeff_ti["min_part_load"]
        rated_capacity = coeff_ti["rated_capacity"]

        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off

                def init_input_off(const):
                    return self.input[t, "electricity"] == 0

                dis.const_input_off = pyo.Constraint(rule=init_input_off)

                def init_output_off(const):
                    return self.output[t, "heat"] == 0

                dis.const_output_off = pyo.Constraint(rule=init_output_off)

            else:  # piecewise definition

                def init_input_on1(const):
                    return (
                        self.input[t, "electricity"]
                        >= bp_x[t - 1, ind] * b_tec.var_size * rated_capacity
                    )

                dis.const_input_on1 = pyo.Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return (
                        self.input[t, "electricity"]
                        <= bp_x[t - 1, ind + 1] * b_tec.var_size * rated_capacity
                    )

                dis.const_input_on2 = pyo.Constraint(rule=init_input_on2)

                def init_output_on(const):
                    return (
                        self.output[t, "heat"]
                        == alpha1[t - 1, ind - 1] * self.input[t, "electricity"]
                        + alpha2[t - 1, ind - 1] * b_tec.var_size * rated_capacity
                    )

                dis.const_input_output_on = pyo.Constraint(rule=init_output_on)

                # min part load relation
                def init_min_partload(const):
                    return (
                        self.input[t, "electricity"]
                        >= min_part_load * b_tec.var_size * rated_capacity
                    )

                dis.const_min_partload = pyo.Constraint(rule=init_min_partload)

        b_tec.dis_input_output = gdp.Disjunct(
            self.set_t_performance, s_indicators, rule=init_input_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]

        b_tec.disjunction_input_output = gdp.Disjunction(
            self.set_t_performance, rule=bind_disjunctions
        )

