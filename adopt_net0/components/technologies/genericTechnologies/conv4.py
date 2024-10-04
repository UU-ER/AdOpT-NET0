import pyomo.environ as pyo
import pyomo.gdp as gdp
import numpy as np
import pandas as pd

from ...component import InputParameters
from ..technology import Technology


class Conv4(Technology):
    """
    Technology with no inputs

    This technology type resembles a technology with fixed output ratios  and no
    inputs, i.e., :math:`output_{car} \\leq S`. This technology is useful for
    modelling a technology for which you do not care about the inputs, i.e., you do not
    wish to construct and solve an energy balance for the input carriers.
    Two different performance function fits are possible.

    **Constraint declarations:**

    For all performance function types, the following constraints hold:

    - Size constraints are formulated on the output.

      .. math::
         Output_{t, maincarrier} \\leq S

    - The ratios of outputs are fixed and given as:

      .. math::
        Output_{t, car} = {\\phi}_{car} * Output_{t, maincarrier}

    - ``performance_function_type == 1``: No further constraints on the performance
      of the technology.

    - ``performance_function_type == 2``: A minimum part load can be specified (
      requiring a big-m transformation for the solving). The following constraints hold:

      When the technology is on:

      .. math::
        Output_{maincarrier} \\geq Output_{min} * S

      When the technology is off, output is set to 0:

      .. math::
         Output_{t, car} = 0
    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.component_options.emissions_based_on = "output"
        self.component_options.main_output_carrier = tec_data["Performance"][
            "main_output_carrier"
        ]

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits conversion technology type 4

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        super(Conv4, self).fit_technology_performance(climate_data, location)

        # Coefficients
        phi = {}
        for car in self.input_parameters.performance_data["output_ratios"]:
            phi[car] = self.input_parameters.performance_data["output_ratios"][car]
        self.processed_coeff.time_independent["phi"] = phi

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(Conv4, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        # Output Bounds
        self.bounds["output"][
            self.input_parameters.performance_data["main_output_carrier"]
        ] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=(time_steps)),
            )
        )

        for car in self.component_options.output_carrier:
            if not car == self.component_options.main_output_carrier:
                self.bounds["output"][car] = (
                    self.bounds["output"][self.component_options.main_output_carrier]
                    * self.input_parameters.performance_data["output_ratios"][car]
                )

    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type CONV4

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(Conv4, self).construct_tech_model(
            b_tec, data, set_t_full, set_t_clustered
        )

        # DATA OF TECHNOLOGY
        coeff_ti = self.processed_coeff.time_independent
        rated_power = self.input_parameters.rated_power
        phi = coeff_ti["phi"]

        # add additional constraints for performance type 2 (min. part load)
        if self.component_options.performance_function_type == 2:
            b_tec = self._performance_function_type_2(b_tec)

        # Size constraints
        # constraint on output ratios
        def init_output_output(const, t, car_output):
            if car_output == self.component_options.main_output_carrier:
                return pyo.Constraint.Skip
            else:
                return (
                    self.output[t, car_output]
                    == phi[car_output]
                    * self.output[t, self.component_options.main_output_carrier]
                )

        b_tec.const_output_output = pyo.Constraint(
            self.set_t_performance, b_tec.set_output_carriers, rule=init_output_output
        )

        # size constraint based on main carrier output
        def init_size_constraint(const, t):
            return (
                self.output[t, self.component_options.main_output_carrier]
                <= b_tec.var_size * rated_power
            )

        b_tec.const_size = pyo.Constraint(
            self.set_t_performance, rule=init_size_constraint
        )

        return b_tec

    def _performance_function_type_2(self, b_tec):
        """
        Sets the minimum part load constraint for a tec based on tec_type CONV4 with
        performance type 2.

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        # Transformation required
        self.big_m_transformation_required = 1

        # Performance Parameters
        rated_power = self.input_parameters.rated_power
        coeff_ti = self.processed_coeff.time_independent
        min_part_load = coeff_ti["min_part_load"]

        # define disjuncts
        s_indicators = range(0, 2)

        def init_output(dis, t, ind):
            if ind == 0:  # technology off

                def init_output_off(const, car_output):
                    return self.output[t, car_output] == 0

                dis.const_output_off = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_off
                )

            else:  # technology on

                def init_min_partload(const):
                    return (
                        self.output[t, self.component_options.main_output_carrier]
                        >= min_part_load * b_tec.var_size * rated_power
                    )

                dis.const_min_partload = pyo.Constraint(rule=init_min_partload)

        b_tec.dis_output = gdp.Disjunct(
            self.set_t_performance, s_indicators, rule=init_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_output[t, i] for i in s_indicators]

        b_tec.disjunction_output = gdp.Disjunction(
            self.set_t_performance, rule=bind_disjunctions
        )

        return b_tec
