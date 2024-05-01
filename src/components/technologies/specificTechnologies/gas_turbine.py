from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

from ..utilities import FittedPerformance, fit_piecewise_function
from ..technology import Technology


class GasTurbine(Technology):
    """
    Resembles gas turbines of different sizes.
    Hydrogen and Natural Gas Turbines are possible at four different sizes, as indicated by the file names
    of the data. Performance data and the model is taken from Weimann, L., Ellerker, M., Kramer, G. J., &
    Gazzani, M. (2019). Modeling gas turbines in multi-energy systems: A linear model accounting for part-load
    operation, fuel, temperature, and sizing effects. International Conference on Applied Energy.
    https://doi.org/10.46855/energy-proceedings-5280

    A small adaption is made: Natural gas turbines can co-fire hydrogen up to 5% of the energy content

    **Parameter declarations:**

    - :math:`Input_{min}`: Minimal input per turbine

    - :math:`Input_{max}`: Maximal input per turbine

    - :math:`in_{H2max}`: Maximal H2 admixture to fuel (only for natural gas turbines, default is 0.05)

    - :math:`{\\alpha}`: Performance parameter for electricity output

    - :math:`{\\beta}`: Performance parameter for electricity output

    - :math:`{\\epsilon}`: Performance parameter for heat output

    - :math:`f({\\Theta})`: Ambient temperature correction factor

    **Variable declarations:**

    - Total fuel input in :math:`t`: :math:`Input_{tot, t}`

    - Number of turbines on in :math:`t`: :math:`N_{on,t}`

    **Constraint declarations:**

    - Input calculation (For hydrogen turbines, :math:`Input_{NG, t}` is zero, and the second constraint is removed):

      .. math::
        Input_{H2, t} + Input_{NG, t} = Input_{tot, t}

      .. math::
        Input_{H2, t} \leq in_{H2max} Input_{tot, t}

    - Turbines on:

      .. math::
        N_{on, t} \leq S

    - If technology is on:

      .. math::
        Output_{el,t} = ({\\alpha} Input_{tot, t} + {\\beta} * N_{on, t}) *f({\\Theta})

      .. math::
        Output_{th,t} = {\\epsilon} Input_{tot, t} - Output_{el,t}

      .. math::
        Input_{min} * N_{on, t} \leq Input_{tot, t} \leq Input_{max} * N_{on, t}

    - If the technology is off, input and output is set to 0:

      .. math::
         \sum(Output_{t, car}) = 0

      .. math::
         \sum(Input_{t, car}) = 0
    """

    def __init__(self, tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()

    def fit_technology_performance(self, climate_data, location):
        """
        Performs fitting for technology type GasTurbine

        The equations and data are based on Weimann, L., Ellerker, M., Kramer, G. J., & Gazzani, M. (2019). Modeling gas
        turbines in multi-energy systems: A linear model accounting for part-load operation, fuel, temperature,
        and sizing effects. International Conference on Applied Energy. https://doi.org/10.46855/energy-proceedings-5280

        :param tec_data: technology data
        :param climate_data: climate data
        :return:
        """
        # Climate data & Number of timesteps
        time_steps = len(climate_data)

        # Ambient air temperature
        T = copy.deepcopy(climate_data["temp_air"])

        # Temperature correction factors
        f = np.empty(shape=(time_steps))
        f[T <= 6] = (
            self.performance_data["gamma"][0]
            * (T[T <= 6] / self.performance_data["T_iso"])
            + self.performance_data["delta"][0]
        )
        f[T > 6] = (
            self.performance_data["gamma"][1]
            * (T[T > 6] / self.performance_data["T_iso"])
            + self.performance_data["delta"][1]
        )

        # Derive return
        fit = {}
        fit["coeff"] = {}
        fit["coeff"]["f"] = f.round(5)
        fit["coeff"]["alpha"] = round(self.performance_data["alpha"], 5)
        fit["coeff"]["beta"] = round(self.performance_data["beta"], 5)
        fit["coeff"]["epsilon"] = round(self.performance_data["epsilon"], 5)
        fit["coeff"]["in_min"] = round(self.performance_data["in_min"], 5)
        fit["coeff"]["in_max"] = round(self.performance_data["in_max"], 5)
        if len(self.performance_data["input_carrier"]) == 2:
            fit["coeff"]["max_H2_admixture"] = self.performance_data["max_H2_admixture"]
        else:
            fit["coeff"]["max_H2_admixture"] = 1

        # Input bounds
        fit["input_bounds"] = {}
        for c in self.performance_data["input_carrier"]:
            if c == "hydrogen":
                fit["input_bounds"][c] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps))
                        * self.performance_data["in_max"]
                        * fit["coeff"]["max_H2_admixture"],
                    )
                )
            else:
                fit["input_bounds"][c] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps)) * self.performance_data["in_max"],
                    )
                )

        # Output bounds
        fit["output_bounds"] = {}
        fit["output_bounds"]["electricity"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                f
                * (
                    self.performance_data["in_max"] * fit["coeff"]["alpha"]
                    + fit["coeff"]["beta"]
                ),
            )
        )
        fit["output_bounds"]["heat"] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                fit["coeff"]["epsilon"] * fit["coeff"]["in_max"]
                - f
                * (
                    self.performance_data["in_max"] * fit["coeff"]["alpha"]
                    + fit["coeff"]["beta"]
                ),
            )
        )

        # Output Bounds
        self.fitted_performance.bounds["output"] = fit["output_bounds"]
        # Input Bounds
        for car in self.performance_data["input_carrier"]:
            self.fitted_performance.bounds["input"][car] = np.column_stack(
                (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
            )
        # Coefficients
        self.fitted_performance.coefficients = fit["coeff"]
        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 1

    def construct_tech_model(self, b_tec, data, set_t, set_t_clustered):
        """
        Adds constraints to technology blocks for gas turbines

        :param obj b_tec: technology block
        :param Energyhub energyhub: energyhub instance
        :return: technology block
        """
        super(GasTurbine, self).construct_tech_model(
            b_tec, data, set_t, set_t_clustered
        )

        # Transformation required
        self.big_m_transformation_required = 1

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        coeff = self.fitted_performance.coefficients
        bounds = self.fitted_performance.bounds

        # Parameter declaration
        in_min = coeff["in_min"]
        in_max = coeff["in_max"]
        max_H2_admixture = coeff["max_H2_admixture"]
        alpha = coeff["alpha"]
        beta = coeff["beta"]
        epsilon = coeff["epsilon"]
        f = coeff["f"]

        # Additional decision variables
        size_max = self.size_max

        def init_input_bounds(bd, t):
            if len(performance_data["input_carrier"]) == 2:
                car = "gas"
            else:
                car = "hydrogen"
            return tuple(bounds["input"][car][t - 1, :] * size_max)

        b_tec.var_total_input = Var(
            self.set_t, within=NonNegativeReals, bounds=init_input_bounds
        )

        b_tec.var_units_on = Var(
            self.set_t, within=NonNegativeIntegers, bounds=(0, size_max)
        )

        # Calculate total input
        def init_total_input(const, t):
            return b_tec.var_total_input[t] == sum(
                self.input[t, car_input] for car_input in b_tec.set_input_carriers
            )

        b_tec.const_total_input = Constraint(self.set_t, rule=init_total_input)

        # Constrain hydrogen input
        if len(performance_data["input_carrier"]) == 2:

            def init_h2_input(const, t):
                return (
                    self.input[t, "hydrogen"]
                    <= b_tec.var_total_input[t] * max_H2_admixture
                )

            b_tec.const_h2_input = Constraint(self.set_t, rule=init_h2_input)

        # LINEAR, MINIMAL PARTLOAD
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off

                def init_input_off(const, car):
                    return self.input[t, car] == 0

                dis.const_input = Constraint(
                    b_tec.set_input_carriers, rule=init_input_off
                )

                def init_output_off(const, car):
                    return self.output[t, car] == 0

                dis.const_output_off = Constraint(
                    b_tec.set_output_carriers, rule=init_output_off
                )

            else:  # technology on
                # input-output relation
                def init_input_output_on_el(const):
                    return (
                        self.output[t, "electricity"]
                        == (
                            alpha * b_tec.var_total_input[t]
                            + beta * b_tec.var_units_on[t]
                        )
                        * f.iloc[t - 1]
                    )

                dis.const_input_output_on_el = Constraint(rule=init_input_output_on_el)

                def init_input_output_on_th(const):
                    return (
                        self.output[t, "heat"]
                        == epsilon * b_tec.var_total_input[t]
                        - self.output[t, "electricity"]
                    )

                dis.const_input_output_on_th = Constraint(rule=init_input_output_on_th)

                # min part load relation
                def init_min_input(const):
                    return b_tec.var_total_input[t] >= in_min * b_tec.var_units_on[t]

                dis.const_min_input = Constraint(rule=init_min_input)

                def init_max_input(const):
                    return b_tec.var_total_input[t] <= in_max * b_tec.var_units_on[t]

                dis.const_max_input = Constraint(rule=init_max_input)

        b_tec.dis_input_output = Disjunct(
            self.set_t, s_indicators, rule=init_input_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]

        b_tec.disjunction_input_output = Disjunction(self.set_t, rule=bind_disjunctions)

        # Technologies on
        def init_n_on(const, t):
            return b_tec.var_units_on[t] <= b_tec.var_size

        b_tec.const_n_on = Constraint(self.set_t, rule=init_n_on)

        # RAMPING RATES
        if "ramping_time" in self.performance_data:
            if not self.performance_data["ramping_time"] == -1:
                b_tec = self._define_ramping_rates(b_tec)

        return b_tec

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        super(GasTurbine, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "modules_on", data=[model_block.var_units_on[t].value for t in self.set_t]
        )

    def _define_ramping_rates(self, b_tec):
        """
        Constraints the inputs for a ramping rate

        :param b_tec: technology model block
        :return:
        """
        ramping_time = self.performance_data["ramping_time"]

        # Calculate ramping rates
        if (
            "ref_size" in self.performance_data
            and not self.performance_data["ref_size"] == -1
        ):
            ramping_rate = self.performance_data["ref_size"] / ramping_time
        else:
            ramping_rate = b_tec.var_size / ramping_time

        # Constraints ramping rates
        if (
            "ramping_const_int" in self.performance_data
            and self.performance_data["ramping_const_int"] == 1
        ):

            s_indicators = range(0, 3)

            def init_ramping_operation_on(dis, t, ind):
                if t > 1:
                    if ind == 0:  # ramping constrained
                        dis.const_ramping_on = Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 0
                        )

                        def init_ramping_down_rate_operation(const):
                            return -ramping_rate <= sum(
                                self.input[t, car_input] - self.input[t - 1, car_input]
                                for car_input in b_tec.set_input_carriers
                            )

                        dis.const_ramping_down_rate = Constraint(
                            rule=init_ramping_down_rate_operation
                        )

                        def init_ramping_up_rate_operation(const):
                            return (
                                sum(
                                    self.input[t, car_input]
                                    - self.input[t - 1, car_input]
                                    for car_input in b_tec.set_input_carriers
                                )
                                <= ramping_rate
                            )

                        dis.const_ramping_up_rate = Constraint(
                            rule=init_ramping_up_rate_operation
                        )

                    elif ind == 1:  # startup, no ramping constraint
                        dis.const_ramping_on = Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 1
                        )

                    else:  # shutdown, no ramping constraint
                        dis.const_ramping_on = Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == -1
                        )

            b_tec.dis_ramping_operation_on = Disjunct(
                self.set_t, s_indicators, rule=init_ramping_operation_on
            )

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_ramping_operation_on[t, i] for i in s_indicators]

            b_tec.disjunction_ramping_operation_on = Disjunction(
                self.set_t, rule=bind_disjunctions
            )

        else:

            def init_ramping_down_rate(const, t):
                if t > 1:
                    return -ramping_rate <= sum(
                        self.input[t, car_input] - self.input[t - 1, car_input]
                        for car_input in b_tec.set_input_carriers
                    )
                else:
                    return Constraint.Skip

            b_tec.const_ramping_down_rate = Constraint(
                self.set_t, rule=init_ramping_down_rate
            )

            def init_ramping_up_rate(const, t):
                if t > 1:
                    return (
                        sum(
                            self.input[t, car_input] - self.input[t - 1, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        <= ramping_rate
                    )
                else:
                    return Constraint.Skip

            b_tec.const_ramping_up_rate = Constraint(
                self.set_t, rule=init_ramping_up_rate
            )

        return b_tec
