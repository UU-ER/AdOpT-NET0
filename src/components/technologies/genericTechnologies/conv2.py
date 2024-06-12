import pyomo.environ as pyo
import pyomo.gdp as gdp
from warnings import warn
import pandas as pd

from ..genericTechnologies.fitting_classes import (
    FitGenericTecTypeType1,
    FitGenericTecTypeType2,
    FitGenericTecTypeType34,
)
from ..technology import Technology
from ...utilities import link_full_resolution_to_clustered


class Conv2(Technology):
    """
    Technology with full input substitution

    This technology type resembles a technology with full input substitution,
    but different performance functions for the respective output carriers,
    i.e. :math:`output_{car} = f_{car}(\sum(inputs))`. Three different performance
    function fits are possible.

    **Constraint declarations:**

    - Size constraints are formulated on the input.

      .. math::
         \sum(Input_{t, car}) \leq S

    - It is possible to limit the maximum input of a carrier. This needs to be specified in the technology JSON files.
      Then it holds:

      .. math::
        Input_{t, car} <= max_in_{car} * \sum(Input_{t, car})

    - ``performance_function_type == 1``: Linear through origin, i.e.:

      .. math::
        Output_{t, car} == {\\alpha}_{1, car} \sum(Input_{t, car})

      .. math::
        \min_part_load * S \leq {\\alpha}_1 \sum(Input_{t, car})

    - ``performance_function_type == 2``: Linear with minimal partload (makes big-m transformation required). If the
      technology is in on, it holds:

      .. math::
        Output_{t, car} = {\\alpha}_{1, car} \sum(Input_{t, car}) + {\\alpha}_{2, car}

      .. math::
        \sum(Input_{car}) \geq Input_{min} * S

      If the technology is off, input and output is set to 0:

      .. math::
         Output_{t, car} = 0

      .. math::
         \sum(Input_{t, car}) = 0

      If the technology has a standby-power, the input of the standy-by power carrier
      is:

      .. math::
         Input_{t, standby-carrier} = standbypower * S

    - ``performance_function_type == 3``: Piecewise linear performance function (
      makes big-m transformation required). The same constraints as for
      ``performance_function_type == 2`` with the exception that the performance
      function is defined piecewise for the respective number of pieces.

    - ``performance_function_type == 4``:Piece-wise linear, minimal partload,
      includes constraints for slow (>1h) startup and shutdown trajectories.
      Based on Equations 9-11, 13 and 15 in Morales-España, G., Ramírez-Elizondo, L.,
      & Hobbs, B. F. (2017). Hidden power system inflexibilities imposed by
      traditional unit commitment formulations. Applied Energy, 191, 223–238.
      https://doi.org/10.1016/J.APENERGY.2017.01.089

    - Additionally, ramping rates of the technology can be constraint.

      .. math::
         -rampingrate \leq \sum(Input_{t, car}) - \sum(Input_{t-1, car})

    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.component_options.emissions_based_on = "input"
        self.component_options.size_based_on = "input"
        self.component_options.main_input_carrier = tec_data["Performance"][
            "main_input_carrier"
        ]

        # Initialize fitting class
        if self.component_options.performance_function_type == 1:
            self.fitting_class = FitGenericTecTypeType1(self.component_options)
        elif self.component_options.performance_function_type == 2:
            self.fitting_class = FitGenericTecTypeType2(self.component_options)
        elif (
            self.component_options.performance_function_type == 3
            or self.component_options.performance_function_type == 4
        ):
            self.fitting_class = FitGenericTecTypeType34(self.component_options)
        else:
            raise Exception(
                "performance_function_type must be an integer between 1 and 4"
            )

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits conversion technology type 2

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        super(Conv2, self).fit_technology_performance(climate_data, location)

        if self.component_options.size_based_on == "output":
            raise Exception("size_based_on == output for CONV2 not possible.")

        # fit coefficients
        self.processed_coeff.time_independent["fit"] = (
            self.fitting_class.fit_performance_function(
                self.input_parameters.performance_data["performance"]
            )
        )

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(Conv2, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        self.bounds["input"] = self.fitting_class.calculate_input_bounds(
            self.component_options.size_based_on, time_steps
        )
        self.bounds["output"] = self.fitting_class.calculate_output_bounds(
            self.component_options.size_based_on, time_steps
        )

    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type CONV2

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(Conv2, self).construct_tech_model(
            b_tec, data, set_t_full, set_t_clustered
        )

        # DATA OF TECHNOLOGY
        coeff_ti = self.processed_coeff.time_independent
        dynamics = self.processed_coeff.dynamics
        rated_power = self.input_parameters.rated_power

        if self.component_options.performance_function_type == 1:
            b_tec = self._performance_function_type_1(b_tec)
        elif self.component_options.performance_function_type == 2:
            b_tec = self._performance_function_type_2(b_tec)
        elif self.component_options.performance_function_type == 3:
            b_tec = self._performance_function_type_3(b_tec)
        elif self.component_options.performance_function_type == 4:
            b_tec = self._performance_function_type_4(b_tec)

        # Size constraints
        # size constraint based on sum of inputs
        def init_size_constraint(const, t):
            return (
                sum(self.input[t, car_input] for car_input in b_tec.set_input_carriers)
                <= b_tec.var_size * rated_power
            )

        b_tec.const_size = pyo.Constraint(
            self.set_t_performance, rule=init_size_constraint
        )

        # Maximum input of carriers
        if "max_input" in coeff_ti:
            b_tec.set_max_input_carriers = pyo.Set(
                initialize=list(coeff_ti["max_input"].keys())
            )

            def init_max_input(const, t, car):
                return self.input[t, car] <= coeff_ti["max_input"][car] * sum(
                    self.input[t, car_input] for car_input in b_tec.set_input_carriers
                )

            b_tec.const_max_input = pyo.Constraint(
                self.set_t_performance,
                b_tec.set_max_input_carriers,
                rule=init_max_input,
            )

        # RAMPING RATES
        if "ramping_time" in dynamics:
            if not dynamics["ramping_time"] == -1:
                b_tec = self._define_ramping_rates(b_tec, data)

        return b_tec

    def _performance_function_type_1(self, b_tec):
        """
        Linear, no minimal partload, through origin

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        # Performance parameter:
        rated_power = self.input_parameters.rated_power
        coeff_ti = self.processed_coeff.time_independent
        min_part_load = coeff_ti["min_part_load"]

        alpha1 = {}
        for car in coeff_ti["fit"]:
            alpha1[car] = coeff_ti["fit"][car]["alpha1"]
        min_part_load = coeff_ti["min_part_load"]

        # Input-output correlation
        def init_input_output(const, t, car_output):
            return self.output[t, car_output] == alpha1[car_output] * sum(
                self.input[t, car_input] for car_input in b_tec.set_input_carriers
            )

        b_tec.const_input_output = pyo.Constraint(
            self.set_t_performance, b_tec.set_output_carriers, rule=init_input_output
        )

        if min_part_load > 0:

            def init_min_part_load(const, t):
                return min_part_load * b_tec.var_size * rated_power <= sum(
                    self.input[t, car_input] for car_input in b_tec.set_input_carriers
                )

            b_tec.const_min_part_load = pyo.Constraint(
                self.set_t_performance, rule=init_min_part_load
            )

        return b_tec

    def _performance_function_type_2(self, b_tec):
        """
        Linear, minimal partload

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        # Transformation required
        self.big_m_transformation_required = 1

        # Performance parameter:
        rated_power = self.input_parameters.rated_power
        coeff_ti = self.processed_coeff.time_independent
        alpha1 = {}
        alpha2 = {}
        for car in coeff_ti["fit"]:
            alpha1[car] = coeff_ti["fit"][car]["alpha1"]
            alpha2[car] = coeff_ti["fit"][car]["alpha2"]
        min_part_load = coeff_ti["min_part_load"]
        standby_power = coeff_ti["standby_power"]

        if standby_power != -1:
            if self.component_options.standby_power_carrier == -1:
                car_standby_power = self.component_options.main_input_carrier
            else:
                car_standby_power = self.component_options.standby_power_carrier

        if not b_tec.find_component("var_x"):
            b_tec.var_x = pyo.Var(
                self.set_t_performance, domain=pyo.NonNegativeReals, bounds=(0, 1)
            )

        if min_part_load == 0:
            warn(
                "Having performance_function_type = 2 with no part-load usually makes no sense. Error occured for "
                + self.name
            )

        # define disjuncts
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off

                dis.const_x_off = pyo.Constraint(expr=b_tec.var_x[t] == 0)

                if standby_power == -1:

                    def init_input_off(const, car_input):
                        return self.input[t, car_input] == 0

                    dis.const_input = pyo.Constraint(
                        b_tec.set_input_carriers, rule=init_input_off
                    )
                else:

                    def init_standby_power(const, car_input):
                        if car_input == self.component_options.main_input_carrier:
                            return (
                                self.input[t, car_standby_power]
                                == standby_power * b_tec.var_size * rated_power
                            )
                        else:
                            return self.input[t, car_input] == 0

                    dis.const_input = pyo.Constraint(
                        b_tec.set_input_carriers, rule=init_standby_power
                    )

                def init_output_off(const, car_output):
                    return self.output[t, car_output] == 0

                dis.const_output_off = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_off
                )

            else:  # technology on

                dis.const_x_on = pyo.Constraint(expr=b_tec.var_x[t] == 1)

                # input-output relation
                def init_input_output_on(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output]
                        * sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        + alpha2[car_output] * b_tec.var_size * rated_power
                    )

                dis.const_input_output_on = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_input_output_on
                )

                # min part load relation
                def init_min_partload(const):
                    return (
                        sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        >= min_part_load * b_tec.var_size * rated_power
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

        return b_tec

    def _performance_function_type_3(self, b_tec):
        """
        Piece-wise linear, minimal partload

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        # Transformation required
        self.big_m_transformation_required = 1

        # Performance parameter:
        rated_power = self.input_parameters.rated_power
        coeff_ti = self.processed_coeff.time_independent
        alpha1 = {}
        alpha2 = {}
        for car in coeff_ti["fit"]:
            alpha1[car] = coeff_ti["fit"][car]["alpha1"]
            alpha2[car] = coeff_ti["fit"][car]["alpha2"]
            bp_x = coeff_ti["fit"][car]["bp_x"]
        min_part_load = coeff_ti["min_part_load"]
        standby_power = coeff_ti["standby_power"]

        if standby_power != -1:
            if self.component_options.standby_power_carrier == -1:
                car_standby_power = self.component_options.main_input_carrier
            else:
                car_standby_power = self.component_options.standby_power_carrier

        if not b_tec.find_component("var_x"):
            b_tec.var_x = pyo.Var(
                self.set_t_performance, domain=pyo.NonNegativeReals, bounds=(0, 1)
            )

        s_indicators = range(0, len(bp_x))

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off

                dis.const_x_off = pyo.Constraint(expr=b_tec.var_x[t] == 0)

                if standby_power == -1:

                    def init_input_off(const, car_input):
                        return self.input[t, car_input] == 0

                    dis.const_input_off = pyo.Constraint(
                        b_tec.set_input_carriers, rule=init_input_off
                    )

                else:

                    def init_standby_power(const, car_input):
                        if car_input == self.component_options.main_input_carrier:
                            return (
                                self.input[t, car_standby_power]
                                == standby_power * b_tec.var_size * rated_power
                            )
                        else:
                            return self.input[t, car_input] == 0

                    dis.const_input = pyo.Constraint(
                        b_tec.set_input_carriers, rule=init_standby_power
                    )

                def init_output_off(const, car_output):
                    return self.output[t, car_output] == 0

                dis.const_output_off = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_off
                )

            else:  # piecewise definition

                dis.const_x_on = pyo.Constraint(expr=b_tec.var_x[t] == 1)

                def init_input_on1(const):
                    return (
                        sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        >= bp_x[ind - 1] * b_tec.var_size * rated_power
                    )

                dis.const_input_on1 = pyo.Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return (
                        sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        <= bp_x[ind] * b_tec.var_size * rated_power
                    )

                dis.const_input_on2 = pyo.Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output][ind - 1]
                        * sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        + alpha2[car_output][ind - 1] * b_tec.var_size * rated_power
                    )

                dis.const_input_output_on = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_on
                )

                # min part load relation
                def init_min_partload(const):
                    return (
                        sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        >= min_part_load * b_tec.var_size * rated_power
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

        return b_tec

    def _performance_function_type_4(self, b_tec):
        """
        Piece-wise linear, minimal partload, includes constraints for slow (>1h) startup and shutdown trajectories.

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        # Transformation required
        self.big_m_transformation_required = 1

        # Performance parameter:
        rated_power = self.input_parameters.rated_power
        coeff_ti = self.processed_coeff.time_independent
        dynamics = self.processed_coeff.dynamics
        alpha1 = {}
        alpha2 = {}
        for car in coeff_ti["fit"]:
            alpha1[car] = coeff_ti["fit"][car]["alpha1"]
            alpha2[car] = coeff_ti["fit"][car]["alpha2"]
            bp_x = coeff_ti["fit"][car]["bp_x"]
        min_part_load = coeff_ti["min_part_load"]
        SU_time = dynamics["SU_time"]
        SD_time = dynamics["SD_time"]

        if SU_time <= 0 and SD_time <= 0:
            warn(
                "Having performance_function_type = 4 with no slow SU/SDs usually makes no sense."
            )
        elif SU_time < 0:
            SU_time = 0
        elif SD_time < 0:
            SD_time = 0

        # Calculate SU and SD trajectories
        if SU_time > 0:
            SU_trajectory = []
            for i in range(1, SU_time + 1):
                SU_trajectory.append((min_part_load / (SU_time + 1)) * i)

        if SD_time > 0:
            SD_trajectory = []
            for i in range(1, SD_time + 1):
                SD_trajectory.append((min_part_load / (SD_time + 1)) * i)
            SD_trajectory = sorted(SD_trajectory, reverse=True)

        # slow startups/shutdowns with trajectories
        s_indicators = range(0, SU_time + SD_time + len(bp_x))

        def init_SUSD_trajectories(dis, t, ind):
            if ind == 0:  # technology off
                dis.const_x_off = pyo.Constraint(expr=b_tec.var_x[t] == 0)

                def init_y_off(const, i):
                    if t < len(self.set_t_full) - SU_time or i > SU_time - (
                        len(self.set_t_full) - t
                    ):
                        return b_tec.var_y[t - i + SU_time + 1] == 0
                    else:
                        return (
                            b_tec.var_y[(t - i + SU_time + 1) - len(self.set_t_full)]
                            == 0
                        )

                dis.const_y_off = pyo.Constraint(range(1, SU_time + 1), rule=init_y_off)

                def init_z_off(const, j):
                    if j <= t:
                        return b_tec.var_z[t - j + 1] == 0
                    else:
                        return b_tec.var_z[len(self.set_t_full) + (t - j + 1)] == 0

                dis.const_z_off = pyo.Constraint(range(1, SD_time + 1), rule=init_z_off)

                def init_input_off(const, car_input):
                    return self.input[t, car_input] == 0

                dis.const_input_off = pyo.Constraint(
                    b_tec.set_input_carriers, rule=init_input_off
                )

                def init_output_off(const, car_output):
                    return self.output[t, car_output] == 0

                dis.const_output_off = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_off
                )

            elif ind in range(1, SU_time + 1):  # technology in startup
                dis.const_x_off = pyo.Constraint(expr=b_tec.var_x[t] == 0)

                def init_y_on(const):
                    if t < len(self.set_t_full) - SU_time or ind > SU_time - (
                        len(self.set_t_full) - t
                    ):
                        return b_tec.var_y[t - ind + SU_time + 1] == 1
                    else:
                        return (
                            b_tec.var_y[(t - ind + SU_time + 1) - len(self.set_t_full)]
                            == 1
                        )

                dis.const_y_on = pyo.Constraint(rule=init_y_on)

                def init_z_off(const):
                    if t < len(self.set_t_full) - SU_time or ind > SU_time - (
                        len(self.set_t_full) - t
                    ):
                        return b_tec.var_z[t - ind + SU_time + 1] == 0
                    else:
                        return (
                            b_tec.var_z[(t - ind + SU_time + 1) - len(self.set_t_full)]
                            == 0
                        )

                dis.const_z_off = pyo.Constraint(rule=init_z_off)

                def init_input_SU(cons):
                    return (
                        sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        == b_tec.var_size * SU_trajectory[ind - 1]
                    )

                dis.const_input_SU = pyo.Constraint(rule=init_input_SU)

                def init_output_SU(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output][0]
                        * sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        + alpha2[car_output][0] * b_tec.var_size * rated_power
                    )

                dis.const_output_SU = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_SU
                )

            elif ind in range(
                SU_time + 1, SU_time + SD_time + 1
            ):  # technology in shutdown
                ind_SD = ind - SU_time
                dis.const_x_off = pyo.Constraint(expr=b_tec.var_x[t] == 0)

                def init_z_on(const):
                    if ind_SD <= t:
                        return b_tec.var_z[t - ind_SD + 1] == 1
                    else:
                        return b_tec.var_z[len(self.set_t_full) + (t - ind_SD + 1)] == 1

                dis.const_z_on = pyo.Constraint(rule=init_z_on)

                def init_y_off(const):
                    if ind_SD <= t:
                        return b_tec.var_y[t - ind_SD + 1] == 0
                    else:
                        return b_tec.var_y[len(self.set_t_full) + (t - ind_SD + 1)] == 0

                dis.const_y_off = pyo.Constraint(rule=init_y_off)

                def init_input_SD(const):
                    return (
                        sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        == b_tec.var_size * SD_trajectory[ind_SD - 1]
                    )

                dis.const_input_SD = pyo.Constraint(rule=init_input_SD)

                def init_output_SD(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output][0]
                        * sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        + alpha2[car_output][0] * b_tec.var_size * rated_power
                    )

                dis.const_output_SD = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_SD
                )

            elif ind > SU_time + SD_time:
                ind_bpx = ind - (SU_time + SD_time)
                dis.const_x_on = pyo.Constraint(expr=b_tec.var_x[t] == 1)

                def init_input_on1(const):
                    return (
                        sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        >= bp_x[ind_bpx - 1] * b_tec.var_size * rated_power
                    )

                dis.const_input_on1 = pyo.Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return (
                        sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        <= bp_x[ind_bpx] * b_tec.var_size * rated_power
                    )

                dis.const_input_on2 = pyo.Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output][ind_bpx - 1]
                        * sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        + alpha2[car_output][ind_bpx - 1] * b_tec.var_size * rated_power
                    )

                dis.const_input_output_on = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_on
                )

                # min part load relation
                def init_min_partload(const):
                    return (
                        sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        >= min_part_load * b_tec.var_size * rated_power
                    )

                dis.const_min_partload = pyo.Constraint(rule=init_min_partload)

        b_tec.dis_SUSD_trajectory = gdp.Disjunct(
            self.set_t_full, s_indicators, rule=init_SUSD_trajectories
        )

        def bind_disjunctions_SUSD(dis, t):
            return [b_tec.dis_SUSD_trajectory[t, k] for k in s_indicators]

        b_tec.disjunction_SUSD_traject = gdp.Disjunction(
            self.set_t_full, rule=bind_disjunctions_SUSD
        )

        return b_tec

    def _define_ramping_rates(self, b_tec, data):
        """
        Constraints the inputs for a ramping rate

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        dynamics = self.processed_coeff.dynamics

        ramping_time = dynamics["ramping_time"]

        # Calculate ramping rates
        if "ref_size" in dynamics and not dynamics["ref_size"] == -1:
            ramping_rate = dynamics["ref_size"] / ramping_time
        else:
            ramping_rate = b_tec.var_size / ramping_time

        # Constraints ramping rates
        if (
            not self.component_options.performance_function_type == 1
            and "ramping_const_int" in dynamics
            and dynamics["ramping_const_int"] == 1
        ):

            s_indicators = range(0, 3)

            def init_ramping_operation_on(dis, t, ind):
                if t > 1:
                    if ind == 0:  # ramping constrained
                        dis.const_ramping_on = pyo.Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 0
                        )

                        def init_ramping_down_rate_operation(const):
                            return -ramping_rate <= sum(
                                self.input[t, car_input] - self.input[t - 1, car_input]
                                for car_input in b_tec.set_input_carriers
                            )

                        dis.const_ramping_down_rate = pyo.Constraint(
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

                        dis.const_ramping_up_rate = pyo.Constraint(
                            rule=init_ramping_up_rate_operation
                        )

                    elif ind == 1:  # startup, no ramping constraint
                        dis.const_ramping_on = pyo.Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 1
                        )

                    else:  # shutdown, no ramping constraint
                        dis.const_ramping_on = pyo.Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == -1
                        )

            b_tec.dis_ramping_operation_on = gdp.Disjunct(
                self.set_t_performance, s_indicators, rule=init_ramping_operation_on
            )

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_ramping_operation_on[t, i] for i in s_indicators]

            b_tec.disjunction_ramping_operation_on = gdp.Disjunction(
                self.set_t_performance, rule=bind_disjunctions
            )

        else:
            if data["config"]["optimization"]["typicaldays"]["N"]["value"] == 0:
                input_aux_rr = self.input
                set_t_rr = self.set_t_performance
            else:
                if (
                    data["config"]["optimization"]["typicaldays"]["method"]["value"]
                    == 1
                ):
                    sequence = data["k_means_specs"]["sequence"]
                elif (
                    data["config"]["optimization"]["typicaldays"]["method"]["value"]
                    == 2
                ):
                    sequence = self.sequence

                # init bounds at full res
                bounds_rr_full = {
                    "input": self.fitting_class.calculate_input_bounds(
                        self.component_options.size_based_on, len(self.set_t_full)
                    )
                }

                # create input variable for full res
                def init_input_bounds(bounds, t, car):
                    return tuple(
                        bounds_rr_full["input"][car][t - 1, :]
                        * self.processed_coeff.time_independent["size_max"]
                        * self.processed_coeff.time_independent["rated_power"]
                    )

                b_tec.var_input_rr_full = pyo.Var(
                    self.set_t_full,
                    b_tec.set_input_carriers,
                    within=pyo.NonNegativeReals,
                    bounds=init_input_bounds,
                )

                b_tec.const_link_full_resolution_rr = link_full_resolution_to_clustered(
                    self.input,
                    b_tec.var_input_rr_full,
                    self.set_t_full,
                    sequence,
                    b_tec.set_input_carriers,
                )

                input_aux_rr = b_tec.var_input_rr_full
                set_t_rr = self.set_t_full

            # Ramping constraint without integers
            def init_ramping_down_rate(const, t):
                if t > 1:
                    return -ramping_rate <= sum(
                        input_aux_rr[t, car_input] - input_aux_rr[t - 1, car_input]
                        for car_input in b_tec.set_input_carriers
                    )
                else:
                    return pyo.Constraint.Skip

            b_tec.const_ramping_down_rate = pyo.Constraint(
                set_t_rr, rule=init_ramping_down_rate
            )

            def init_ramping_up_rate(const, t):
                if t > 1:
                    return (
                        sum(
                            input_aux_rr[t, car_input] - input_aux_rr[t - 1, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        <= ramping_rate
                    )
                else:
                    return pyo.Constraint.Skip

            b_tec.const_ramping_up_rate = pyo.Constraint(
                set_t_rr, rule=init_ramping_up_rate
            )

        return b_tec
