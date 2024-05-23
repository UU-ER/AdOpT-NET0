import pandas as pd
import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
from warnings import warn

from ..genericTechnologies.utilities import (
    FitGenericTecTypeType1,
    FitGenericTecTypeType2,
    FitGenericTecTypeType34,
)
from ..technology import Technology


class Conv1(Technology):
    """
    This technology type resembles a technology with full input and output substitution,
    i.e. :math:`\sum(output) = f(\sum(inputs))`
    Three different performance function fits are possible.

    **Constraint declarations:**

    - Size constraints can be formulated on the input or the output.
      For size_based_on == 'input' it holds:

      .. math::
         \sum(Input_{t, car}) \leq S

      For size_based_on == 'output' it holds:

      .. math::
         \sum(Output_{t, car}) \leq S

    - It is possible to limit the maximum input of a carrier. This needs to be specified in the technology JSON files.
      Then it holds:

      .. math::
        Input_{t, car} <= max_in_{car} * \sum(Input_{t, car})

    - ``performance_function_type == 1``: Linear through origin, i.e.:

      .. math::
        \sum(Output_{t, car}) == {\\alpha}_1 \sum(Input_{t, car})

    - ``performance_function_type == 2``: Linear with minimal partload (makes big-m transformation required). If the
      technology is in on, it holds:

      .. math::
        \sum(Output_{t, car}) = {\\alpha}_1 \sum(Input_{t, car}) + {\\alpha}_2

      .. math::
        \sum(Input_{car}) \geq Input_{min} * S

      If the technology is off, input and output is set to 0:

      .. math::
         \sum(Output_{t, car}) = 0

      .. math::
         \sum(Input_{t, car}) = 0

    - ``performance_function_type == 3``: Piecewise linear performance function (makes big-m transformation required).
      The same constraints as for ``performance_function_type == 2`` with the exception that the performance function
      is defined piecewise for the respective number of pieces
    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.options.emissions_based_on = "input"
        self.options.size_based_on = tec_data["size_based_on"]
        self.info.main_input_carrier = tec_data["Performance"]["main_input_carrier"]

        # Initialize fitting class
        if self.options.performance_function_type == 1:
            self.f = FitGenericTecTypeType1(self.info)
        elif self.options.performance_function_type == 2:
            self.f = FitGenericTecTypeType2(self.info)
        elif (
            self.options.performance_function_type == 3
            or self.options.performance_function_type == 4
        ):
            self.f = FitGenericTecTypeType34(self.info)
        else:
            raise Exception(
                "performance_function_type must be an integer between 1 and 4"
            )

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits conversion technology type 1 and returns fitted parameters as a dict

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        super(Conv1, self).fit_technology_performance(climate_data, location)

        # reshape parameters for CONV1
        temp = copy.deepcopy(self.parameters.unfitted_data["performance"]["out"])
        self.parameters.unfitted_data["performance"]["out"] = {}
        self.parameters.unfitted_data["performance"]["out"]["out"] = temp

        # fit coefficients
        self.coeff.time_independent["fit"] = self.f.fit_performance_function(
            self.parameters.unfitted_data["performance"]
        )

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(Conv1, self)._calculate_bounds()

        time_steps = len(self.set_t)

        self.bounds["input"] = self.f.calculate_input_bounds(
            self.options.size_based_on, time_steps
        )
        self.bounds["output"] = self.f.calculate_output_bounds(
            self.options.size_based_on, time_steps
        )

    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type CONV1

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(Conv1, self).construct_tech_model(
            b_tec, data, set_t_full, set_t_clustered
        )

        # DATA OF TECHNOLOGY
        c_ti = self.coeff.time_independent
        dynamics = self.coeff.dynamics
        rated_power = self.parameters.rated_power

        # Technology Constraints
        if self.options.performance_function_type == 1:
            b_tec = self._performance_function_type_1(b_tec)
        elif self.options.performance_function_type == 2:
            b_tec = self._performance_function_type_2(b_tec)
        elif self.options.performance_function_type == 3:
            b_tec = self._performance_function_type_3(b_tec)
        elif self.options.performance_function_type == 4:
            b_tec = self._performance_function_type_4(b_tec)

        # Size constraints
        # size constraint based on sum of input/output
        def init_size_constraint(const, t):
            if self.options.size_based_on == "input":
                return (
                    sum(
                        self.input[t, car_input]
                        for car_input in b_tec.set_input_carriers
                    )
                    <= b_tec.var_size * rated_power
                )
            elif self.options.size_based_on == "output":
                return (
                    sum(
                        self.output[t, car_output]
                        for car_output in b_tec.set_output_carriers
                    )
                    <= b_tec.var_size * rated_power
                )

        b_tec.const_size = pyo.Constraint(self.set_t, rule=init_size_constraint)

        # Maximum input of carriers
        if "max_input" in c_ti:
            b_tec.set_max_input_carriers = pyo.Set(
                initialize=list(c_ti["max_input"].keys())
            )

            def init_max_input(const, t, car):
                return self.input[t, car] <= c_ti["max_input"][car] * sum(
                    self.input[t, car_input] for car_input in b_tec.set_input_carriers
                )

            b_tec.const_max_input = pyo.Constraint(
                self.set_t, b_tec.set_max_input_carriers, rule=init_max_input
            )

        # RAMPING RATES
        if "ramping_time" in dynamics:
            if not dynamics["ramping_time"] == -1:
                b_tec = self._define_ramping_rates(b_tec)

        return b_tec

    def _performance_function_type_1(self, b_tec):
        """
        Linear, no minimal partload, through origin

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        # Performance parameter:
        rated_power = self.parameters.rated_power
        c_ti = self.coeff.time_independent
        alpha1 = c_ti["fit"]["out"]["alpha1"]
        min_part_load = c_ti["min_part_load"]

        # Input-output correlation
        def init_input_output(const, t):
            return sum(
                self.output[t, car_output] for car_output in b_tec.set_output_carriers
            ) == alpha1 * sum(
                self.input[t, car_input] for car_input in b_tec.set_input_carriers
            )

        b_tec.const_input_output = pyo.Constraint(self.set_t, rule=init_input_output)

        if min_part_load > 0:

            def init_min_part_load(const, t):
                return min_part_load * b_tec.var_size * rated_power <= sum(
                    self.input[t, car_input] for car_input in b_tec.set_input_carriers
                )

            b_tec.const_min_part_load = pyo.Constraint(
                self.set_t, rule=init_min_part_load
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

        # Performance Parameters
        rated_power = self.parameters.rated_power
        c_ti = self.coeff.time_independent
        alpha1 = c_ti["fit"]["out"]["alpha1"]
        alpha2 = c_ti["fit"]["out"]["alpha2"]
        min_part_load = c_ti["min_part_load"]
        standby_power = c_ti["standby_power"]

        if standby_power != -1:
            if self.options.standby_power_carrier == -1:
                car_standby_power = self.info.main_input_carrier
            else:
                car_standby_power = self.options.standby_power_carrier

        if not b_tec.find_component("var_x"):
            b_tec.var_x = pyo.Var(
                self.set_t_full, domain=pyo.NonNegativeReals, bounds=(0, 1)
            )

        if min_part_load == 0:
            warn(
                "Having performance_function_type = 2 with no part-load usually makes no sense. Error occured for "
                + self.name
            )

        # define disjuncts for on/off
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
                        if car_input == self.info.main_input_carrier:
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
                def init_input_output_on(const):
                    return (
                        sum(
                            self.output[t, car_output]
                            for car_output in b_tec.set_output_carriers
                        )
                        == alpha1
                        * sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        + alpha2 * b_tec.var_size * rated_power
                    )

                dis.const_input_output_on = pyo.Constraint(rule=init_input_output_on)

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
            self.set_t, s_indicators, rule=init_input_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]

        b_tec.disjunction_input_output = gdp.Disjunction(
            self.set_t, rule=bind_disjunctions
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

        # Performance Parameters
        rated_power = self.parameters.rated_power
        c_ti = self.coeff.time_independent
        alpha1 = c_ti["fit"]["out"]["alpha1"]
        alpha2 = c_ti["fit"]["out"]["alpha2"]
        bp_x = c_ti["fit"]["out"]["bp_x"]
        min_part_load = c_ti["min_part_load"]
        standby_power = c_ti["standby_power"]

        if standby_power != -1:
            if self.options.standby_power_carrier == -1:
                car_standby_power = self.info.main_input_carrier
            else:
                car_standby_power = self.options.standby_power_carrier

        if not b_tec.find_component("var_x"):
            b_tec.var_x = pyo.Var(
                self.set_t_full, domain=pyo.NonNegativeReals, bounds=(0, 1)
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
                        if car_input == self.info.main_input_carrier:
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

                def init_output_on(const):
                    return (
                        sum(
                            self.output[t, car_output]
                            for car_output in b_tec.set_output_carriers
                        )
                        == alpha1[ind - 1]
                        * sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        + alpha2[ind - 1] * b_tec.var_size * rated_power
                    )

                dis.const_input_output_on = pyo.Constraint(rule=init_output_on)

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
            self.set_t, s_indicators, rule=init_input_output
        )

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]

        b_tec.disjunction_input_output = gdp.Disjunction(
            self.set_t, rule=bind_disjunctions
        )

        return b_tec

    def _performance_function_type_4(self, b_tec):
        """
        Piece-wise linear, minimal partload, includes constraints for slow (>1h) startup and shutdown trajectories.

        Based on Equations 9-11, 13 and 15 in Morales-España, G., Ramírez-Elizondo, L., & Hobbs, B. F. (2017). Hidden
        power system inflexibilities imposed by traditional unit commitment formulations. Applied Energy, 191, 223–238.
        https://doi.org/10.1016/J.APENERGY.2017.01.089

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        # Transformation required
        self.big_m_transformation_required = 1

        # Performance Parameters
        rated_power = self.parameters.rated_power
        c_ti = self.coeff.time_independent
        dynamics = self.coeff.dynamics
        alpha1 = c_ti["fit"]["out"]["alpha1"]
        alpha2 = c_ti["fit"]["out"]["alpha2"]
        bp_x = c_ti["fit"]["out"]["bp_x"]
        min_part_load = c_ti["min_part_load"]
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

                def init_output_SU(const):
                    return (
                        sum(
                            self.output[t, car_output]
                            for car_output in b_tec.set_output_carriers
                        )
                        == alpha1[0]
                        * sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        + alpha2[0] * b_tec.var_size * rated_power
                    )

                dis.const_output_SU = pyo.Constraint(rule=init_output_SU)

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

                def init_input_SD(cons):
                    return (
                        sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        == b_tec.var_size * SD_trajectory[ind_SD - 1]
                    )

                dis.const_input_SD = pyo.Constraint(rule=init_input_SD)

                def init_output_SD(const):
                    return (
                        sum(
                            self.output[t, car_output]
                            for car_output in b_tec.set_output_carriers
                        )
                        == alpha1[0]
                        * sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        + alpha2[0] * b_tec.var_size * rated_power
                    )

                dis.const_output_SD = pyo.Constraint(rule=init_output_SD)

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

                def init_output_on(const):
                    return (
                        sum(
                            self.output[t, car_output]
                            for car_output in b_tec.set_output_carriers
                        )
                        == alpha1[ind_bpx - 1]
                        * sum(
                            self.input[t, car_input]
                            for car_input in b_tec.set_input_carriers
                        )
                        + alpha2[ind_bpx - 1] * b_tec.var_size * rated_power
                    )

                dis.const_input_output_on = pyo.Constraint(rule=init_output_on)

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

    def _define_ramping_rates(self, b_tec):
        """
        Constraints the inputs for a ramping rate

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        dynamics = self.coeff.dynamics

        ramping_time = dynamics["ramping_time"]

        # Calculate ramping rates
        if "ref_size" in dynamics and not dynamics["ref_size"] == -1:
            ramping_rate = dynamics["ref_size"] / ramping_time
        else:
            ramping_rate = b_tec.var_size / ramping_time

        # Constraints ramping rates
        if (
            not self.options.performance_function_type == 1
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
                self.set_t, s_indicators, rule=init_ramping_operation_on
            )

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_ramping_operation_on[t, i] for i in s_indicators]

            b_tec.disjunction_ramping_operation_on = gdp.Disjunction(
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
                    return pyo.Constraint.Skip

            b_tec.const_ramping_down_rate = pyo.Constraint(
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
                    return pyo.Constraint.Skip

            b_tec.const_ramping_up_rate = pyo.Constraint(
                self.set_t, rule=init_ramping_up_rate
            )

        return b_tec
