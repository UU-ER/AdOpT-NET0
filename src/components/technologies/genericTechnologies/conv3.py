import pyomo.environ as pyo
import pyomo.gdp as gdp
from warnings import warn
import pandas as pd

from ..genericTechnologies.utilities import fit_performance_generic_tecs
from ..technology import Technology


class Conv3(Technology):
    """
    This technology type resembles a technology for which the output can be written as a function of the input,
    according to different performance functions that can be specified in the JSON files (``performance_function_type``).
    Four different performance function fits of the technology data (again specified in the JSON file) are possible,
    and for all the function is based on the input of the main carrier , i.e.,:
     :math:`output_{car} = f_{car}(input_{maincarrier})`.
    Note that the ratio between all input carriers is fixed.

    **Constraint declarations:**

    For all technologies modelled with CONV3 (regardless of performance function type):
    - Size constraints are formulated on the input.

      .. math::
         Input_{t, maincarrier} \leq S

    - The ratios of inputs are fixed and given as:

      .. math::
        Input_{t, car} = {\\phi}_{car} * Input_{t, maincarrier}

    Type 1 is a linear performance function through the origin. However, a minimum part load can be specified,
    basically meaning that the part of the performance function from the origin to this minimum part load value
    cannot be met, thus it also cannot be turned off. So, for ``performance_function_type == 1`` the following
    constraint holds:

      .. math::
        Output_{t, car} = {\\alpha}_{1, car} Input_{t, maincarrier}

    Type 2 is a linear performance function with a minimum part load. In this case, the linear line does not have to
    be in line with the origin, and the technology can be turned off as well. Thus, the performance is either at the
    origin (off) or it is at a linear line. Therefore, a big-m transformation is required. So, for
    ``performance_function_type == 2``, the following constraints hold:


    - If the technology is in on, it holds:

      .. math::
        Output_{t, car} = {\\alpha}_{1, car} Input_{t, maincarrier} + {\\alpha}_{2, car}

      .. math::
        Input_{maincarrier} \geq Input_{min} * S

    - If the technology is off, input and output are set to 0:

      .. math::
         Output_{t, car} = 0

      .. math::
         Input_{t, maincarrier} = 0

    For ``performance_function_type == 3``, the performance is modelled as a piecewise linear function. Note that this
    requires a big-m transformation. For this case, the same constraints as for ``performance_function_type == 2`` hold,
    but for each "piece" (segment) of the performance function (as specified in the JSON file, ``nr_seg``), the alpha_1
    and alpha_2 change, so the performance function (output = f(input)) is written for each segment separately.

    For ``performance_function_type == 4``, the performance is also modelled as a piecewise linear function. However,
    this type additionally includes constraints for slow (>1h) startup and shutdown trajectories.
    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.fitted_performance = None
        self.main_car = self.performance_data["main_input_carrier"]

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits conversion technology type 3 and returns fitted parameters as a dict

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """

        if self.performance_data["size_based_on"] == "output":
            raise Exception("size_based_on == output for CONV3 not possible.")
        self.fitted_performance = fit_performance_generic_tecs(
            self.performance_data, time_steps=len(climate_data)
        )

        # Input bounds recalculation
        for car in self.fitted_performance.input_carrier:
            if not car == self.performance_data["main_input_carrier"]:
                self.fitted_performance.bounds["input"][car] = (
                    self.fitted_performance.bounds["input"][
                        self.performance_data["main_input_carrier"]
                    ]
                    * self.performance_data["input_ratios"][car]
                )

    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type CONV3

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(Conv3, self).construct_tech_model(
            b_tec, data, set_t_full, set_t_clustered
        )

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        rated_power = self.fitted_performance.rated_power
        performance_function_type = performance_data["performance_function_type"]
        standby_power = self.performance_data["standby_power"]
        phi = {}
        for car in self.performance_data["input_ratios"]:
            phi[car] = self.performance_data["input_ratios"][car]

        if performance_function_type == 1:
            b_tec = self._performance_function_type_1(b_tec)
        elif performance_function_type == 2:
            b_tec = self._performance_function_type_2(b_tec)
        elif performance_function_type == 3:
            b_tec = self._performance_function_type_3(b_tec)
        elif performance_function_type == 4:
            b_tec = self._performance_function_type_4(b_tec)

        # Size constraints
        # constraint on input ratios
        if standby_power == -1:

            def init_input_input(const, t, car_input):
                if car_input == self.main_car:
                    return pyo.Constraint.Skip
                else:
                    return (
                        self.input[t, car_input]
                        == phi[car_input] * self.input[t, self.main_car]
                    )

            b_tec.const_input_input = pyo.Constraint(
                self.set_t_full, b_tec.set_input_carriers, rule=init_input_input
            )
        else:
            s_indicators = range(0, 2)

            def init_input_input(dis, t, ind):
                if ind == 0:  # technology off
                    dis.const_x_off = pyo.Constraint(expr=b_tec.var_x[t] == 0)

                    def init_input_off(const, car_input):
                        if car_input == self.main_car:
                            return pyo.Constraint.Skip
                        else:
                            return self.input[t, car_input] == 0

                    dis.const_input_off = pyo.Constraint(
                        b_tec.set_input_carriers, rule=init_input_off
                    )

                else:  # technology on
                    dis.const_x_on = pyo.Constraint(expr=b_tec.var_x[t] == 1)

                    def init_input_on(const, car_input):
                        if car_input == self.main_car:
                            return pyo.Constraint.Skip
                        else:
                            return (
                                self.input[t, car_input]
                                == phi[car_input] * self.input[t, self.main_car]
                            )

                    dis.const_input_on = pyo.Constraint(
                        b_tec.set_input_carriers, rule=init_input_on
                    )

            b_tec.dis_input_input = gdp.Disjunct(
                self.set_t, s_indicators, rule=init_input_input
            )

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_input_input[t, i] for i in s_indicators]

            b_tec.disjunction_input_input = gdp.Disjunction(
                self.set_t, rule=bind_disjunctions
            )

        # size constraint based on main carrier input
        def init_size_constraint(const, t):
            return self.input[t, self.main_car] <= b_tec.var_size * rated_power

        b_tec.const_size = pyo.Constraint(self.set_t, rule=init_size_constraint)

        # RAMPING RATES
        if "ramping_time" in self.performance_data:
            if not self.performance_data["ramping_time"] == -1:
                b_tec = self._define_ramping_rates(b_tec)

        return b_tec

    def _performance_function_type_1(self, b_tec):
        """
        Sets the input-output constraint for a tec based on tec_type CONV3 with performance type 1.

        Type 1 is a linear performance function through the origin. However, a minimum part load can be specified,
        basically meaning that the part of the performance function from the origin to this minimum part load value
        cannot be met, thus it also cannot be turned off.

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """

        # Performance parameters:
        alpha1 = {}
        for car in self.performance_data["performance"]["out"]:
            alpha1[car] = self.fitted_performance.coefficients[car]["alpha1"]
        rated_power = self.fitted_performance.rated_power
        min_part_load = self.performance_data["min_part_load"]

        # Input-output relation
        def init_input_output(const, t, car_output):
            return (
                self.output[t, car_output]
                == alpha1[car_output] * self.input[t, self.main_car]
            )

        b_tec.const_input_output = pyo.Constraint(
            self.set_t, b_tec.set_output_carriers, rule=init_input_output
        )

        # setting the minimum part load constraint if applicable
        if min_part_load > 0:

            def init_min_part_load(const, t):
                return (
                    min_part_load * b_tec.var_size * rated_power
                    <= self.input[t, self.main_car]
                )

            b_tec.const_min_part_load = pyo.Constraint(
                self.set_t, rule=init_min_part_load
            )

        return b_tec

    def _performance_function_type_2(self, b_tec):
        """
        Sets the input-output constraint for a tec based on tec_type CONV3 with performance type 2.

        Type 2 is a linear performance function with a minimum part load. In this case, the linear line does not have to
        be in line with the origin, and the technology can be turned off as well. Thus, the performance is either at the
        origin (off) or it is at a linear line. Therefore, a big-m transformation is required.

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """

        # Transformation required
        self.big_m_transformation_required = 1

        # Performance Parameters
        alpha1 = {}
        alpha2 = {}
        for car in self.performance_data["performance"]["out"]:
            alpha1[car] = self.fitted_performance.coefficients[car]["alpha1"]
            alpha2[car] = self.fitted_performance.coefficients[car]["alpha2"]
        rated_power = self.fitted_performance.rated_power
        min_part_load = self.performance_data["min_part_load"]
        standby_power = self.performance_data["standby_power"]

        if standby_power != -1:
            if (
                "standby_power_carrier" not in self.performance_data
            ) or self.performance_data["standby_power_carrier"] == -1:
                car_standby_power = self.main_car
            else:
                car_standby_power = self.performance_data["standby_power_carrier"]

        if not b_tec.find_component("var_x"):
            b_tec.var_x = pyo.Var(
                self.set_t_full, domain=pyo.NonNegativeReals, bounds=(0, 1)
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
                        if car_input == self.main_car:
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
                        == alpha1[car_output] * self.input[t, self.main_car]
                        + alpha2[car_output] * b_tec.var_size * rated_power
                    )

                dis.const_input_output_on = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_input_output_on
                )

                # min part load constraint
                def init_min_partload(const):
                    return (
                        self.input[t, self.main_car]
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
        Sets the input-output constraint for a tec based on tec_type CONV3 with performance type 3.

        Type 3 is a piecewise linear fit to the performance data, based on the number of segments specified. Note that
        this requires a big-m transformation. Again, a minimum part load is possible.

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        # Transformation required
        self.big_m_transformation_required = 1

        # Performance Parameters
        alpha1 = {}
        alpha2 = {}
        for car in self.performance_data["performance"]["out"]:
            bp_x = self.fitted_performance.coefficients[car]["bp_x"]
            alpha1[car] = self.fitted_performance.coefficients[car]["alpha1"]
            alpha2[car] = self.fitted_performance.coefficients[car]["alpha2"]
        rated_power = self.fitted_performance.rated_power
        min_part_load = self.performance_data["min_part_load"]
        standby_power = self.performance_data["standby_power"]

        if standby_power != -1:
            if (
                "standby_power_carrier" not in self.performance_data
            ) or self.performance_data["standby_power_carrier"] == -1:
                car_standby_power = self.main_car
            else:
                car_standby_power = self.performance_data["standby_power_carrier"]

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
                        if car_input == self.main_car:
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
                        self.input[t, self.main_car]
                        >= bp_x[ind - 1] * b_tec.var_size * rated_power
                    )

                dis.const_input_on1 = pyo.Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return (
                        self.input[t, self.main_car]
                        <= bp_x[ind] * b_tec.var_size * rated_power
                    )

                dis.const_input_on2 = pyo.Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output][ind - 1] * self.input[t, self.main_car]
                        + alpha2[car_output][ind - 1] * b_tec.var_size * rated_power
                    )

                dis.const_input_output_on = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_on
                )

                # min part load constraint
                def init_min_partload(const):
                    return (
                        self.input[t, self.main_car]
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
        Sets the constraints (input-output and startup/shutdown) for a tec based on tec_type CONV3 with performance
        type 4.

        Type 4 is also a piecewise linear fit to the performance data, based on the number of segments specified. Note
        that this requires a big-m transformation. Again, a minimum part load is possible. Additionally, type 4 includes
        constraints for slow (>1h) startup and shutdown trajectories.

        Based on Equations 9-11, 13 and 15 in Morales-España, G., Ramírez-Elizondo, L., & Hobbs, B. F. (2017). Hidden
        power system inflexibilities imposed by traditional unit commitment formulations. Applied Energy, 191, 223–238.
        https://doi.org/10.1016/J.APENERGY.2017.01.089

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """

        # Transformation required
        self.big_m_transformation_required = 1

        # Performance Parameters
        SU_time = self.performance_data["SU_time"]
        SD_time = self.performance_data["SD_time"]
        alpha1 = {}
        alpha2 = {}
        for car in self.performance_data["performance"]["out"]:
            bp_x = self.fitted_performance.coefficients[car]["bp_x"]
            alpha1[car] = self.fitted_performance.coefficients[car]["alpha1"]
            alpha2[car] = self.fitted_performance.coefficients[car]["alpha2"]
        rated_power = self.fitted_performance.rated_power
        min_part_load = self.performance_data["min_part_load"]

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
                        self.input[t, self.main_car]
                        == b_tec.var_size * SU_trajectory[ind - 1]
                    )

                dis.const_input_SU = pyo.Constraint(rule=init_input_SU)

                def init_output_SU(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output][0] * self.input[t, self.main_car]
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

                def init_input_SD(cons):
                    return (
                        self.input[t, self.main_car]
                        == b_tec.var_size * SD_trajectory[ind_SD - 1]
                    )

                dis.const_input_SD = pyo.Constraint(rule=init_input_SD)

                def init_output_SD(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output][0] * self.input[t, self.main_car]
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
                        self.input[t, self.main_car]
                        >= bp_x[ind_bpx - 1] * b_tec.var_size * rated_power
                    )

                dis.const_input_on1 = pyo.Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return (
                        self.input[t, self.main_car]
                        <= bp_x[ind_bpx] * b_tec.var_size * rated_power
                    )

                dis.const_input_on2 = pyo.Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output][ind_bpx - 1]
                        * self.input[t, self.main_car]
                        + alpha2[car_output][ind_bpx - 1] * b_tec.var_size * rated_power
                    )

                dis.const_input_output_on = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_on
                )

                # min part load relation
                def init_min_partload(const):
                    return (
                        self.input[t, self.main_car]
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
            not self.performance_data["performance_function_type"] == 1
            and "ramping_const_int" in self.performance_data
            and self.performance_data["ramping_const_int"] == 1
        ):

            s_indicators = range(0, 3)

            def init_ramping_operation_on(dis, t, ind):
                if t > 1:
                    if ind == 0:  # ramping constrained
                        dis.const_ramping_on = pyo.Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 0
                        )

                        def init_ramping_down_rate_operation(const):
                            return (
                                -ramping_rate
                                <= self.input[t, self.main_car]
                                - self.input[t - 1, self.main_car]
                            )

                        dis.const_ramping_down_rate = pyo.Constraint(
                            rule=init_ramping_down_rate_operation
                        )

                        def init_ramping_up_rate_operation(const):
                            return (
                                self.input[t, self.main_car]
                                - self.input[t - 1, self.main_car]
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
                    return (
                        -ramping_rate
                        <= self.input[t, self.main_car]
                        - self.input[t - 1, self.main_car]
                    )
                else:
                    return pyo.Constraint.Skip

            b_tec.const_ramping_down_rate = pyo.Constraint(
                self.set_t, rule=init_ramping_down_rate
            )

            def init_ramping_up_rate(const, t):
                if t > 1:
                    return (
                        self.input[t, self.main_car] - self.input[t - 1, self.main_car]
                        <= ramping_rate
                    )
                else:
                    return pyo.Constraint.Skip

            b_tec.const_ramping_up_rate = pyo.Constraint(
                self.set_t, rule=init_ramping_up_rate
            )

        return b_tec
