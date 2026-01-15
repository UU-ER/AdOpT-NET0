"""
Technology with no input/output substitution

This technology type resembles a technology for which the output can be written
as a function of the input, according to different performance functions that
can be specified in the JSON files (``performance_function_type``).
Four different performance function fits of the technology data (again specified
in the JSON file) are possible, and for all the function is based on the input
of the main carrier , i.e.,: :math:`output_{car} = f_{car}(input_{maincarrier})`.
Note that the ratio between all input carriers is fixed.

**Constraint declarations:**

For all technologies modelled with CONV3 (regardless of performance function type):
- Size constraints are formulated on the input.

  .. math::
     Input_{t, maincarrier} \\leq S

- The ratios of inputs are fixed and given as:

  .. math::
    Input_{t, car} = {\\phi}_{car} * Input_{t, maincarrier}

  If the technology is turned off, all inputs are set to zero.

- ``performance_function_type == 1``: Linear through origin. Note that if
  min_part_load is larger 0, the technology cannot be turned off.

  .. math::
    Output_{t, car} = {\\alpha}_{1, car} Input_{t, maincarrier}

  .. math::
    min_part_load * S \\leq {\\alpha}_1 Input_{t, maincarrier}

- ``performance_function_type == 2``: Linear with minimal partload. If the
  technology is in on, it holds:

  If the technology is in on, it holds:

  .. math::
    Output_{t, car} = {\\alpha}_{1, car} Input_{t, maincarrier} + {\\alpha}_{2, car}

  .. math::
    Input_{maincarrier} \\geq Input_{min} * S

- If the technology is off, input and output are set to 0:

  .. math::
     Output_{t, car} = 0

  .. math::
     Input_{t, maincarrier} = 0

- ``performance_function_type == 3``: Piecewise linear performance function (
  makes big-m transformation required). The same constraints as for
  ``performance_function_type == 2`` with the exception that the performance
  function is defined piecewise for the respective number of pieces.

- ``performance_function_type == 4``:Piece-wise linear, minimal partload. Enables the modeling
  of technologies with slow (>1h) startup and shutdown trajectories. For more information
  please refer to dynamics under advanced topics. Based on Equations 9-11, 13 and 15 in Morales-España, G., Ramírez-Elizondo, L.,
  & Hobbs, B. F. (2017). Hidden power system inflexibilities imposed by
  traditional unit commitment formulations. Applied Energy, 191, 223–238.
  https://doi.org/10.1016/J.APENERGY.2017.01.089

- Additionally, ramping rates of the technology can be constrained.

  .. math::
     -rampingrate \\leq Input_{t, main-car} - Input_{t-1, car} \\leq rampingrate

"""

import pyomo.environ as pyo
import pyomo.gdp as gdp
from warnings import warn

from adopt_net0.core.components.technologies.utilities import FitGenericTecTypeType1, FitGenericTecTypeType2, \
    FitGenericTecTypeType34
from adopt_net0.core.components.technology import Technology
from adopt_net0.core.components.utilities import link_full_resolution_to_clustered, get_attribute_from_dict


class Conv3(Technology):

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.emissions_based_on = "input"
        self.size_based_on = "input"

        self.standby_power_carrier = get_attribute_from_dict(
            tec_data["Performance"], "standby_power_carrier", -1
        )

        self.main_input_carrier = tec_data["Performance"]["main_input_carrier"]

        # Initialize fitting class
        if self.performance_function_type == 1:
            self.fitting_class = FitGenericTecTypeType1(
                self.input_carrier, self.output_carrier
            )
        elif self.performance_function_type == 2:
            self.fitting_class = FitGenericTecTypeType2(
                self.input_carrier, self.output_carrier
            )
        elif self.performance_function_type == 3 or self.performance_function_type == 4:
            self.fitting_class = FitGenericTecTypeType34(
                self.input_carrier, self.output_carrier
            )
        else:
            raise Exception(
                "performance_function_type must be an integer between 1 and 4"
            )

    def fit_performance(self, modelhub: object, component_id: tuple, **kwargs):
        """
        Fits technology performance and writes it to self.

        :param modelhub: model hub
        :param tuple component_id: component id containing (period, node, component name)
        """
        super(Conv3, self).fit_performance(modelhub, component_id)

        if self.size_based_on == "output":
            raise Exception("size_based_on == output for CONV3 not possible.")

        # fit coefficients
        self.processed_coeff.time_independent["fit"] = (
            self.fitting_class.fit_performance_function(
                self.performance_data["performance"]
            )
        )

        phi = {}
        for car in self.performance_data["input_ratios"]:
            phi[car] = self.performance_data["input_ratios"][car]
        self.processed_coeff.time_independent["phi"] = phi

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(Conv3, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        self.bounds["input"] = self.fitting_class.calculate_input_bounds(
            self.size_based_on, time_steps
        )
        self.bounds["output"] = self.fitting_class.calculate_output_bounds(
            self.emissions_based_on, time_steps
        )

        # Input bounds recalculation
        for car in self.input_carrier:
            if not car == self.main_input_carrier:
                self.bounds["input"][car] = (
                    self.bounds["input"][self.main_input_carrier]
                    * self.performance_data["input_ratios"][car]
                )

    def construct_model(self, b_tec, modelhub, set_t_full, set_t_clustered, **kwargs):
        """
        Adds constraints to technology blocks for tec_type CONV3

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(Conv3, self).construct_model(
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
        elif self.performance_function_type == 3:
            self._performance_function_type_3(b_tec)
        elif self.performance_function_type == 4:
            self._performance_function_type_4(b_tec)

        # Size constraints
        # constraint on input ratios
        standby_power = coeff_ti["standby_power"]
        phi = coeff_ti["phi"]

        if self.performance_function_type == 1 or standby_power == -1:

            def init_input_input(const, t, car_input):
                if car_input == self.main_input_carrier:
                    return pyo.Constraint.Skip
                else:
                    return (
                        self.input[t, car_input]
                        == phi[car_input] * self.input[t, self.main_input_carrier]
                    )

            b_tec.const_input_input = pyo.Constraint(
                self.set_t_performance, b_tec.set_input_carriers, rule=init_input_input
            )
        else:

            self.big_m_transformation_required = 1

            if self.standby_power_carrier == -1:
                car_standby_power = self.main_input_carrier
            else:
                car_standby_power = self.standby_power_carrier

            s_indicators = range(0, 2)

            def init_input_input(dis, t, ind):
                if ind == 0:  # technology off
                    dis.const_x_off = pyo.Constraint(expr=b_tec.var_on_off[t] == 0)

                    def init_input_off(const, car_input):
                        if car_input == car_standby_power:
                            return pyo.Constraint.Skip
                        else:
                            return self.input[t, car_input] == 0

                    dis.const_input_off = pyo.Constraint(
                        b_tec.set_input_carriers, rule=init_input_off
                    )

                else:  # technology on
                    dis.const_x_on = pyo.Constraint(expr=b_tec.var_on_off[t] == 1)

                    def init_input_on(const, car_input):
                        if car_input == self.main_input_carrier:
                            return pyo.Constraint.Skip
                        else:
                            return (
                                self.input[t, car_input]
                                == phi[car_input]
                                * self.input[t, self.main_input_carrier]
                            )

                    dis.const_input_on = pyo.Constraint(
                        b_tec.set_input_carriers, rule=init_input_on
                    )

            b_tec.dis_input_input = gdp.Disjunct(
                self.set_t_performance, s_indicators, rule=init_input_input
            )

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_input_input[t, i] for i in s_indicators]

            b_tec.disjunction_input_input = gdp.Disjunction(
                self.set_t_performance, rule=bind_disjunctions
            )

        # size constraint based on main carrier input
        def init_size_constraint(const, t):
            return (
                self.input[t, self.main_input_carrier]
                <= b_tec.var_size * rated_capacity
            )

        b_tec.const_size = pyo.Constraint(
            self.set_t_performance, rule=init_size_constraint
        )

    def _performance_function_type_1(self, b_tec):
        """
        Linear, through origin, min partload possible

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """

        # Performance parameters:
        coeff_ti = self.processed_coeff.time_independent
        rated_capacity = coeff_ti["rated_capacity"]

        alpha1 = {}
        for car in coeff_ti["fit"]:
            alpha1[car] = coeff_ti["fit"][car]["alpha1"]
        min_part_load = coeff_ti["min_part_load"]

        # Input-output relation
        def init_input_output(const, t, car_output):
            return (
                self.output[t, car_output]
                == alpha1[car_output] * self.input[t, self.main_input_carrier]
            )

        b_tec.const_input_output = pyo.Constraint(
            self.set_t_performance, b_tec.set_output_carriers, rule=init_input_output
        )

        # setting the minimum part load constraint if applicable
        if min_part_load > 0:

            def init_min_part_load(const, t):
                return (
                    min_part_load * b_tec.var_size * rated_capacity
                    <= self.input[t, self.main_input_carrier]
                )

            b_tec.const_min_part_load = pyo.Constraint(
                self.set_t_performance, rule=init_min_part_load
            )



    def _performance_function_type_2(self, b_tec):
        """
        Linear, minimal partload

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """

        # Transformation required
        self.big_m_transformation_required = 1

        # Performance Parameters
        coeff_ti = self.processed_coeff.time_independent
        rated_capacity = coeff_ti["rated_capacity"]

        alpha1 = {}
        alpha2 = {}
        for car in coeff_ti["fit"]:
            alpha1[car] = coeff_ti["fit"][car]["alpha1"]
            alpha2[car] = coeff_ti["fit"][car]["alpha2"]
        min_part_load = coeff_ti["min_part_load"]
        standby_power = coeff_ti["standby_power"]

        # Performance Parameters

        if standby_power != -1:
            if self.standby_power_carrier == -1:
                car_standby_power = self.main_input_carrier
            else:
                car_standby_power = self.standby_power_carrier

        if not b_tec.find_component("var_on_off"):
            b_tec.var_on_off = pyo.Var(
                self.set_t_performance, domain=pyo.Binary
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

                dis.const_x_off = pyo.Constraint(expr=b_tec.var_on_off[t] == 0)

                if standby_power == -1:

                    def init_input_off(const, car_input):
                        return self.input[t, car_input] == 0

                    dis.const_input = pyo.Constraint(
                        b_tec.set_input_carriers, rule=init_input_off
                    )

                else:

                    def init_standby_power(const, car_input):
                        if car_input == self.main_input_carrier:
                            return (
                                self.input[t, car_standby_power]
                                == standby_power * b_tec.var_size * rated_capacity
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

                dis.const_x_on = pyo.Constraint(expr=b_tec.var_on_off[t] == 1)

                # input-output relation
                def init_input_output_on(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output] * self.input[t, self.main_input_carrier]
                        + alpha2[car_output] * b_tec.var_size * rated_capacity
                    )

                dis.const_input_output_on = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_input_output_on
                )

                # min part load constraint
                def init_min_partload(const):
                    return (
                        self.input[t, self.main_input_carrier]
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
        Sets the input-output constraint for a tec based on tec_type CONV3 with performance type 3.

        Type 3 is a piecewise linear fit to the performance data, based on the number of segments specified. Note that
        this requires a big-m transformation. Again, a minimum part load is possible.

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        # Transformation required
        self.big_m_transformation_required = 1

        # Performance Parameters
        coeff_ti = self.processed_coeff.time_independent
        rated_capacity = coeff_ti["rated_capacity"]

        alpha1 = {}
        alpha2 = {}
        for car in coeff_ti["fit"]:
            bp_x = coeff_ti["fit"][car]["bp_x"]
            alpha1[car] = coeff_ti["fit"][car]["alpha1"]
            alpha2[car] = coeff_ti["fit"][car]["alpha2"]
        min_part_load = coeff_ti["min_part_load"]
        standby_power = coeff_ti["standby_power"]

        if standby_power != -1:
            if self.standby_power_carrier == -1:
                car_standby_power = self.main_input_carrier
            else:
                car_standby_power = self.standby_power_carrier

        if not b_tec.find_component("var_on_off"):
            b_tec.var_on_off = pyo.Var(
                self.set_t_performance, domain=pyo.Binary
            )

        s_indicators = range(0, len(bp_x))

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off

                dis.const_x_off = pyo.Constraint(expr=b_tec.var_on_off[t] == 0)

                if standby_power == -1:

                    def init_input_off(const, car_input):
                        return self.input[t, car_input] == 0

                    dis.const_input_off = pyo.Constraint(
                        b_tec.set_input_carriers, rule=init_input_off
                    )

                else:

                    def init_standby_power(const, car_input):
                        if car_input == self.main_input_carrier:
                            return (
                                self.input[t, car_standby_power]
                                == standby_power * b_tec.var_size * rated_capacity
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

                dis.const_x_on = pyo.Constraint(expr=b_tec.var_on_off[t] == 1)

                def init_input_on1(const):
                    return (
                        self.input[t, self.main_input_carrier]
                        >= bp_x[ind - 1] * b_tec.var_size * rated_capacity
                    )

                dis.const_input_on1 = pyo.Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return (
                        self.input[t, self.main_input_carrier]
                        <= bp_x[ind] * b_tec.var_size * rated_capacity
                    )

                dis.const_input_on2 = pyo.Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output][ind - 1]
                        * self.input[t, self.main_input_carrier]
                        + alpha2[car_output][ind - 1] * b_tec.var_size * rated_capacity
                    )

                dis.const_input_output_on = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_on
                )

                # min part load constraint
                def init_min_partload(const):
                    return (
                        self.input[t, self.main_input_carrier]
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
        coeff_ti = self.processed_coeff.time_independent
        rated_capacity = coeff_ti["rated_capacity"]
        dynamics = self.processed_coeff.dynamics
        alpha1 = {}
        alpha2 = {}
        for car in coeff_ti["fit"]:
            bp_x = coeff_ti["fit"][car]["bp_x"]
            alpha1[car] = coeff_ti["fit"][car]["alpha1"]
            alpha2[car] = coeff_ti["fit"][car]["alpha2"]
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
                dis.const_x_off = pyo.Constraint(expr=b_tec.var_on_off[t] == 0)

                def init_y_off(const, i):
                    if t < len(self.set_t_full) - SU_time or i > SU_time - (
                        len(self.set_t_full) - t
                    ):
                        return b_tec.var_in_startup[t - i + SU_time + 1] == 0
                    else:
                        return (
                            b_tec.var_in_startup[(t - i + SU_time + 1) - len(self.set_t_full)]
                            == 0
                        )

                dis.const_y_off = pyo.Constraint(range(1, SU_time + 1), rule=init_y_off)

                def init_z_off(const, j):
                    if j <= t:
                        return b_tec.var_in_shutdown[t - j + 1] == 0
                    else:
                        return b_tec.var_in_shutdown[len(self.set_t_full) + (t - j + 1)] == 0

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
                dis.const_x_off = pyo.Constraint(expr=b_tec.var_on_off[t] == 0)

                def init_y_on(const):
                    if t < len(self.set_t_full) - SU_time or ind > SU_time - (
                        len(self.set_t_full) - t
                    ):
                        return b_tec.var_in_startup[t - ind + SU_time + 1] == 1
                    else:
                        return (
                            b_tec.var_in_startup[(t - ind + SU_time + 1) - len(self.set_t_full)]
                            == 1
                        )

                dis.const_y_on = pyo.Constraint(rule=init_y_on)

                def init_z_off(const):
                    if t < len(self.set_t_full) - SU_time or ind > SU_time - (
                        len(self.set_t_full) - t
                    ):
                        return b_tec.var_in_shutdown[t - ind + SU_time + 1] == 0
                    else:
                        return (
                            b_tec.var_in_shutdown[(t - ind + SU_time + 1) - len(self.set_t_full)]
                            == 0
                        )

                dis.const_z_off = pyo.Constraint(rule=init_z_off)

                def init_input_SU(const):
                    return (
                        self.input[t, self.main_input_carrier]
                        == b_tec.var_size * SU_trajectory[ind - 1]
                    )

                dis.const_input_SU = pyo.Constraint(rule=init_input_SU)

                def init_output_SU(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output][0]
                        * self.input[t, self.main_input_carrier]
                        + alpha2[car_output][0] * b_tec.var_size * rated_capacity
                    )

                dis.const_output_SU = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_SU
                )

            elif ind in range(
                SU_time + 1, SU_time + SD_time + 1
            ):  # technology in shutdown
                ind_SD = ind - SU_time
                dis.const_x_off = pyo.Constraint(expr=b_tec.var_on_off[t] == 0)

                def init_z_on(const):
                    if ind_SD <= t:
                        return b_tec.var_in_shutdown[t - ind_SD + 1] == 1
                    else:
                        return b_tec.var_in_shutdown[len(self.set_t_full) + (t - ind_SD + 1)] == 1

                dis.const_z_on = pyo.Constraint(rule=init_z_on)

                def init_y_off(const):
                    if ind_SD <= t:
                        return b_tec.var_in_startup[t - ind_SD + 1] == 0
                    else:
                        return b_tec.var_in_startup[len(self.set_t_full) + (t - ind_SD + 1)] == 0

                dis.const_y_off = pyo.Constraint(rule=init_y_off)

                def init_input_SD(const):
                    return (
                        self.input[t, self.main_input_carrier]
                        == b_tec.var_size * SD_trajectory[ind_SD - 1]
                    )

                dis.const_input_SD = pyo.Constraint(rule=init_input_SD)

                def init_output_SD(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output][0]
                        * self.input[t, self.main_input_carrier]
                        + alpha2[car_output][0] * b_tec.var_size * rated_capacity
                    )

                dis.const_output_SD = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_SD
                )

            elif ind > SU_time + SD_time:
                ind_bpx = ind - (SU_time + SD_time)
                dis.const_x_on = pyo.Constraint(expr=b_tec.var_on_off[t] == 1)

                def init_input_on1(const):
                    return (
                        self.input[t, self.main_input_carrier]
                        >= bp_x[ind_bpx - 1] * b_tec.var_size * rated_capacity
                    )

                dis.const_input_on1 = pyo.Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return (
                        self.input[t, self.main_input_carrier]
                        <= bp_x[ind_bpx] * b_tec.var_size * rated_capacity
                    )

                dis.const_input_on2 = pyo.Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return (
                        self.output[t, car_output]
                        == alpha1[car_output][ind_bpx - 1]
                        * self.input[t, self.main_input_carrier]
                        + alpha2[car_output][ind_bpx - 1]
                        * b_tec.var_size
                        * rated_capacity
                    )

                dis.const_input_output_on = pyo.Constraint(
                    b_tec.set_output_carriers, rule=init_output_on
                )

                # min part load relation
                def init_min_partload(const):
                    return (
                        self.input[t, self.main_input_carrier]
                        >= min_part_load * b_tec.var_size * rated_capacity
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
