from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn

from ..genericTechnologies.utilities import fit_performance_generic_tecs
from ..technology import Technology


class Conv3(Technology):
    r"""
    This technology type resembles a technology with different performance functions for the respective output
    carriers. The performance function is based on the input of the main carrier , i.e.
    :math:`output_{car} = f_{car}(input_{maincarrier})`.
    The ratio between all input carriers is fixed.
    Three different performance function fits are possible.

    **Constraint declarations:**

    - Size constraints are formulated on the input.

      .. math::
         Input_{t, maincarrier} \leq S

    - The ratios of inputs for all performance function types are fixed and given as:

      .. math::
        Input_{t, car} = {\\phi}_{car} * Input_{t, maincarrier}

    - ``performance_function_type == 1``: Linear through origin, i.e.:

      .. math::
        Output_{t, car} = {\\alpha}_{1, car} Input_{t, maincarrier}

    - ``performance_function_type == 2``: Linear with minimal partload (makes big-m transformation required). If the
      technology is in on, it holds:

      .. math::
        Output_{t, car} = {\\alpha}_{1, car} Input_{t, maincarrier} + {\\alpha}_{2, car}

      .. math::
        Input_{maincarrier} \geq Input_{min} * S

      If the technology is off, input and output is set to 0:

      .. math::
         Output_{t, car} = 0

      .. math::
         Input_{t, maincarrier} = 0

    - ``performance_function_type == 3``: Piecewise linear performance function (makes big-m transformation required).
      The same constraints as for ``performance_function_type == 2`` with the exception that the performance function
      is defined piecewise for the respective number of pieces

    """

    def __init__(self, tec_data):
        super().__init__(tec_data)

        self.fitted_performance = None
        self.main_car = self.performance_data['main_input_carrier']

    def fit_technology_performance(self, node_data):
        """
        Fits conversion technology type 3 and returns fitted parameters as a dict

        :param performance_data: contains X and y data of technology performance
        :param performance_function_type: options for type of performance function (linear, piecewise,...)
        :param nr_seg: number of segments on piecewise defined function
        """

        climate_data = node_data.data['climate_data']

        if self.performance_data['size_based_on'] == 'output':
            raise Exception('size_based_on == output for CONV3 not possible.')
        self.fitted_performance = fit_performance_generic_tecs(self.performance_data, time_steps=len(climate_data))

        # Input bounds recalculation
        for car in self.fitted_performance.input_carrier:
            if not car == self.performance_data['main_input_carrier']:
                self.fitted_performance.bounds['input'][car] = self.fitted_performance.bounds['input'][self.performance_data['main_input_carrier']] \
                                               * self.performance_data['input_ratios'][car]

    def construct_tech_model(self, b_tec, energyhub):
        """
        Adds constraints to technology blocks for tec_type CONV3

        :param obj b_tec: technology block
        :param Energyhub energyhub: energyhub instance
        :return: technology block
        """
        super(Conv3, self).construct_tech_model(b_tec, energyhub)

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        rated_power = self.fitted_performance.rated_power
        performance_function_type = performance_data['performance_function_type']
        standby_power = self.performance_data['standby_power']
        phi = {}
        for car in self.performance_data['input_ratios']:
            phi[car] = self.performance_data['input_ratios'][car]

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
                    return Constraint.Skip
                else:
                    return self.input[t, car_input] == phi[car_input] * self.input[t, self.main_car]
            b_tec.const_input_input = Constraint(self.set_t_full, b_tec.set_input_carriers, rule=init_input_input)
        else:
            s_indicators = range(0, 2)

            def init_input_input(dis, t, ind):
                if ind == 0:  # technology off
                    dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                    def init_input_off(const, car_input):
                        if car_input == self.main_car:
                            return Constraint.Skip
                        else:
                            return self.input[t, car_input] == 0
                    dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                else:  # technology on
                    dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

                    def init_input_on(const, car_input):
                        if car_input == self.main_car:
                            return Constraint.Skip
                        else:
                            return self.input[t, car_input] == phi[car_input] * self.input[t, self.main_car]

                    dis.const_input_on = Constraint(b_tec.set_input_carriers, rule=init_input_on)

            b_tec.dis_input_input = Disjunct(self.set_t, s_indicators, rule=init_input_input)

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_input_input[t, i] for i in s_indicators]

            b_tec.disjunction_input_input = Disjunction(self.set_t, rule=bind_disjunctions)

        # size constraint based main carrier input
        def init_size_constraint(const, t):
            return self.input[t, self.main_car] <= b_tec.var_size * rated_power

        b_tec.const_size = Constraint(self.set_t, rule=init_size_constraint)

        # RAMPING RATES
        if "ramping_rate" in self.performance_data:
            if not self.performance_data['ramping_rate']   == -1:
                b_tec = self._define_ramping_rates(b_tec)

        return b_tec

    def _performance_function_type_1(self, b_tec):
        """
        Linear, no minimal partload, through origin
        :param b_tec: technology block
        :return: technology block
        """
        # Performance parameter:
        alpha1 = {}
        for car in self.performance_data['performance']['out']:
            alpha1[car] = self.fitted_performance.coefficients[car]['alpha1']
        rated_power = self.fitted_performance.rated_power
        min_part_load = self.performance_data['min_part_load']

        # Input-output correlation
        def init_input_output(const, t, car_output):
            return self.output[t, car_output] == \
                   alpha1[car_output] * self.input[t, self.main_car]
        b_tec.const_input_output = Constraint(self.set_t, b_tec.set_output_carriers,
                                              rule=init_input_output)

        if min_part_load > 0:
            def init_min_part_load(const, t):
                return min_part_load * b_tec.var_size * rated_power <= self.input[t, self.main_car]
            b_tec.const_min_part_load = Constraint(self.set_t, rule=init_min_part_load)

        return b_tec

    def _performance_function_type_2(self, b_tec):
        """
        Linear, minimal partload
        :param b_tec: technology block
        :return: technology block
        """
        # Transformation required
        self.big_m_transformation_required = 1

        # Performance Parameters
        alpha1 = {}
        alpha2 = {}
        for car in self.performance_data['performance']['out']:
            alpha1[car] = self.fitted_performance.coefficients[car]['alpha1']
            alpha2[car] = self.fitted_performance.coefficients[car]['alpha2']
        rated_power = self.fitted_performance.rated_power
        min_part_load = self.performance_data['min_part_load']
        standby_power = self.performance_data['standby_power']

        if not b_tec.find_component('var_x'):
            b_tec.var_x = Var(self.set_t_full, domain=NonNegativeReals, bounds=(0, 1))

        if min_part_load == 0:
            warn('Having performance_function_type = 2 with no part-load usually makes no sense. Error occured for ' + self.name)

        # define disjuncts
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off

                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                if standby_power == -1:

                    def init_input_off(const, car_input):
                        return self.input[t, car_input] == 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                else:
                    def init_standby_power(const, car_input):
                        if car_input == self.main_car:
                            return self.input[t, self.main_car] == standby_power * b_tec.var_size * rated_power
                        else:
                            return self.input[t, car_input] == 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_standby_power)

                def init_output_off(const, car_output):
                    return self.output[t, car_output] == 0

                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # technology on

                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 1)

                # input-output relation
                def init_input_output_on(const, car_output):
                    return self.output[t, car_output] == \
                           alpha1[car_output] * self.input[t, self.main_car] + \
                           alpha2[car_output] * b_tec.var_size * rated_power

                dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_input_output_on)

                # min part load relation
                def init_min_partload(const):
                    return self.input[t, self.main_car] >= min_part_load * b_tec.var_size * rated_power

                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(self.set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]

        b_tec.disjunction_input_output = Disjunction(self.set_t, rule=bind_disjunctions)

        return b_tec

    def _performance_function_type_3(self, b_tec):
        """
        Piece-wise linear, minimal partload
        :param b_tec: technology block
        :return: technology block
        """
        # Transformation required
        self.big_m_transformation_required = 1

        # Performance Parameters
        alpha1 = {}
        alpha2 = {}
        for car in self.performance_data['performance']['out']:
            bp_x = self.fitted_performance.coefficients[car]['bp_x']
            alpha1[car] = self.fitted_performance.coefficients[car]['alpha1']
            alpha2[car] = self.fitted_performance.coefficients[car]['alpha2']
        rated_power = self.fitted_performance.rated_power
        min_part_load = self.performance_data['min_part_load']
        standby_power = self.performance_data['standby_power']

        if not b_tec.find_component('var_x'):
            b_tec.var_x = Var(self.set_t_full, domain=NonNegativeReals, bounds=(0, 1))

        s_indicators = range(0, len(bp_x))

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off

                if standby_power == -1:
                    def init_input_off(const, car_input):
                        return self.input[t, car_input] == 0
                    dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                else:
                    def init_standby_power(const, car_input):
                        if car_input == self.main_car:
                            return self.input[t, self.main_car] == standby_power * b_tec.var_size * rated_power
                        else:
                            return self.input[t, car_input] == 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_standby_power)

                def init_output_off(const, car_output):
                    return self.output[t, car_output] == 0

                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # piecewise definition
                def init_input_on1(const):
                    return self.input[t, self.main_car] >= bp_x[ind - 1] * b_tec.var_size * rated_power

                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return self.input[t, self.main_car] <= bp_x[ind] * b_tec.var_size * rated_power

                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return self.output[t, car_output] == \
                           alpha1[car_output][ind - 1] * self.input[t, self.main_car] + \
                           alpha2[car_output][ind - 1] * b_tec.var_size * rated_power

                dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_output_on)

                # min part load relation
                def init_min_partload(const):
                    return self.input[t, self.main_car] >= min_part_load * b_tec.var_size * rated_power

                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(self.set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]

        b_tec.disjunction_input_output = Disjunction(self.set_t, rule=bind_disjunctions)

        return b_tec


    def _performance_function_type_4(self, b_tec):
        """
        Piece-wise linear, minimal partload, includes constraints for slow (>1h) startup and shutdown trajectories.

        Based on Equations 9-11, 13 and 15 in Morales-España, G., Ramírez-Elizondo, L., & Hobbs, B. F. (2017). Hidden
        power system inflexibilities imposed by traditional unit commitment formulations. Applied Energy, 191, 223–238.
        https://doi.org/10.1016/J.APENERGY.2017.01.089

        :param b_tec: technology block
        :return: technology block
        """
        # Transformation required
        self.big_m_transformation_required = 1

        # Performance Parameters
        SU_time = self.performance_data['SU_time']
        SD_time = self.performance_data['SD_time']
        alpha1 = {}
        alpha2 = {}
        for car in self.performance_data['performance']['out']:
            bp_x = self.fitted_performance.coefficients[car]['bp_x']
            alpha1[car] = self.fitted_performance.coefficients[car]['alpha1']
            alpha2[car] = self.fitted_performance.coefficients[car]['alpha2']
        rated_power = self.fitted_performance.rated_power
        min_part_load = self.performance_data['min_part_load']

        if SU_time <= 0 and SD_time <= 0:
            warn('Having performance_function_type = 4 with no slow SU/SDs usually makes no sense.')
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
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                def init_y_off(const, i):
                    if t < len(self.set_t_full) - SU_time or i > SU_time - (len(self.set_t_full) - t):
                        return b_tec.var_y[t - i + SU_time + 1] == 0
                    else:
                        return b_tec.var_y[(t - i + SU_time + 1) - len(self.set_t_full)] == 0

                dis.const_y_off = Constraint(range(1, SU_time + 1), rule=init_y_off)

                def init_z_off(const, j):
                    if j <= t:
                        return b_tec.var_z[t - j + 1] == 0
                    else:
                        return b_tec.var_z[len(self.set_t_full) + (t - j + 1)] == 0

                dis.const_z_off = Constraint(range(1, SD_time + 1), rule=init_z_off)

                def init_input_off(const, car_input):
                    return self.input[t, car_input] == 0

                dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return self.output[t, car_output] == 0

                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            elif ind in range(1, SU_time + 1):  # technology in startup
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                def init_y_on(const):
                    if t < len(self.set_t_full) - SU_time or ind > SU_time - (len(self.set_t_full) - t):
                        return b_tec.var_y[t - ind + SU_time + 1] == 1
                    else:
                        return b_tec.var_y[(t - ind + SU_time + 1) - len(self.set_t_full)] == 1

                dis.const_y_on = Constraint(rule=init_y_on)

                def init_z_off(const):
                    if t < len(self.set_t_full) - SU_time or ind > SU_time - (len(self.set_t_full) - t):
                        return b_tec.var_z[t - ind + SU_time + 1] == 0
                    else:
                        return b_tec.var_z[(t - ind + SU_time + 1) - len(self.set_t_full)] == 0

                dis.const_z_off = Constraint(rule=init_z_off)

                def init_input_SU(cons):
                    return self.input[t, self.main_car] == b_tec.var_size * SU_trajectory[ind - 1]

                dis.const_input_SU = Constraint(rule=init_input_SU)

                def init_output_SU(const, car_output):
                    return self.output[t, car_output] == alpha1[car_output][0] * self.input[t, self.main_car] + \
                           alpha2[car_output][0] * b_tec.var_size * rated_power

                dis.const_output_SU = Constraint(b_tec.set_output_carriers, rule=init_output_SU)

            elif ind in range(SU_time + 1, SU_time + SD_time + 1):  # technology in shutdown
                ind_SD = ind - SU_time
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                def init_z_on(const):
                    if ind_SD <= t:
                        return b_tec.var_z[t - ind_SD + 1] == 1
                    else:
                        return b_tec.var_z[len(self.set_t_full) + (t - ind_SD + 1)] == 1

                dis.const_z_on = Constraint(rule=init_z_on)

                def init_y_off(const):
                    if ind_SD <= t:
                        return b_tec.var_y[t - ind_SD + 1] == 0
                    else:
                        return b_tec.var_y[len(self.set_t_full) + (t - ind_SD + 1)] == 0

                dis.const_y_off = Constraint(rule=init_y_off)

                def init_input_SD(cons):
                    return self.input[t, self.main_car] == b_tec.var_size * SD_trajectory[ind_SD - 1]

                dis.const_input_SD = Constraint(rule=init_input_SD)

                def init_output_SD(const, car_output):
                    return self.output[t, car_output] == alpha1[car_output][0] * self.input[t, self.main_car] + \
                           alpha2[car_output][0] * b_tec.var_size * rated_power

                dis.const_output_SD = Constraint(b_tec.set_output_carriers, rule=init_output_SD)

            elif ind > SU_time + SD_time:
                ind_bpx = ind - (SU_time + SD_time)
                dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

                def init_input_on1(const):
                    return self.input[t, self.main_car] >= bp_x[ind_bpx - 1] * b_tec.var_size * rated_power
                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return self.input[t, self.main_car] <= bp_x[ind_bpx] * b_tec.var_size * rated_power
                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return self.output[t, car_output] == \
                           alpha1[car_output][ind_bpx - 1] * self.input[t, self.main_car] + \
                           alpha2[car_output][ind_bpx - 1] * b_tec.var_size * rated_power
                dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_output_on)

                # min part load relation
                def init_min_partload(const):
                    return self.input[t, self.main_car] >= min_part_load * b_tec.var_size * rated_power

                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_SUSD_trajectory = Disjunct(self.set_t_full, s_indicators, rule=init_SUSD_trajectories)

        def bind_disjunctions_SUSD(dis, t):
            return [b_tec.dis_SUSD_trajectory[t, k] for k in s_indicators]

        b_tec.disjunction_SUSD_traject = Disjunction(self.set_t_full, rule=bind_disjunctions_SUSD)

        return b_tec

    def _define_ramping_rates(self, b_tec):
        """
        Constraints the inputs for a ramping rate

        :param b_tec: technology model block
        :return:
        """
        ramping_rate = self.performance_data['ramping_rate']

        def init_ramping_down_rate(const, t):
            if t > 1:
                return -ramping_rate <= self.input[t, self.main_car] - self.input[t - 1, self.main_car]
            else:
                return Constraint.Skip
        b_tec.const_ramping_down_rate = Constraint(self.set_t, rule=init_ramping_down_rate)

        def init_ramping_up_rate(const, t):
            if t > 1:
                return self.input[t, self.main_car] - self.input[t - 1, self.main_car] <= ramping_rate
            else:
                return Constraint.Skip
        b_tec.const_ramping_up_rate = Constraint(self.set_t, rule=init_ramping_up_rate)

        return b_tec
