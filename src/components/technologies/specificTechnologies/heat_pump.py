from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
import statsmodels.api as sm

from ..utilities import FittedPerformance, fit_piecewise_function, fit_linear_function
from ..technology import Technology


class HeatPump(Technology):

    def __init__(self,
                 tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance(tec_data)

    def fit_technology_performance(self, node_data):
        """
        Performs fitting for technology type HeatPump

        The equations are based on Ruhnau, O., Hirth, L., & Praktiknjo, A. (2019). Time series of heat demand and
        heat pump efficiency for energy system modeling. Scientific Data, 6(1).
        https://doi.org/10.1038/s41597-019-0199-y

        :param tec_data: technology data
        :param climate_data: climate data
        :return:
        """
        # Climate data & Number of timesteps
        climate_data = node_data.data['climate_data']
        time_steps = len(climate_data)

        # Ambient air temperature
        T = copy.deepcopy(climate_data['temp_air'])

        # Determine T_out
        if self.performance_data['application'] == 'radiator_heating':
            t_out = 40 - T
        elif self.performance_data['application'] == 'floor_heating':
            t_out = 30 - 0.5 * T
        else:
            t_out = self.performance_data['T_out']

        # Determine delta T
        delta_T = t_out - T

        # Determine COP
        if self.name == 'HeatPump_AirSourced':
            cop = 6.08 - 0.09 * delta_T + 0.0005 * delta_T ** 2
        elif self.name == 'HeatPump_GroundSourced':
            cop = 10.29 - 0.21 * delta_T + 0.0012 * delta_T ** 2
        elif self.name == 'HeatPump_WaterSourced':
            cop = 9.97 - 0.20 * delta_T + 0.0012 * delta_T ** 2

        print('Deriving performance data for Heat Pump...')

        if self.performance_data['performance_function_type'] == 1 or self.performance_data['performance_function_type'] == 2:  # Linear performance function
            size_alpha = 1
        elif self.performance_data['performance_function_type'] == 3:
            size_alpha = 2
        else:
            raise Exception("performance_function_type must be an integer between 1 and 3")

        fit = {}
        fit['out'] = {}
        alpha1 = np.empty(shape=(time_steps, size_alpha))
        alpha2 = np.empty(shape=(time_steps, size_alpha))
        bp_x = np.empty(shape=(time_steps, size_alpha + 1))

        for idx, cop_t in enumerate(cop):
            if idx % 100 == 1:
                print("\rComplete: ", round(idx / time_steps, 2) * 100, "%", end="")

            if self.performance_data['performance_function_type'] == 1:
                x = np.linspace(self.performance_data['min_part_load'], 1, 9)
                y = (x / (1 - 0.9 * (1 - x))) * cop_t * x
                coeff = fit_linear_function(x, y)
                alpha1[idx, :] = coeff[0]

            elif self.performance_data['performance_function_type'] == 2:
                x = np.linspace(self.performance_data['min_part_load'], 1, 9)
                y = (x / (1 - 0.9 * (1 - x))) * cop_t * x
                x = sm.add_constant(x)
                coeff = fit_linear_function(x, y)
                alpha1[idx, :] = coeff[1]
                alpha2[idx, :] = coeff[0]

            elif self.performance_data['performance_function_type'] == 3:  # piecewise performance function
                y = {}
                x = np.linspace(self.performance_data['min_part_load'], 1, 9)
                y['out'] = (x / (1 - 0.9 * (1 - x))) * cop_t * x
                time_step_fit = fit_piecewise_function(x, y, 2)
                alpha1[idx, :] = time_step_fit['out']['alpha1']
                alpha2[idx, :] = time_step_fit['out']['alpha2']
                bp_x[idx, :] = time_step_fit['out']['bp_x']
        print("Complete: ", 100, "%")

        # Calculate input bounds
        fit['output_bounds'] = {}
        fit['coeff'] = {}
        if self.performance_data['performance_function_type'] == 1:
            fit['coeff']['alpha1'] = alpha1.round(5)
            for c in self.performance_data['output_carrier']:
                fit['output_bounds'][c] = np.column_stack((np.zeros(shape=(time_steps)),
                                                           np.ones(shape=(time_steps)) * fit['coeff']['alpha1']))

        elif self.performance_data['performance_function_type'] == 2:  # Linear performance function
            fit['coeff']['alpha1'] = alpha1.round(5)
            fit['coeff']['alpha2'] = alpha2.round(5)
            for c in self.performance_data['output_carrier']:
                fit['output_bounds'][c] = np.column_stack((np.zeros(shape=(time_steps)),
                                                           np.ones(shape=(time_steps)) * fit['coeff']['alpha1'] + \
                                                           fit['coeff']['alpha2']))

        elif self.performance_data['performance_function_type'] == 3:  # Piecewise performance function
            fit['coeff']['alpha1'] = alpha1.round(5)
            fit['coeff']['alpha2'] = alpha2.round(5)
            fit['coeff']['bp_x'] = bp_x.round(5)
            for c in self.performance_data['output_carrier']:
                fit['output_bounds'][c] = np.column_stack((np.zeros(shape=(time_steps)),
                                                           fit['coeff']['alpha1'][:, -1] + \
                                                           fit['coeff']['alpha2'][:, -1]))

        # Output Bounds
        self.fitted_performance.bounds['output'] = fit['output_bounds']
        # Input Bounds
        for car in self.performance_data['input_carrier']:
            self.fitted_performance.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                            np.ones(shape=(time_steps))))
        # Coefficients
        self.fitted_performance.coefficients = fit['coeff']

        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 1

    def construct_tech_model(self, b_tec, energyhub):
        """
        Adds constraints to technology blocks for tec_type HP (Heat Pump)

        Three different types of heat pumps are possible: air sourced ('HeatPump_AirSourced'), ground sourced
        ('HeatPump_GroundSourced') and water sourced ('HeatPump_WaterSourced'). Additionally, a heating curve is determined for
        heating for buildings. Then, the application needs to be set to either 'floor_heating' or 'radiator_heating'
        in the data file. Otherwise, the output temperature of the heat pump can also be set to a given temperature.
        The coefficient of performance at full load is calculated in the respective fitting function with the equations
        provided in Ruhnau, O., Hirth, L., & Praktiknjo, A. (2019). Time series of heat demand and
        heat pump efficiency for energy system modeling. Scientific Data, 6(1). https://doi.org/10.1038/s41597-019-0199-y

        The part load behavior is modelled after equation (3) in Xu, Z., Li, H., Xu, W., Shao, S., Wang, Z., Gou, X., Zhao,
        M., & Li, J. (2022). Investigation on the efficiency degradation characterization of low ambient temperature
        air source heat pump under partial load operation. International Journal of Refrigeration,
        133, 99–110. https://doi.org/10.1016/J.IJREFRIG.2021.10.002

        Essentially, the equations for the heat pump model are the same as for generic conversion technology type 1 (with
        time-dependent performance parameter).

        :param obj b_tec: technology block
        :param Energyhub energyhub: energyhub instance
        :return: technology block
        """
        super(HeatPump, self).construct_tech_model(b_tec, energyhub)

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        rated_power = self.fitted_performance.rated_power
        performance_function_type = performance_data['performance_function_type']

        if performance_function_type == 1:
            b_tec = self.__performance_function_type_1(b_tec)
        elif performance_function_type == 2:
            b_tec = self.__performance_function_type_2(b_tec)
        elif performance_function_type == 3:
            b_tec = self.__performance_function_type_3(b_tec)

        # size constraint based on input
        def init_size_constraint(const, t):
            return self.input[t, 'electricity'] <= b_tec.var_size * rated_power
        b_tec.const_size = Constraint(self.set_t, rule=init_size_constraint)

        # RAMPING RATES
        if "ramping_rate" in self.performance_data:
            if not self.performance_data['ramping_rate']   == -1:
                b_tec = self.__define_ramping_rates(b_tec)

        return b_tec

    def __performance_function_type_1(self, b_tec):
        """
        Linear, no minimal partload, through origin
        :param b_tec: technology block
        :return: technology block
        """
        # Get performance parameters
        alpha1 = self.fitted_performance.coefficients['alpha1']

        def init_input_output(const, t):
            return self.output[t, 'heat'] == alpha1[t - 1] * self.input[t, 'electricity']
        b_tec.const_input_output = Constraint(self.set_t, rule=init_input_output)

        return b_tec

    def __performance_function_type_2(self, b_tec):
        """
        Linear, minimal partload
        :param b_tec: technology block
        :return: technology block
        """
        # Get performance parameters
        alpha1 = self.fitted_performance.coefficients['alpha1']
        alpha2 = self.fitted_performance.coefficients['alpha2']
        min_part_load = self.performance_data['min_part_load']
        rated_power = self.fitted_performance.rated_power

        # define disjuncts for on/off
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const):
                    return self.input[t, 'electricity'] == 0

                dis.const_input = Constraint(rule=init_input_off)

                def init_output_off(const):
                    return self.output[t, 'heat'] == 0

                dis.const_output_off = Constraint(rule=init_output_off)
            else:  # technology on
                # input-output relation
                def init_input_output_on(const):
                    return self.output[t, 'heat'] == alpha1[t - 1] * self.input[t, 'electricity'] + \
                           alpha2[t - 1] * b_tec.var_size * rated_power

                dis.const_input_output_on = Constraint(rule=init_input_output_on)

                # min part load relation
                def init_min_partload(const):
                    return self.input[t, 'electricity'] >= \
                           min_part_load * b_tec.var_size * rated_power

                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(self.set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]

        b_tec.disjunction_input_output = Disjunction(self.set_t, rule=bind_disjunctions)

        return b_tec


    def __performance_function_type_3(self, b_tec):
        """
        Piece-wise linear, minimal partload
        :param b_tec: technology block
        :return: technology block
        """
        # Get performance parameters
        alpha1 = self.fitted_performance.coefficients['alpha1']
        alpha2 = self.fitted_performance.coefficients['alpha2']
        bp_x = self.fitted_performance.coefficients['bp_x']
        min_part_load = self.performance_data['min_part_load']
        rated_power = self.fitted_performance.rated_power

        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const):
                    return self.input[t, 'electricity'] == 0

                dis.const_input_off = Constraint(rule=init_input_off)

                def init_output_off(const):
                    return self.output[t, 'heat'] == 0

                dis.const_output_off = Constraint(rule=init_output_off)

            else:  # piecewise definition
                def init_input_on1(const):
                    return self.input[t, 'electricity'] >= \
                           bp_x[t - 1, ind] * b_tec.var_size * rated_power

                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return self.input[t, 'electricity'] <= \
                           bp_x[t - 1, ind + 1] * b_tec.var_size * rated_power

                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const):
                    return self.output[t, 'heat'] == \
                           alpha1[t - 1, ind - 1] * self.input[t, 'electricity'] + \
                           alpha2[t - 1, ind - 1] * b_tec.var_size * rated_power

                dis.const_input_output_on = Constraint(rule=init_output_on)

                # min part load relation
                def init_min_partload(const):
                    return self.input[t, 'electricity'] >= \
                           min_part_load * b_tec.var_size * rated_power

                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(self.set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(self.set_t, rule=bind_disjunctions)

        return b_tec

    def __define_ramping_rates(self, b_tec):
        """
        Constraints the inputs for a ramping rate

        :param b_tec: technology model block
        :return:
        """
        ramping_rate = self.performance_data['ramping_rate']

        def init_ramping_down_rate(const, t):
            if t > 1:
                return -ramping_rate <= sum(self.input[t, car_input] - self.input[t-1, car_input]
                                                for car_input in b_tec.set_input_carriers)
            else:
                return Constraint.Skip
        b_tec.const_ramping_down_rate = Constraint(self.set_t, rule=init_ramping_down_rate)

        def init_ramping_up_rate(const, t):
            if t > 1:
                return sum(self.input[t, car_input] - self.input[t-1, car_input]
                               for car_input in b_tec.set_input_carriers) <= ramping_rate
            else:
                return Constraint.Skip
        b_tec.const_ramping_up_rate = Constraint(self.set_t, rule=init_ramping_up_rate)

        return b_tec
