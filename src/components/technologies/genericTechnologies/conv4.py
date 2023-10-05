from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import numpy as np

from ..utilities import FittedPerformance
from ..technology import Technology
from ...utilities import read_dict_value


class Conv4(Technology):

    def __init__(self,
                 tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance(self.performance_data)

        self.main_car = self.performance_data['main_output_carrier']

    def fit_technology_performance(self, node_data):
        """
        Fits conversion technology type 4 and returns fitted parameters as a dict

        :param node_data: contains data on demand, climate data, etc.
        :param performance_data: contains X and y data of technology performance
        :param performance_function_type: options for type of performance function (linear, piecewise,...)
        """

        climate_data = node_data.data['climate_data']

        self.fitted_performance.bounds['output'][self.performance_data['main_output_carrier']] = np.column_stack((np.zeros(shape=(len(climate_data))),
                                                                                     np.ones(shape=(len(climate_data)))))
        for car in self.fitted_performance.output_carrier:
            if not car == self.performance_data['main_output_carrier']:
                self.fitted_performance.bounds['output'][car] = self.fitted_performance.bounds['output'][self.performance_data['main_output_carrier']] \
                                                * self.performance_data['output_ratios'][car]

    def construct_tech_model(self, b_tec, energyhub):
        """
        Adds constraints to technology blocks for tec_type CONV4, i.e. :math:`output_{car} \leq S>)`

        This technology type resembles a technology with fixed output ratios and no inputs
        Two different performance function fits are possible. 

        **Constraint declarations:**

        - Size constraints are formulated on the output.

          .. math::
             Output_{t, maincarrier} \leq S

        - The ratios of outputs for all performance function types are fixed and given as:

          .. math::
            Output_{t, car} = {\\phi}_{car} * Output_{t, maincarrier}

        - ``performance_function_type == 1``: No further constraints:

        - ``performance_function_type == 2``: Minimal partload (makes big-m transformation required). If the
          technology is in on, it holds:

          .. math::
            Output_{maincarrier} \geq Output_{min} * S

          If the technology is off, output is set to 0:

          .. math::
             Output_{t, car} = 0

        :param obj b_tec: technology block
        :param Energyhub energyhub: energyhub instance
        :return: technology block
        """
        super(Conv4, self).construct_tech_model(b_tec, energyhub)

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        rated_power = self.fitted_performance.rated_power
        performance_function_type = performance_data['performance_function_type']
        phi = {}
        for car in self.performance_data['output_ratios']:
            phi[car] = self.performance_data['output_ratios'][car]

        if performance_function_type == 2:
            b_tec = self.__performance_function_type_2(b_tec)


        # Size constraints
        # constraint on output ratios
        def init_output_output(const, t, car_output):
            if car_output == self.main_car:
                return Constraint.Skip
            else:
                return self.output[t, car_output] == phi[car_output] * self.output[t, self.main_car]
        b_tec.const_output_output = Constraint(self.set_t, b_tec.set_output_carriers, rule=init_output_output)

        # size constraint based main carrier output
        def init_size_constraint(const, t):
            return self.output[t, self.main_car] <= b_tec.var_size * rated_power
        b_tec.const_size = Constraint(self.set_t, rule=init_size_constraint)

        return b_tec

    def __performance_function_type_2(self, b_tec):
        """
        Linear, minimal partload
        :param b_tec: technology block
        :return: technology block
        """
        # Transformation required
        self.big_m_transformation_required = 1

        # Performance Parameters
        rated_power = self.fitted_performance.rated_power
        min_part_load = self.performance_data['min_part_load']

        # define disjuncts
        s_indicators = range(0, 2)

        def init_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_output_off(const, car_output):
                    return self.output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # technology on
                def init_min_partload(const):
                    return self.output[t, self.main_car] >= min_part_load * b_tec.var_size * rated_power
                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_output = Disjunct(self.set_t, s_indicators, rule=init_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_output[t, i] for i in s_indicators]
        b_tec.disjunction_output = Disjunction(self.set_t, rule=bind_disjunctions)

        return b_tec

    def scale_model(self, b_tec, model, configuration):
        """
        Scales technology model
        """
        super(Conv4, self).scale_model(b_tec, model, configuration)

        if self.scaling_factors:

            f = self.scaling_factors
            f_global = configuration.scaling_factors

            # Constraints
            model.scaling_factor[b_tec.const_output_output] = read_dict_value(f, 'const_output_output') * f_global.energy_vars
            model.scaling_factor[b_tec.const_size] = read_dict_value(f, 'const_size') * f_global.energy_vars
            if self.performance_data['performance_function_type'] > 1:
                warn('Model Scaling for Conv4 only implemented for performance function type 1')

        return model
