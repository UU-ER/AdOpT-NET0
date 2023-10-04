from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn

from ..genericTechnologies.utilities import fit_performance_generic_tecs
from ..technology import Technology


class Conv2(Technology):

    def __init__(self,
                 tec_data):
        super().__init__(tec_data)

        self.fitted_performance = None

    def fit_technology_performance(self, node_data):
        """
        Fits conversion technology type 2 and returns fitted parameters as a dict

        :param performance_data: contains X and y data of technology performance
        :param performance_function_type: options for type of performance function (linear, piecewise,...)
        :param nr_seg: number of segments on piecewise defined function
        """

        climate_data = node_data.data['climate_data']

        if self.performance_data['size_based_on'] == 'output':
            raise Exception('size_based_on == output for CONV2 not possible.')
        self.fitted_performance = fit_performance_generic_tecs(self.performance_data, time_steps=len(climate_data))

    def construct_tech_model(self, b_tec, energyhub):
        """
        Adds constraints to technology blocks for tec_type CONV2, i.e. :math:`output_{car} = f_{car}(\sum(inputs))`

        This technology type resembles a technology with full input substitution, but different performance functions
        for the respective output carriers.
        Three different performance function fits are possible. 

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

        - ``performance_function_type == 3``: Piecewise linear performance function (makes big-m transformation required).
          The same constraints as for ``performance_function_type == 2`` with the exception that the performance function
          is defined piecewise for the respective number of pieces

        :param obj b_tec: technology block
        :param Energyhub energyhub: energyhub instance
        :return: technology block
        """
        super(Conv2, self).construct_tech_model(b_tec, energyhub)

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

        # Size constraints
        # size constraint based on sum of inputs
        def init_size_constraint(const, t):
            return sum(self.input[t, car_input] for car_input in b_tec.set_input_carriers) \
                   <= b_tec.var_size * rated_power
        b_tec.const_size = Constraint(self.set_t, rule=init_size_constraint)

        # Maximum input of carriers
        if 'max_input' in performance_data:
            b_tec.set_max_input_carriers = Set(initialize=performance_data['max_input'].keys())

            def init_max_input(const, t, car):
                return self.input[t, car] <= performance_data['max_input'][car] * \
                       sum(self.input[t, car_input] for car_input in b_tec.set_input_carriers)

            b_tec.const_max_input = Constraint(self.set_t, b_tec.set_max_input_carriers, rule=init_max_input)

        return b_tec

    def __performance_function_type_1(self, b_tec):
        """
        Linear, no minimal partload, through origin
        :param b_tec: technology block
        :return: technology block
        """
        # Performance parameter:
        alpha1 = {}
        for car in self.performance_data['performance']['out']:
            alpha1[car] = self.fitted_performance.coefficients[car]['alpha1']

        # Input-output correlation
        def init_input_output(const, t, car_output):
            return self.output[t, car_output] == \
                   alpha1[car_output] * sum(self.input[t, car_input]
                                            for car_input in b_tec.set_input_carriers)
        b_tec.const_input_output = Constraint(self.set_t, b_tec.set_output_carriers,
                                              rule=init_input_output)

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
        alpha1 = {}
        alpha2 = {}
        for car in self.performance_data['performance']['out']:
            alpha1[car] = self.fitted_performance.coefficients[car]['alpha1']
            alpha2[car] = self.fitted_performance.coefficients[car]['alpha2']
        rated_power = self.fitted_performance.rated_power
        min_part_load = self.performance_data['min_part_load']

        if min_part_load == 0:
            warn('Having performance_function_type = 2 with no part-load usually makes no sense. Error occured for ' + b_tec.local_name)

        # define disjuncts
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const, car_input):
                    return self.input[t, car_input] == 0

                dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return self.output[t, car_output] == 0

                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)
            else:  # technology on
                # input-output relation
                def init_input_output_on(const, car_output):
                    return self.output[t, car_output] == \
                           alpha1[car_output] * sum(self.input[t, car_input] for car_input
                                                    in b_tec.set_input_carriers) \
                           + alpha2[car_output] * b_tec.var_size * rated_power

                dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_input_output_on)

                # min part load relation
                def init_min_partload(const):
                    return sum(self.input[t, car_input]
                               for car_input in b_tec.set_input_carriers) >= \
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

        s_indicators = range(0, len(bp_x))

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const, car_input):
                    return self.input[t, car_input] == 0

                dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return self.output[t, car_output] == 0

                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # piecewise definition
                def init_input_on1(const):
                    return sum(self.input[t, car_input] for car_input in b_tec.set_input_carriers) >= \
                           bp_x[ind - 1] * b_tec.var_size * rated_power

                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return sum(self.input[t, car_input] for car_input in b_tec.set_input_carriers) <= \
                           bp_x[ind] * b_tec.var_size * rated_power

                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return self.output[t, car_output] == \
                           alpha1[car_output][ind - 1] * sum(self.input[t, car_input]
                                                             for car_input in b_tec.set_input_carriers) + \
                           alpha2[car_output][ind - 1] * b_tec.var_size * rated_power

                dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_output_on)

                # min part load relation
                def init_min_partload(const):
                    return sum(self.input[t, car_input]
                               for car_input in b_tec.set_input_carriers) >= \
                           min_part_load * b_tec.var_size * rated_power

                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(self.set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]

        b_tec.disjunction_input_output = Disjunction(self.set_t, rule=bind_disjunctions)

        return b_tec

    def scale_model(self, b_tec, model, configuration):
        """
        Scales technology model
        """
        super(Conv2, self).scale_model(b_tec, model, configuration)

        if self.scaling_factors:

            f = self.scaling_factors

            # Constraints
            model.scaling_factor[b_tec.const_size] = f['const_size']
            if b_tec.find_component('const_max_input'):
                model.scaling_factor[b_tec.const_max_input] = f['const_max_input']
            if self.performance_data['performance_function_type'] == 1:
                model.scaling_factor[b_tec.const_input_output] = f['const_input_output']
            else:
                warn('Model Scaling for Conv2 only implemented for performance function type 1')

        return model
