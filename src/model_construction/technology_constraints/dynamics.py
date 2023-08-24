from pyomo.environ import *
import warnings
from pyomo.gdp import *
import src.global_variables as global_variables
import src.model_construction as mc

def constraints_ramping_rate(b_tec, tec_data, energyhub):
    """ Add description here """

    set_t = energyhub.model.set_t_full
    ramping_rate = tec_data.performance_data['ramping_rate']
    input = b_tec.var_input
    main_car = tec_data.performance_data['main_input_carrier']
    technology_model = tec_data.technology_model

    def init_ramping_down_rate(const, t):
        if t > 1:
            if technology_model == 'CONV1' or technology_model == 'CONV2':
                return -ramping_rate <= sum(input[t, car_input] - input[t-1, car_input]
                                            for car_input in b_tec.set_input_carriers)
            elif technology_model == 'CONV3':
                return -ramping_rate <= input[t, main_car] - input[t-1, main_car]
            else:
                warnings.warn(
                    'Ramping rate can not be used for this performance type.')
        else:
            return Constraint.Skip
    b_tec.const_ramping_down_rate = Constraint(set_t, rule=init_ramping_down_rate)

    def init_ramping_up_rate(const, t):
        if t > 1:
            if technology_model == 'CONV1' or technology_model == 'CONV2':
                return sum(input[t, car_input] - input[t-1, car_input]
                           for car_input in b_tec.set_input_carriers) <= ramping_rate
            elif technology_model == 'CONV3':
                return input[t, main_car] - input[t-1, main_car] <= ramping_rate
            else:
                warnings.warn(
                    'Ramping rate can not be used for this performance type.')
        else:
            return Constraint.Skip
    b_tec.const_ramping_up_rate = Constraint(set_t, rule=init_ramping_up_rate)


# def constraint_standby_power(b_tec, tec_data, energyhub)
#     print('this')
#     return

