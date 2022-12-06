from pyomo.environ import *
from pyomo.environ import units as u
from pyomo.gdp import *
import warnings


def constraints_tec_type_1(model, b_tec, tec_data):
    """ Adds constraints to technology blocks for tec_type 1 (renewable technology)
    :param model: full model
    :param b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    tec_fit = tec_data['fit']
    size_is_integer = tec_data['TechnologyPerf']['size_is_int']
    if size_is_integer:
        unit_size = u.dimensionless
    else:
        unit_size = u.MW

    if 'curtailment' in tec_data['TechnologyPerf']:
        curtailment = tec_data['TechnologyPerf']['curtailment']
    else:
        curtailment = 0

    # Set capacity factors as a parameter
    def init_capfactors(para, t):
        return tec_fit['capacity_factor'][t - 1]
    b_tec.para_capfactor = Param(model.set_t, domain=Reals, rule=init_capfactors)

    if curtailment == 0:  # no curtailment allowed (default
        def init_input_output(const, t, c_output):
            return b_tec.var_output[t, c_output] == \
                   b_tec.para_capfactor[t] * b_tec.var_size * unit_size
        b_tec.const_input_output = Constraint(model.set_t, b_tec.set_output_carriers, rule=init_input_output)

    elif curtailment == 1:  # continuous curtailment
        def init_input_output(const, t, c_output):
            return b_tec.var_output[t, c_output] <= \
                   b_tec.para_capfactor[t] * b_tec.var_size * unit_size
        b_tec.const_input_output = Constraint(model.set_t, b_tec.set_output_carriers,
                                              rule=init_input_output)

    elif curtailment == 2:  # discrete curtailment
        b_tec.var_size_on = Var(model.set_t, within=NonNegativeIntegers, bounds=(b_tec.para_size_min, b_tec.para_size_max))
        def init_curtailed_units(const, t):
            return b_tec.var_size_on[t] <= b_tec.var_size
        b_tec.const_curtailed_units = Constraint(model.set_t, rule=init_curtailed_units)
        def init_input_output(const, t, c_output):
            return b_tec.var_output[t, c_output] <= \
                   b_tec.para_capfactor[t] * b_tec.var_size_on[t] * unit_size
        b_tec.const_input_output = Constraint(model.set_t, b_tec.set_output_carriers,
                                              rule=init_input_output)

    return b_tec

def constraints_tec_type_2(model, b_tec, tec_data):
    """ Adds constraints for technology type 2 (n inputs -> n output, fuel and output substitution)
    :param model: full model
    :param b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    tec_fit = tec_data['fit']
    performance_function_type = tec_data['TechnologyPerf']['performance_function_type']
    if 'min_part_load' in tec_fit:
        min_part_load = tec_fit['min_part_load']
    else:
        min_part_load = 0

    # Formulate Constraints for each performance function type
    # linear through origin
    if performance_function_type == 1:
        def init_input_output(const, t):
            return sum(b_tec.var_output[t, car_output]
                       for car_output in b_tec.set_output_carriers) == \
                   alpha1 * sum(b_tec.var_input[t, car_input]
                                for car_input in b_tec.set_input_carriers)

        b_tec.const_input_output = Constraint(model.set_t, rule=init_input_output)

    # linear not through origin
    elif performance_function_type == 2:
        if min_part_load == 0:
            warnings.warn(
                'Having performance_function_type = 2 with no part-load usually makes no sense. Error occured for ' + tec)

        # define disjuncts
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const, car_input):
                    return b_tec.var_input[t, car_input] == 0
                dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return b_tec.var_output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)
            else:  # technology on
                # input-output relation
                def init_input_output_on(con):
                    return sum(b_tec.var_output[t, car_output] for car_output in b_tec.set_output_carriers) == \
                           alpha2 * sum(b_tec.var_input[t, car_input] for car_input in b_tec.set_input_carriers) + \
                           alpha1
                dis.const_input_output_on = Constraint(rule=init_input_output_on)

                # min part load relation
                def init_min_partload(con):
                    return sum(b_tec.var_input[t, car_input]
                               for car_input in b_tec.set_input_carriers) >= min_part_load * b_tec.var_size
                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(model.set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(model.set_t, rule=bind_disjunctions)

    # piecewise affine function
    elif performance_function_type == 3:
        s_indicators = range(0, len(bp_x))

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const, car_input):
                    return b_tec.var_input[t, car_input] == 0
                dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return b_tec.var_output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # piecewise definition
                def init_input_on1(const):
                    return sum(b_tec.var_input[t, car_input] for car_input in b_tec.set_input_carriers) >= \
                           bp_x[ind - 1] * b_tec.var_size
                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return sum(b_tec.var_input[t, car_input] for car_input in b_tec.set_input_carriers) <= \
                           bp_x[ind] * b_tec.var_size
                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const):
                    return sum(b_tec.var_output[t, car_output] for car_output in b_tec.set_output_carriers) == \
                           alpha2[ind - 1] * sum(b_tec.var_input[t, car_input] for car_input in b_tec.set_input_carriers) + \
                           alpha1[ind - 1]
                dis.const_input_output_on = Constraint(rule=init_output_on)

        b_tec.dis_input_output = Disjunct(model.set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(model.set_t, rule=bind_disjunctions)

    return b_tec

def constraints_tec_type_3(model, b_tec, tec_data):
    tec_fit = tec_data['fit']
    performance_function_type = tec_data['TechnologyPerf']['performance_function_type']
    # Get performance parameters
    alpha1 = tec_fit['alpha1']
    if performance_function_type == 2:
        alpha2 = tec_fit['alpha2']
    if 'min_part_load' in tec_fit:
        min_part_load = tec_fit['min_part_load']
    else:
        min_part_load = 0
    if performance_function_type == 3:
        bp_x = tec_fit['bp_x']

    # Formulate Constraints for each performance function type
    # linear through origin
    if performance_function_type == 1:
        def init_input_output(const, car_output, t):
            return b_tec.var_output[t, car_output] == \
                   alpha1[car_output] * sum(b_tec.var_input[t, car_input]
                                            for car_input in b_tec.set_input_carriers)
        b_tec.const_input_output = Constraint(model.set_t, b_tec.set_output_carriers,
                                              rule=init_input_output)

    elif performance_function_type == 2:
        if min_part_load == 0:
            warnings.warn(
                'Having performance_function_type = 2 with no part-load usually makes no sense.')
            # TODO: Switch to performance_function_type 1!

        # define disjuncts
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const, car_input):
                    return b_tec.var_input[t, car_input] == 0
                dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return b_tec.var_output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)
            else:  # technology on
                # input-output relation
                def init_input_output_on(const, car_output):
                    return b_tec.var_output[t, car_output] == \
                           alpha2[car_output] * sum(b_tec.var_input[t, car_input]
                                                    for car_input in b_tec.set_input_carriers) + alpha1[car_output]
                dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_input_output_on)

                # min part load relation
                def init_min_partload(const):
                    return sum(b_tec.var_input[t, car_input]
                               for car_input in b_tec.set_input_carriers) >= min_part_load * b_tec.var_size
                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(model.set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(model.set_t, rule=bind_disjunctions)

    # piecewise affine function
    elif performance_function_type == 3:
        s_indicators = range(0, len(bp_x))

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const, car_input):
                    return b_tec.var_input[t, car_input] == 0
                dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return b_tec.var_output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # piecewise definition
                def init_input_on1(const):
                    return sum(b_tec.var_input[t, car_input] for car_input in b_tec.set_input_carriers) >= \
                           bp_x[ind - 1] * b_tec.var_size
                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return sum(b_tec.var_input[t, car_input] for car_input in b_tec.set_input_carriers) <= \
                           bp_x[ind] * b_tec.var_size
                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return b_tec.var_output[t, car_output] == \
                           alpha2[car_output][ind - 1] * sum(b_tec.var_input[t, car_input]
                                                             for car_input in b_tec.set_input_carriers) + \
                           alpha1[car_output][ind - 1]
            dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_output_on)

        b_tec.dis_input_output = Disjunct(model.set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(model.set_t, rule=bind_disjunctions)

    return b_tec

def constraints_tec_type_6(model, b_tec, tec_data):
    tec_fit = tec_data['fit']

    # Additional decision variables
    b_tec.var_storage_level = Var(model.set_t, b_tec.set_input_carriers, domain=NonNegativeReals)

    # Additional parameters
    b_tec.para_eta_in = Param(domain=NonNegativeReals, initialize=tec_fit['eta_in'])
    b_tec.para_eta_out = Param(domain=NonNegativeReals, initialize=tec_fit['eta_out'])
    b_tec.para_eta_lambda = Param(domain=NonNegativeReals, initialize=tec_fit['lambda'])
    b_tec.para_charge_max = Param(domain=NonNegativeReals, initialize=tec_fit['charge_max'])
    b_tec.para_discharge_max = Param(domain=NonNegativeReals, initialize=tec_fit['discharge_max'])
    def init_ambient_loss_factor(para, t):
        return tec_fit['ambient_loss_factor'].values[t - 1]
    b_tec.para_ambient_loss_factor = Param(model.set_t, domain=NonNegativeReals, rule=init_ambient_loss_factor)

    # Size constraint
    def init_size_constraint(const, t, car):
        return b_tec.var_storage_level[t, car] <= b_tec.var_size
    b_tec.const_size = Constraint(model.set_t, b_tec.set_input_carriers, rule=init_size_constraint)

    # Storage level calculation
    def init_storage_level(const, t, car):
        if t == 1: # couple first and last time interval
            return b_tec.var_storage_level[t, car] == \
                  b_tec.var_storage_level[max(model.set_t), car] * (1 - b_tec.para_eta_lambda) - \
                  b_tec.para_ambient_loss_factor[max(model.set_t)] * b_tec.var_storage_level[max(model.set_t), car] + \
                  b_tec.para_eta_in * b_tec.var_input[t, car] - \
                  1 / b_tec.para_eta_out * b_tec.var_output[t, car]
        else: # all other time intervalls
            return b_tec.var_storage_level[t, car] == \
                b_tec.var_storage_level[t-1, car] * (1 - b_tec.para_eta_lambda) - \
                b_tec.para_ambient_loss_factor[t] * b_tec.var_storage_level[t-1, car] + \
                b_tec.para_eta_in * b_tec.var_input[t, car] - \
                1/b_tec.para_eta_out * b_tec.var_output[t, car]
    b_tec.const_storage_level = Constraint(model.set_t, b_tec.set_input_carriers, rule=init_storage_level)

    def init_maximal_charge(const,t,car):
        return b_tec.var_input[t, car] <= b_tec.para_eta_in * b_tec.var_size
    b_tec.const_max_charge = Constraint(model.set_t, b_tec.set_input_carriers, rule=init_maximal_charge)

    def init_maximal_discharge(const,t,car):
        return b_tec.var_output[t, car] <= b_tec.para_eta_out * b_tec.var_size
    b_tec.const_max_discharge = Constraint(model.set_t, b_tec.set_input_carriers, rule=init_maximal_discharge)