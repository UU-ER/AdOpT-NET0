from pyomo.environ import *
from pyomo.gdp import *
import warnings
import src.global_variables as global_variables

def constraints_tec_RES(b_tec, tec_data, energyhub):
    """
    Adds constraints to technology blocks for tec_type RES (renewable technology)

    **Parameter declarations:**

    - Capacity Factor of technology for each time step. The capacity factor has been calculated in
      ``src.model_construction.technology_performance_fitting``

    **Constraint declarations:**

    - Output of technology. The output can be curtailed in three different ways. For ``curtailment == 0``, there is
      no curtailment possible. For ``curtailment == 1``, the curtailment is continuous. For ``curtailment == 2``,
      the size needs to be an integer, and the technology can only be curtailed discretely, i.e. by turning full
      modules off. For ``curtailment == 0`` (default), it holds:

    .. math::
        Output_{t, car} = CapFactor_t * Size

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    model = energyhub.model

    # DATA OF TECHNOLOGY
    performance_data = tec_data.performance_data
    coeff = tec_data.fitted_performance.coefficients
    rated_power = tec_data.fitted_performance.rated_power
    modelled_with_full_res = tec_data.modelled_with_full_res

    if 'curtailment' in performance_data:
        curtailment = performance_data['curtailment']
    else:
        curtailment = 0

    # Full or reduced resolution
    if global_variables.clustered_data and not modelled_with_full_res:
        output = b_tec.var_output_aux
        set_t = model.set_t_clustered
    else:
        output = b_tec.var_output
        set_t = model.set_t_full

    # PARAMETERS
    # Set capacity factors
    capfactor = coeff['capfactor']

    # CONSTRAINTS
    if curtailment == 0:  # no curtailment allowed (default)
        def init_input_output(const, t, c_output):
            return output[t, c_output] == \
                   capfactor[t-1] * b_tec.var_size * rated_power
        b_tec.const_input_output = Constraint(set_t, b_tec.set_output_carriers, rule=init_input_output)

    elif curtailment == 1:  # continuous curtailment
        def init_input_output(const, t, c_output):
            return output[t, c_output] <= \
                  capfactor[t-1] * b_tec.var_size * rated_power
        b_tec.const_input_output = Constraint(set_t, b_tec.set_output_carriers,
                                              rule=init_input_output)

    elif curtailment == 2:  # discrete curtailment
        b_tec.var_size_on = Var(set_t, within=NonNegativeIntegers, bounds=(b_tec.para_size_min, b_tec.para_size_max))
        def init_curtailed_units(const, t):
            return b_tec.var_size_on[t] <= b_tec.var_size
        b_tec.const_curtailed_units = Constraint(set_t, rule=init_curtailed_units)
        def init_input_output(const, t, c_output):
            return output[t, c_output] == \
                  capfactor[t-1] * b_tec.var_size_on[t] * rated_power
        b_tec.const_input_output = Constraint(set_t, b_tec.set_output_carriers,
                                              rule=init_input_output)

    return b_tec

def constraints_tec_CONV1(b_tec, tec_data, energyhub):
    """
    Adds constraints to technology blocks for tec_type CONV1, i.e. :math:`\sum(output) = f(\sum(inputs))`

    This technology type resembles a technology with full input and output substitution.
    As for all conversion technologies, three different performance function fits are possible. The performance
    functions are fitted in ``src.model_construction.technology_performance_fitting``.

    **Constraint declarations:**

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

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    model = energyhub.model

    # DATA OF TECHNOLOGY
    performance_data = tec_data.performance_data
    coeff = tec_data.fitted_performance.coefficients
    rated_power = tec_data.fitted_performance.rated_power
    modelled_with_full_res = tec_data.modelled_with_full_res


    # Full or reduced resolution
    if global_variables.clustered_data and not modelled_with_full_res:
        input = b_tec.var_input_aux
        output = b_tec.var_output_aux
        set_t = model.set_t_clustered
    else:
        input = b_tec.var_input
        output = b_tec.var_output
        set_t = model.set_t_full

    performance_function_type = performance_data['performance_function_type']

    # Get performance parameters
    alpha1 = coeff['out']['alpha1']
    if performance_function_type == 2:
        alpha2 = coeff['out']['alpha2']
    if performance_function_type == 3:
        bp_x = coeff['bp_x']
        alpha2 = coeff['out']['alpha2']

    min_part_load = performance_data['min_part_load']

    if performance_function_type >= 2:
        global_variables.big_m_transformation_required = 1

    # LINEAR, NO MINIMAL PARTLOAD, THROUGH ORIGIN
    if performance_function_type == 1:
        def init_input_output(const, t):
            return sum(output[t, car_output]
                       for car_output in b_tec.set_output_carriers) == \
                   alpha1 * sum(input[t, car_input]
                                for car_input in b_tec.set_input_carriers)
        b_tec.const_input_output = Constraint(set_t, rule=init_input_output)

    # LINEAR, MINIMAL PARTLOAD
    elif performance_function_type == 2:
        if min_part_load == 0:
            warnings.warn(
                'Having performance_function_type = 2 with no part-load usually makes no sense. Error occured for ' + b_tec.local_name)

        # define disjuncts for on/off
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const, car_input):
                    return input[t, car_input] == 0
                dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)
            else:  # technology on
                # input-output relation
                def init_input_output_on(const):
                    return sum(output[t, car_output] for car_output in b_tec.set_output_carriers) == \
                           alpha1 * sum(input[t, car_input] for car_input in b_tec.set_input_carriers) + \
                           alpha2 * b_tec.var_size * rated_power
                dis.const_input_output_on = Constraint(rule=init_input_output_on)

                # min part load relation
                def init_min_partload(const):
                    return sum(input[t, car_input]
                               for car_input in b_tec.set_input_carriers) >= \
                           min_part_load * b_tec.var_size * rated_power
                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(set_t, rule=bind_disjunctions)

    # PIECEWISE-AFFINE
    elif performance_function_type == 3:
        s_indicators = range(0, len(bp_x))

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const, car_input):
                    return input[t, car_input] == 0
                dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # piecewise definition
                def init_input_on1(const):
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) >= \
                           bp_x[ind - 1] * b_tec.var_size * rated_power
                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) <= \
                           bp_x[ind] * b_tec.var_size * rated_power
                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const):
                    return sum(output[t, car_output] for car_output in b_tec.set_output_carriers) == \
                           alpha1[ind - 1] * sum(input[t, car_input] for car_input in b_tec.set_input_carriers) + \
                           alpha2[ind - 1] * b_tec.var_size * rated_power
                dis.const_input_output_on = Constraint(rule=init_output_on)

                # min part load relation
                def init_min_partload(const):
                    return sum(input[t, car_input]
                               for car_input in b_tec.set_input_carriers) >= \
                           min_part_load * b_tec.var_size * rated_power
                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(set_t, rule=bind_disjunctions)

    # size constraint based on sum of inputs
    def init_size_constraint(const, t):
        return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
               <= b_tec.var_size * rated_power
    b_tec.const_size = Constraint(set_t, rule=init_size_constraint)

    # Maximum input of carriers
    if 'max_input' in performance_data:
        b_tec.set_max_input_carriers = Set(initialize=performance_data['max_input'].keys())
        def init_max_input(const, t, car):
            return input[t, car] <= performance_data['max_input'][car] * \
                sum(input[t, car_input] for car_input in b_tec.set_input_carriers)
        b_tec.const_max_input = Constraint(set_t, b_tec.set_max_input_carriers, rule=init_max_input)

    return b_tec

def constraints_tec_CONV2(b_tec, tec_data, energyhub):
    """
    Adds constraints to technology blocks for tec_type CONV2, i.e. :math:`output_{car} = f_{car}(\sum(inputs))`

    This technology type resembles a technology with full input substitution, but different performance functions
    for the respective output carriers.
    As for all conversion technologies, three different performance function fits are possible. The performance
    functions are fitted in ``src.model_construction.technology_performance_fitting``.

    **Constraint declarations:**

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

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    model = energyhub.model

    # DATA OF TECHNOLOGY
    performance_data = tec_data.performance_data
    coeff = tec_data.fitted_performance.coefficients
    rated_power = tec_data.fitted_performance.rated_power
    modelled_with_full_res = tec_data.modelled_with_full_res

    # Full or reduced resolution
    if global_variables.clustered_data and not modelled_with_full_res:
        input = b_tec.var_input_aux
        output = b_tec.var_output_aux
        set_t = model.set_t_clustered
    else:
        input = b_tec.var_input
        output = b_tec.var_output
        set_t = model.set_t_full

    performance_function_type = performance_data['performance_function_type']

    alpha1 = {}
    alpha2 = {}
    # Get performance parameters
    for c in performance_data['performance']['out']:
        alpha1[c] = coeff[c]['alpha1']
        if performance_function_type == 2:
            alpha2[c] = coeff[c]['alpha2']
        if performance_function_type == 3:
            bp_x = coeff['bp_x']
            alpha2[c] = coeff[c]['alpha2']

    min_part_load = performance_data['min_part_load']

    if performance_function_type >= 2:
        global_variables.big_m_transformation_required = 1

    # LINEAR, NO MINIMAL PARTLOAD, THROUGH ORIGIN
    if performance_function_type == 1:
        def init_input_output(const, t, car_output):
            return output[t, car_output] == \
                   alpha1[car_output] * sum(input[t, car_input]
                                            for car_input in b_tec.set_input_carriers)
        b_tec.const_input_output = Constraint(set_t, b_tec.set_output_carriers,
                                              rule=init_input_output)

    # LINEAR, MINIMAL PARTLOAD
    elif performance_function_type == 2:
        if min_part_load == 0:
            warnings.warn(
                'Having performance_function_type = 2 with no part-load usually makes no sense. Error occured for ' + b_tec.local_name)

        # define disjuncts
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const, car_input):
                    return input[t, car_input] == 0
                dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)
            else:  # technology on
                # input-output relation
                def init_input_output_on(const, car_output):
                    return output[t, car_output] == \
                           alpha1[car_output] * sum(input[t, car_input] for car_input
                                                    in b_tec.set_input_carriers) \
                           + alpha2[car_output] * b_tec.var_size * rated_power
                dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_input_output_on)

                # min part load relation
                def init_min_partload(const):
                    return sum(input[t, car_input]
                               for car_input in b_tec.set_input_carriers) >= \
                           min_part_load * b_tec.var_size * rated_power
                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(set_t, rule=bind_disjunctions)

    # piecewise affine function
    elif performance_function_type == 3:
        s_indicators = range(0, len(bp_x))

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const, car_input):
                    return input[t, car_input] == 0
                dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # piecewise definition
                def init_input_on1(const):
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) >= \
                           bp_x[ind - 1] * b_tec.var_size * rated_power
                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) <= \
                           bp_x[ind] * b_tec.var_size * rated_power
                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return output[t, car_output] == \
                           alpha1[car_output][ind - 1] * sum(input[t, car_input]
                                                             for car_input in b_tec.set_input_carriers) + \
                           alpha2[car_output][ind - 1] * b_tec.var_size * rated_power
                dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_output_on)

                # min part load relation
                def init_min_partload(const):
                    return sum(input[t, car_input]
                               for car_input in b_tec.set_input_carriers) >= \
                           min_part_load * b_tec.var_size * rated_power
                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(set_t, rule=bind_disjunctions)

    # size constraint based on sum of inputs
    def init_size_constraint(const, t):
        return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
               <= b_tec.var_size * rated_power
    b_tec.const_size = Constraint(set_t, rule=init_size_constraint)

    # Maximum input of carriers
    if 'max_input' in performance_data:
        b_tec.set_max_input_carriers = Set(initialize=performance_data['max_input'].keys())
        def init_max_input(const, t, car):
            return input[t, car] <= performance_data['max_input'][car] * \
                sum(input[t, car_input] for car_input in b_tec.set_input_carriers)
        b_tec.const_max_input = Constraint(set_t, b_tec.set_max_input_carriers, rule=init_max_input)

    return b_tec

def constraints_tec_CONV3(b_tec, tec_data, energyhub):
    """
    Adds constraints to technology blocks for tec_type CONV3, i.e. :math:`output_{car} = f_{car}(input_{maincarrier})`

    This technology type resembles a technology with different performance functions for the respective output
    carriers. The performance function is based on the input of the main carrier.
    The ratio between all input carriers is fixed.
    As for all conversion technologies, three different performance function fits are possible. The performance
    functions are fitted in ``src.model_construction.technology_performance_fitting``.

    **Constraint declarations:**
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

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    model = energyhub.model

    # DATA OF TECHNOLOGY
    performance_data = tec_data.performance_data
    coeff = tec_data.fitted_performance.coefficients
    rated_power = tec_data.fitted_performance.rated_power
    modelled_with_full_res = tec_data.modelled_with_full_res

    # Full or reduced resolution
    if global_variables.clustered_data and not modelled_with_full_res:
        input = b_tec.var_input_aux
        output = b_tec.var_output_aux
        set_t = model.set_t_clustered
    else:
        input = b_tec.var_input
        output = b_tec.var_output
        set_t = model.set_t_full

    performance_function_type = performance_data['performance_function_type']

    alpha1 = {}
    alpha2 = {}
    phi = {}
    # Get performance parameters
    for c in performance_data['performance']['out']:
        alpha1[c] = coeff[c]['alpha1']
        if performance_function_type == 2:
            alpha2[c] = coeff[c]['alpha2']
        if performance_function_type == 3:
            bp_x = coeff['bp_x']
            alpha2[c] = coeff[c]['alpha2']

    min_part_load = performance_data['min_part_load']

    if 'input_ratios' in performance_data:
        main_car = performance_data['main_input_carrier']
        for c in performance_data['input_ratios']:
            phi[c] = performance_data['input_ratios'][c]
    else:
        warnings.warn(
            'Using CONV3 without input ratios makes no sense. Error occured for ' + b_tec.local_name)

    # LINEAR, NO MINIMAL PARTLOAD, THROUGH ORIGIN
    if performance_function_type == 1:
        def init_input_output(const, t, car_output):
            return output[t, car_output] == \
                   alpha1[car_output] * input[t, main_car]

        b_tec.const_input_output = Constraint(set_t, b_tec.set_output_carriers,
                                              rule=init_input_output)

    # LINEAR, MINIMAL PARTLOAD
    elif performance_function_type == 2:
        global_variables.big_m_transformation_required = 1
        if min_part_load == 0:
            warnings.warn(
                'Having performance_function_type = 2 with no part-load usually makes no sense.')

        # define disjuncts
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const, car_input):
                    return input[t, car_input] == 0
                dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)
            else:  # technology on
                # input-output relation
                def init_input_output_on(const, car_output):
                    return output[t, car_output] == \
                           alpha1[car_output] * input[t, main_car] + \
                           alpha2[car_output] * b_tec.var_size * rated_power
                dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_input_output_on)

                # min part load relation
                def init_min_partload(const):
                    return input[t, main_car] >= min_part_load * b_tec.var_size * rated_power
                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(set_t, rule=bind_disjunctions)

    # piecewise affine function
    elif performance_function_type == 3:
        global_variables.big_m_transformation_required = 1
        s_indicators = range(0, len(bp_x))

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const, car_input):
                    return input[t, car_input] == 0
                dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # piecewise definition
                def init_input_on1(const):
                    return input[t, main_car] >= bp_x[ind - 1] * b_tec.var_size * rated_power
                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return input[t, main_car] <= bp_x[ind] * b_tec.var_size * rated_power
                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return output[t, car_output] == \
                           alpha1[car_output][ind - 1] * input[t, main_car] + \
                           alpha2[car_output][ind - 1] * b_tec.var_size * rated_power
                dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_output_on)

                # min part load relation
                def init_min_partload(const):
                    return input[t, main_car] >= min_part_load * b_tec.var_size * rated_power
                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(set_t, rule=bind_disjunctions)

    # constraint on input ratios
    def init_input_input(const, t, car_input):
        if car_input == main_car:
            return Constraint.Skip
        else:
            return input[t, car_input] == phi[car_input] * input[t, main_car]
    b_tec.const_input_input = Constraint(set_t, b_tec.set_input_carriers, rule=init_input_input)

    # size constraint based main carrier input
    def init_size_constraint(const, t):
        return input[t, main_car] <= b_tec.var_size * rated_power
    b_tec.const_size = Constraint(set_t, rule=init_size_constraint)

    return b_tec

def constraints_tec_STOR(b_tec, tec_data, energyhub):
    """
    Adds constraints to technology blocks for tec_type STOR, resembling a storage technology

    As for all conversion technologies, three different performance function fits are possible. The performance
    functions are fitted in ``src.model_construction.technology_performance_fitting``.
    Note that this technology only works for one carrier, and thus the carrier index is dropped in the below notation.

    **Parameter declarations:**

    - :math:`{\\eta}_{in}`: Charging efficiency

    - :math:`{\\eta}_{out}`: Discharging efficiency

    - :math:`{\\lambda_1}`: Self-Discharging coefficient (independent of environment)

    - :math:`{\\lambda_2(\\Theta)}`: Self-Discharging coefficient (dependent on environment)

    - :math:`Input_{max}`: Maximal charging capacity in one time-slice

    - :math:`Output_{max}`: Maximal discharging capacity in one time-slice

    **Variable declarations:**

    - Storage level in :math:`t`: :math:`E_t`

    - Charging in in :math:`t`: :math:`Input_{t}`

    - Discharging in in :math:`t`: :math:`Output_{t}`

    **Constraint declarations:**

    - Maximal charging and discharging:

      .. math::
        Input_{t} \leq Input_{max}

      .. math::
        Output_{t} \leq Output_{max}

    - Size constraint:

      .. math::
        E_{t} \leq S

    - Storage level calculation:

      .. math::
        E_{t} = E_{t-1} * (1 - \\lambda_1) - \\lambda_2(\\Theta) * E_{t-1} + {\\eta}_{in} * Input_{t} - 1 / {\\eta}_{out} * Output_{t}

    - If ``allow_only_one_direction == 1``, then only input or output can be unequal to zero in each respective time
      step (otherwise, simultanous charging and discharging can lead to unwanted 'waste' of energy/material).

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    model = energyhub.model

    # DATA OF TECHNOLOGY
    performance_data = tec_data.performance_data
    coeff = tec_data.fitted_performance.coefficients
    modelled_with_full_res = tec_data.modelled_with_full_res

    # Full resolution
    input = b_tec.var_input
    output = b_tec.var_output
    set_t = model.set_t_full

    if 'allow_only_one_direction' in performance_data:
        allow_only_one_direction = performance_data['allow_only_one_direction']
    else:
        allow_only_one_direction = 0

    nr_timesteps_averaged = global_variables.averaged_data_specs.nr_timesteps_averaged

    # Additional decision variables
    b_tec.var_storage_level = Var(set_t, b_tec.set_input_carriers,
                                  domain=NonNegativeReals,
                                  bounds=(b_tec.para_size_min, b_tec.para_size_max))

    # Abdditional parameters
    eta_in = coeff['eta_in']
    eta_out = coeff['eta_out']
    eta_lambda = coeff['lambda']
    charge_max = coeff['charge_max']
    discharge_max = coeff['discharge_max']
    ambient_loss_factor = coeff['ambient_loss_factor']

    # Size constraint
    def init_size_constraint(const, t, car):
        return b_tec.var_storage_level[t, car] <= b_tec.var_size
    b_tec.const_size = Constraint(set_t, b_tec.set_input_carriers, rule=init_size_constraint)

    # Storage level calculation
    def init_storage_level(const, t, car):
        if t == 1: # couple first and last time interval
            return b_tec.var_storage_level[t, car] == \
                  b_tec.var_storage_level[max(set_t), car] * (1 - eta_lambda) ** nr_timesteps_averaged - \
                  b_tec.var_storage_level[max(set_t), car] * ambient_loss_factor[max(set_t)-1] ** nr_timesteps_averaged + \
                  (eta_in * input[t, car] - 1 / eta_out * output[t, car]) * \
                  sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged))
        else: # all other time intervalls
            return b_tec.var_storage_level[t, car] == \
                b_tec.var_storage_level[t-1, car] * (1 - eta_lambda) ** nr_timesteps_averaged - \
                ambient_loss_factor[t-1] * ambient_loss_factor[max(set_t)-1] ** nr_timesteps_averaged + \
                (eta_in * input[t, car] - 1/eta_out * output[t, car]) * \
                sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged))
    b_tec.const_storage_level = Constraint(set_t, b_tec.set_input_carriers, rule=init_storage_level)

    # This makes sure that only either input or output is larger zero.
    if allow_only_one_direction == 1:
        global_variables.big_m_transformation_required = 1
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # input only
                def init_output_to_zero(const, car_input):
                    return output[t, car_input] == 0
                dis.const_output_to_zero = Constraint(b_tec.set_input_carriers, rule=init_output_to_zero)

            elif ind == 1:  # output only
                def init_input_to_zero(const, car_input):
                    return input[t, car_input] == 0
                dis.const_input_to_zero = Constraint(b_tec.set_input_carriers, rule=init_input_to_zero)

        b_tec.dis_input_output = Disjunct(set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(set_t, rule=bind_disjunctions)

    # Maximal charging and discharging rates
    def init_maximal_charge(const,t,car):
        return input[t, car] <= charge_max * b_tec.var_size
    b_tec.const_max_charge = Constraint(set_t, b_tec.set_input_carriers, rule=init_maximal_charge)

    def init_maximal_discharge(const,t,car):
        return output[t, car] <= discharge_max * b_tec.var_size
    b_tec.const_max_discharge = Constraint(set_t, b_tec.set_input_carriers, rule=init_maximal_discharge)

    return b_tec