from pyomo.environ import *
from pyomo.gdp import *
import warnings
import src.global_variables as global_variables
from src.model_construction.technology_constraints.dynamics import *


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

    if 'curtailment' in performance_data:
        curtailment = performance_data['curtailment']
    else:
        curtailment = 0

    # Full or reduced resolution
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
    Three different performance function fits are possible. The performance
    functions are fitted in ``src.model_construction.technology_performance_fitting``.

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

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    model = energyhub.model
    configuration = energyhub.configuration

    # DATA OF TECHNOLOGY
    performance_data = tec_data.performance_data
    coeff = tec_data.fitted_performance.coefficients
    rated_power = tec_data.fitted_performance.rated_power
    size_based_on = performance_data['size_based_on']

    # Full or reduced resolution
    input = b_tec.var_input
    output = b_tec.var_output
    set_t = model.set_t_full

    performance_function_type = performance_data['performance_function_type']

    # Get performance parameters
    alpha1 = coeff['out']['alpha1']
    if performance_function_type == 2:
        alpha2 = coeff['out']['alpha2']
    if performance_function_type == 3:
        bp_x = coeff['out']['bp_x']
        alpha2 = coeff['out']['alpha2']
    if performance_function_type == 4:
        bp_x = coeff['out']['bp_x']
        alpha2 = coeff['out']['alpha2']

    min_part_load = performance_data['min_part_load']
    ramping_rate = tec_data.performance_data['ramping_rate']
    standby_power = tec_data.performance_data['standby_power']
    SU_load = tec_data.performance_data['SU_load']
    SD_load = tec_data.performance_data['SD_load']
    SU_time = tec_data.performance_data['SU_time']
    SD_time = tec_data.performance_data['SD_time']
    max_startups = tec_data.performance_data['max_startups']
    main_car = performance_data['main_input_carrier']

    # Add integers, bigM and the dynamics constraints
    if performance_function_type >= 2:
        global_variables.big_m_transformation_required = 1
        b_tec.var_x = Var(set_t, domain=NonNegativeReals, bounds=(0, 1))

        # global dynamics switch
        if configuration.performance.dynamics == 1:
            if performance_function_type == 4 or max_startups > -1 or SU_load + SD_load < 2:
                b_tec = constraints_SUSD_logic(b_tec, tec_data, energyhub)
            if not performance_function_type == 4 and SU_load + SD_load < 2:
                b_tec = constraints_fast_SUSD_dynamics(b_tec, tec_data, energyhub)

        else:
            if performance_function_type == 4:
                performance_function_type = 3
                warnings.warn(
                    'Switching dynamics off for performance function type 4, type changed to 3 for ' + b_tec.local_name)

    # LINEAR, NO MINIMAL PARTLOAD, THROUGH ORIGIN
    if performance_function_type == 1:
        def init_input_output(const, t):
            return sum(output[t, car_output]
                       for car_output in b_tec.set_output_carriers) == \
                   alpha1 * sum(input[t, car_input]
                                for car_input in b_tec.set_input_carriers)
        b_tec.const_input_output = Constraint(set_t, rule=init_input_output)

        if min_part_load > 0:
            def init_min_part_load(const, t):
                return min_part_load * b_tec.var_size * rated_power <= \
                       sum(input[t, car_input] for car_input in b_tec.set_input_carriers)
            b_tec.const_min_part_load = Constraint(set_t, rule=init_min_part_load)

    # LINEAR, MINIMAL PARTLOAD
    elif performance_function_type == 2:
        if min_part_load == 0:
            warnings.warn(
                'Having performance_function_type = 2 with no part-load usually makes no sense. Error occured for ' + b_tec.local_name)

        # define disjuncts for on/off
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                if standby_power == -1:
                    def init_input_off(const, car_input):
                        return input[t, car_input] == 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)
                else:
                    def init_input_off(const, car_input):
                        return input[t, car_input] >= 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                    def init_standby_power(const):
                        return input[t, main_car] == standby_power * b_tec.var_size * rated_power
                    dis.const_standby_power = Constraint(rule=init_standby_power)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # technology on
                dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

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
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                if standby_power == -1:
                    def init_input_off(const, car_input):
                        return input[t, car_input] == 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)
                else:
                    def init_input_off(const, car_input):
                        return input[t, car_input] >= 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                    def init_standby_power(const):
                        return input[t, main_car] == standby_power * b_tec.var_size * rated_power
                    dis.const_standby_power = Constraint(rule=init_standby_power)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # piecewise definition
                dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

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

        # slow startup and shutdown dynamics
    elif performance_function_type == 4:
        if SU_time <= 0 and SD_time <= 0:
            warnings.warn('Having performance_function_type = 4 with no slow SU/SDs usually makes no sense.')
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
                    if t < len(set_t) - SU_time or i > SU_time - (len(set_t) - t):
                        return b_tec.var_y[t - i + SU_time + 1] == 0
                    else:
                        return b_tec.var_y[(t - i + SU_time + 1) - len(set_t)] == 0

                dis.const_y_off = Constraint(range(1, SU_time + 1), rule=init_y_off)

                def init_z_off(const, j):
                    if j <= t:
                        return b_tec.var_z[t - j + 1] == 0
                    else:
                        return b_tec.var_z[len(set_t) + (t - j + 1)] == 0

                dis.const_z_off = Constraint(range(1, SD_time + 1), rule=init_z_off)

                def init_input_off(const, car_input):
                    return input[t, car_input] == 0

                dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0

                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            elif ind in range(1, SU_time + 1):  # technology in startup
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                def init_y_on(const):
                    if t < len(set_t) - SU_time or ind > SU_time - (len(set_t) - t):
                        return b_tec.var_y[t - ind + SU_time + 1] == 1
                    else:
                        return b_tec.var_y[(t - ind + SU_time + 1) - len(set_t)] == 1

                dis.const_y_on = Constraint(rule=init_y_on)

                def init_z_off(const):
                    if t < len(set_t) - SU_time or ind > SU_time - (len(set_t) - t):
                        return b_tec.var_z[t - ind + SU_time + 1] == 0
                    else:
                        return b_tec.var_z[(t - ind + SU_time + 1) - len(set_t)] == 0

                dis.const_z_off = Constraint(rule=init_z_off)

                def init_input_SU(cons):
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
                           == b_tec.var_size * SU_trajectory[ind - 1]

                dis.const_input_SU = Constraint(rule=init_input_SU)

                def init_output_SU(const):
                    return sum(output[t, car_output] for car_output in b_tec.set_output_carriers) == \
                           alpha1[0] * sum(input[t, car_input] for car_input in b_tec.set_input_carriers) + \
                           alpha2[0] * b_tec.var_size * rated_power

                dis.const_output_SU = Constraint(rule=init_output_SU)

            elif ind in range(SU_time + 1, SU_time + SD_time + 1):  # technology in shutdown
                ind_SD = ind - SU_time
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                def init_z_on(const):
                    if ind_SD <= t:
                        return b_tec.var_z[t - ind_SD + 1] == 1
                    else:
                        return b_tec.var_z[len(set_t) + (t - ind_SD + 1)] == 1

                dis.const_z_on = Constraint(rule=init_z_on)

                def init_y_off(const):
                    if ind_SD <= t:
                        return b_tec.var_y[t - ind_SD + 1] == 0
                    else:
                        return b_tec.var_y[len(set_t) + (t - ind_SD + 1)] == 0

                dis.const_y_off = Constraint(rule=init_y_off)

                def init_input_SD(cons):
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
                           == b_tec.var_size * SD_trajectory[ind_SD - 1]

                dis.const_input_SD = Constraint(rule=init_input_SD)

                def init_output_SD(const):
                    return sum(output[t, car_output] for car_output in b_tec.set_output_carriers) == \
                           alpha1[0] * sum(input[t, car_input] for car_input in b_tec.set_input_carriers) + \
                           alpha2[0] * b_tec.var_size * rated_power

                dis.const_output_SD = Constraint(rule=init_output_SD)

            elif ind > SU_time + SD_time:
                ind_bpx = ind - (SU_time + SD_time)
                dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

                def init_input_on1(const):
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) >= \
                           bp_x[ind_bpx - 1] * b_tec.var_size * rated_power

                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) <= \
                           bp_x[ind_bpx] * b_tec.var_size * rated_power

                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const):
                    return sum(output[t, car_output] for car_output in b_tec.set_output_carriers) == \
                           alpha1[ind_bpx - 1] * sum(input[t, car_input] for car_input in b_tec.set_input_carriers) + \
                           alpha2[ind_bpx - 1] * b_tec.var_size * rated_power

                dis.const_input_output_on = Constraint(rule=init_output_on)

                # min part load relation
                def init_min_partload(const):
                    return sum(input[t, car_input]
                               for car_input in b_tec.set_input_carriers) >= \
                           min_part_load * b_tec.var_size * rated_power

                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_SUSD_trajectory = Disjunct(set_t, s_indicators, rule=init_SUSD_trajectories)

        def bind_disjunctions_SUSD(dis, t):
            return [b_tec.dis_SUSD_trajectory[t, k] for k in s_indicators]

        b_tec.disjunction_SUSD_traject = Disjunction(set_t, rule=bind_disjunctions_SUSD)

    # add ramping rates
    if not ramping_rate == 0:
        def init_ramping_down_rate(const, t):
            if t > 1:
                return -ramping_rate <= sum(input[t, car_input] - input[t-1, car_input]
                                                for car_input in b_tec.set_input_carriers)
            else:
                return Constraint.Skip
        b_tec.const_ramping_down_rate = Constraint(set_t, rule=init_ramping_down_rate)

        def init_ramping_up_rate(const, t):
            if t > 1:
                return sum(input[t, car_input] - input[t-1, car_input]
                               for car_input in b_tec.set_input_carriers) <= ramping_rate
            else:
                return Constraint.Skip
        b_tec.const_ramping_up_rate = Constraint(set_t, rule=init_ramping_up_rate)

    # size constraint based on sum of input/output
    def init_size_constraint(const, t):
        if size_based_on == 'input':
            return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
                   <= b_tec.var_size * rated_power
        elif size_based_on == 'output':
            return sum(output[t, car_output] for car_output in b_tec.set_output_carriers) \
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
    Three different performance function fits are possible. The performance
    functions are fitted in ``src.model_construction.technology_performance_fitting``.

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

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    model = energyhub.model
    configuration = energyhub.configuration

    # DATA OF TECHNOLOGY
    performance_data = tec_data.performance_data
    coeff = tec_data.fitted_performance.coefficients
    rated_power = tec_data.fitted_performance.rated_power

    # Full or reduced resolution
    input = b_tec.var_input
    output = b_tec.var_output
    set_t = model.set_t_full

    performance_function_type = performance_data['performance_function_type']

    alpha1 = {}
    alpha2 = {}
    # Get performance parameters
    for car in performance_data['performance']['out']:
        alpha1[car] = coeff[car]['alpha1']
        if performance_function_type == 2:
            alpha2[car] = coeff[car]['alpha2']
        if performance_function_type == 3:
            bp_x = coeff[car]['bp_x']
            alpha2[car] = coeff[car]['alpha2']
        if performance_function_type == 4:
            bp_x = coeff[car]['bp_x']
            alpha2[car] = coeff[car]['alpha2']

    min_part_load = performance_data['min_part_load']
    ramping_rate = tec_data.performance_data['ramping_rate']
    standby_power = tec_data.performance_data['standby_power']
    SU_load = tec_data.performance_data['SU_load']
    SD_load = tec_data.performance_data['SD_load']
    SU_time = tec_data.performance_data['SU_time']
    SD_time = tec_data.performance_data['SD_time']
    max_startups = tec_data.performance_data['max_startups']
    main_car = performance_data['main_input_carrier']

    # Add integers, bigM and the dynamics constraints
    if performance_function_type >= 2:
        global_variables.big_m_transformation_required = 1
        b_tec.var_x = Var(set_t, domain=NonNegativeReals, bounds=(0, 1))

        # global dynamics switch
        if configuration.performance.dynamics == 1:
            if performance_function_type == 4 or max_startups > -1 or SU_load + SD_load < 2:
                b_tec = constraints_SUSD_logic(b_tec, tec_data, energyhub)
            if not performance_function_type == 4 and SU_load + SD_load < 2:
                b_tec = constraints_fast_SUSD_dynamics(b_tec, tec_data, energyhub)

        else:
            if performance_function_type == 4:
                performance_function_type = 3
                warnings.warn(
                    'Switching dynamics off for performance function type 4, type changed to 3 for ' + b_tec.local_name)

    # LINEAR, NO MINIMAL PARTLOAD, THROUGH ORIGIN
    if performance_function_type == 1:
        def init_input_output(const, t, car_output):
            return output[t, car_output] == \
                   alpha1[car_output] * sum(input[t, car_input]
                                            for car_input in b_tec.set_input_carriers)
        b_tec.const_input_output = Constraint(set_t, b_tec.set_output_carriers,
                                              rule=init_input_output)

        if min_part_load > 0:
            def init_min_part_load(const, t):
                return min_part_load * b_tec.var_size * rated_power <= \
                       sum(input[t, car_input] for car_input in b_tec.set_input_carriers)
            b_tec.const_min_part_load = Constraint(set_t, rule=init_min_part_load)

    # LINEAR, MINIMAL PARTLOAD
    elif performance_function_type == 2:
        if min_part_load == 0:
            warnings.warn(
                'Having performance_function_type = 2 with no part-load usually makes no sense. Error occured for ' + b_tec.local_name)

        # define disjuncts
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                if standby_power == -1:
                    def init_input_off(const, car_input):
                        return input[t, car_input] == 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)
                else:
                    def init_input_off(const, car_input):
                        return input[t, car_input] >= 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                    def init_standby_power(const):
                        return input[t, main_car] == standby_power * b_tec.var_size * rated_power
                    dis.const_standby_power = Constraint(rule=init_standby_power)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)
            else:  # technology on
                dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

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
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                if standby_power == -1:
                    def init_input_off(const, car_input):
                        return input[t, car_input] == 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)
                else:
                    def init_input_off(const, car_input):
                        return input[t, car_input] >= 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                    def init_standby_power(const):
                        return input[t, main_car] == standby_power * b_tec.var_size * rated_power
                    dis.const_standby_power = Constraint(rule=init_standby_power)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # piecewise definition
                dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

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

        # slow startup and shutdown dynamics
    elif performance_function_type == 4:
        if SU_time + SD_time == 0:
            warnings.warn(
                'Having performance_function_type = 4 with no slow SU/SDs usually makes no sense.')

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
                    if t < len(set_t) - SU_time or i > SU_time - (len(set_t) - t):
                        return b_tec.var_y[t - i + SU_time + 1] == 0
                    else:
                        return b_tec.var_y[(t - i + SU_time + 1) - len(set_t)] == 0

                dis.const_y_off = Constraint(range(1, SU_time + 1), rule=init_y_off)

                def init_z_off(const, j):
                    if j <= t:
                        return b_tec.var_z[t - j + 1] == 0
                    else:
                        return b_tec.var_z[len(set_t) + (t - j + 1)] == 0

                dis.const_z_off = Constraint(range(1, SD_time + 1), rule=init_z_off)

                def init_input_off(const, car_input):
                    return input[t, car_input] == 0

                dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0

                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            elif ind in range(1, SU_time + 1):  # technology in startup
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                def init_y_on(const):
                    if t < len(set_t) - SU_time or ind > SU_time - (len(set_t) - t):
                        return b_tec.var_y[t - ind + SU_time + 1] == 1
                    else:
                        return b_tec.var_y[(t - ind + SU_time + 1) - len(set_t)] == 1

                dis.const_y_on = Constraint(rule=init_y_on)

                def init_z_off(const):
                    if t < len(set_t) - SU_time or ind > SU_time - (len(set_t) - t):
                        return b_tec.var_z[t - ind + SU_time + 1] == 0
                    else:
                        return b_tec.var_z[(t - ind + SU_time + 1) - len(set_t)] == 0

                dis.const_z_off = Constraint(rule=init_z_off)

                def init_input_SU(cons):
                    return  sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
                           == b_tec.var_size * SU_trajectory[ind - 1]

                dis.const_input_SU = Constraint(rule=init_input_SU)

                def init_output_SU(const, car_output):
                    return output[t, car_output] == \
                           alpha1[car_output][0] * sum(input[t, car_input] for car_input in b_tec.set_input_carriers) + \
                           alpha2[car_output][0] * b_tec.var_size * rated_power

                dis.const_output_SU = Constraint(b_tec.set_output_carriers, rule=init_output_SU)

            elif ind in range(SU_time + 1, SU_time + SD_time + 1):  # technology in shutdown
                ind_SD = ind - SU_time
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                def init_z_on(const):
                    if ind_SD <= t:
                        return b_tec.var_z[t - ind_SD + 1] == 1
                    else:
                        return b_tec.var_z[len(set_t) + (t - ind_SD + 1)] == 1

                dis.const_z_on = Constraint(rule=init_z_on)

                def init_y_off(const):
                    if ind_SD <= t:
                        return b_tec.var_y[t - ind_SD + 1] == 0
                    else:
                        return b_tec.var_y[len(set_t) + (t - ind_SD + 1)] == 0

                dis.const_y_off = Constraint(rule=init_y_off)

                def init_input_SD(cons):
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
                           == b_tec.var_size * SD_trajectory[ind_SD - 1]

                dis.const_input_SD = Constraint(rule=init_input_SD)

                def init_output_SD(const, car_output):
                    return output[t, car_output] == \
                           alpha1[car_output][0] * sum(input[t, car_input] for car_input in b_tec.set_input_carriers) + \
                           alpha2[car_output][0] * b_tec.var_size * rated_power

                dis.const_output_SD = Constraint(b_tec.set_output_carriers, rule=init_output_SD)

            elif ind > SU_time + SD_time:
                ind_bpx = ind - (SU_time + SD_time)
                dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

                def init_input_on1(const):
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
                           >= bp_x[ind_bpx - 1] * b_tec.var_size * rated_power

                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
                           <= bp_x[ind_bpx] * b_tec.var_size * rated_power

                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return output[t, car_output] == \
                           alpha1[car_output][ind_bpx - 1] * sum(input[t, car_input] for car_input in b_tec.set_input_carriers) + \
                           alpha2[car_output][ind_bpx - 1] * b_tec.var_size * rated_power

                dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_output_on)

                # min part load relation
                def init_min_partload(const):
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
                           >= min_part_load * b_tec.var_size * rated_power

                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_SUSD_trajectory = Disjunct(set_t, s_indicators, rule=init_SUSD_trajectories)

        def bind_disjunctions_SUSD(dis, t):
            return [b_tec.dis_SUSD_trajectory[t, k] for k in s_indicators]

        b_tec.disjunction_SUSD_traject = Disjunction(set_t, rule=bind_disjunctions_SUSD)

    # add ramping rates
    if not ramping_rate == 0:
        def init_ramping_down_rate(const, t):
            if t > 1:
                return -ramping_rate <= sum(input[t, car_input] - input[t - 1, car_input]
                                            for car_input in b_tec.set_input_carriers)
            else:
                return Constraint.Skip
        b_tec.const_ramping_down_rate = Constraint(set_t, rule=init_ramping_down_rate)

        def init_ramping_up_rate(const, t):
            if t > 1:
                return sum(input[t, car_input] - input[t - 1, car_input]
                           for car_input in b_tec.set_input_carriers) <= ramping_rate
            else:
                return Constraint.Skip
        b_tec.const_ramping_up_rate = Constraint(set_t, rule=init_ramping_up_rate)

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
    Three different performance function fits are possible. The performance
    functions are fitted in ``src.model_construction.technology_performance_fitting``.

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

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    model = energyhub.model
    configuration = energyhub.configuration

    # DATA OF TECHNOLOGY
    performance_data = tec_data.performance_data
    coeff = tec_data.fitted_performance.coefficients
    rated_power = tec_data.fitted_performance.rated_power

    # Full or reduced resolution
    input = b_tec.var_input
    output = b_tec.var_output
    set_t = model.set_t_full

    performance_function_type = performance_data['performance_function_type']

    alpha1 = {}
    alpha2 = {}
    phi = {}
    # Get performance parameters
    for car in performance_data['performance']['out']:
        alpha1[car] = coeff[car]['alpha1']
        if performance_function_type == 2:
            alpha2[car] = coeff[car]['alpha2']
        if performance_function_type == 3:
            bp_x = coeff[car]['bp_x']
            alpha2[car] = coeff[car]['alpha2']
        if performance_function_type == 4:
            bp_x = coeff[car]['bp_x']
            alpha2[car] = coeff[car]['alpha2']

    min_part_load = performance_data['min_part_load']
    ramping_rate = tec_data.performance_data['ramping_rate']
    standby_power = tec_data.performance_data['standby_power']
    SU_load = tec_data.performance_data['SU_load']
    SD_load = tec_data.performance_data['SD_load']
    SU_time = tec_data.performance_data['SU_time']
    SD_time = tec_data.performance_data['SD_time']
    max_startups = tec_data.performance_data['max_startups']
    main_car = performance_data['main_input_carrier']

    if 'input_ratios' in performance_data:
        main_car = performance_data['main_input_carrier']
        for car in performance_data['input_ratios']:
            phi[car] = performance_data['input_ratios'][car]
    else:
        warnings.warn(
            'Using CONV3 without input ratios makes no sense. Error occured for ' + b_tec.local_name)

    # Add integers, bigM and the dynamics constraints
    if performance_function_type >= 2:
        global_variables.big_m_transformation_required = 1
        b_tec.var_x = Var(set_t, domain=NonNegativeReals, bounds=(0, 1))

        # global dynamics switch
        if configuration.performance.dynamics == 1:
            if performance_function_type == 4 or max_startups > -1 or SU_load + SD_load < 2:
                b_tec = constraints_SUSD_logic(b_tec, tec_data, energyhub)
            if not performance_function_type == 4 and SU_load + SD_load < 2:
                b_tec = constraints_fast_SUSD_dynamics(b_tec, tec_data, energyhub)

        else:
            if performance_function_type == 4:
                performance_function_type = 3
                warnings.warn(
                    'Switching dynamics off for performance function type 4, type changed to 3 for ' + b_tec.local_name)

    # LINEAR, NO MINIMAL PARTLOAD, THROUGH ORIGIN
    if performance_function_type == 1:
        def init_input_output(const, t, car_output):
            return output[t, car_output] == \
                   alpha1[car_output] * input[t, main_car]
        b_tec.const_input_output = Constraint(set_t, b_tec.set_output_carriers,
                                              rule=init_input_output)

        if min_part_load > 0:
            def init_min_part_load(const, t):
                return min_part_load * b_tec.var_size * rated_power <= input[t, main_car]
            b_tec.const_min_part_load = Constraint(set_t, rule=init_min_part_load)

    # LINEAR, MINIMAL PARTLOAD
    elif performance_function_type == 2:
        if min_part_load == 0:
            warnings.warn(
                'Having performance_function_type = 2 with no part-load usually makes no sense.')

        # define disjuncts
        s_indicators = range(0, 2)
        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                if not standby_power == -1:
                    def init_input_off(const, car_input):
                        return input[t, car_input] >= 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                    def init_standby_power(const):
                        return input[t, main_car] == standby_power * b_tec.var_size * rated_power
                    dis.const_standby_power = Constraint(rule=init_standby_power)
                else:
                    def init_input_off(const, car_input):
                        return input[t, car_input] == 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # technology on
                dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

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
        s_indicators = range(0, len(bp_x))

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                if standby_power == -1:
                    def init_input_off(const, car_input):
                        return input[t, car_input] == 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)
                else:
                    def init_input_off(const, car_input):
                        return input[t, car_input] >= 0
                    dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                    def init_standby_power(const):
                        return input[t, main_car] == standby_power * b_tec.var_size * rated_power
                    dis.const_standby_power = Constraint(rule=init_standby_power)


                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # piecewise definition
                dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

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

    # slow startup and shutdown dynamics
    elif performance_function_type == 4:
        if SU_time + SD_time == 0:
            warnings.warn(
                'Having performance_function_type = 4 with no slow SU/SDs usually makes no sense.')

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
                    if t < len(set_t) - SU_time or i > SU_time - (len(set_t) - t):
                        return b_tec.var_y[t - i + SU_time + 1] == 0
                    else:
                        return b_tec.var_y[(t - i + SU_time + 1) - len(set_t)] == 0

                dis.const_y_off = Constraint(range(1, SU_time + 1), rule=init_y_off)

                def init_z_off(const, j):
                    if j <= t:
                        return b_tec.var_z[t - j + 1] == 0
                    else:
                        return b_tec.var_z[len(set_t) + (t - j + 1)] == 0

                dis.const_z_off = Constraint(range(1, SD_time + 1), rule=init_z_off)

                def init_input_off(const, car_input):
                    return input[t, car_input] == 0

                dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                def init_output_off(const, car_output):
                    return output[t, car_output] == 0

                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            elif ind in range(1, SU_time + 1):  # technology in startup
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                def init_y_on(const):
                    if t < len(set_t) - SU_time or ind > SU_time - (len(set_t) - t):
                        return b_tec.var_y[t - ind + SU_time + 1] == 1
                    else:
                        return b_tec.var_y[(t - ind + SU_time + 1) - len(set_t)] == 1

                dis.const_y_on = Constraint(rule=init_y_on)

                def init_z_off(const):
                    if t < len(set_t) - SU_time or ind > SU_time - (len(set_t) - t):
                        return b_tec.var_z[t - ind + SU_time + 1] == 0
                    else:
                        return b_tec.var_z[(t - ind + SU_time + 1) - len(set_t)] == 0

                dis.const_z_off = Constraint(rule=init_z_off)

                def init_input_SU(cons):
                    return input[t, main_car] == b_tec.var_size * SU_trajectory[ind - 1]

                dis.const_input_SU = Constraint(rule=init_input_SU)

                def init_output_SU(const, car_output):
                    return output[t, car_output] == alpha1[car_output][0] * input[t, main_car] + \
                           alpha2[car_output][0] * b_tec.var_size * rated_power

                dis.const_output_SU = Constraint(b_tec.set_output_carriers, rule=init_output_SU)

            elif ind in range(SU_time + 1, SU_time + SD_time + 1):  # technology in shutdown
                ind_SD = ind - SU_time
                dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                def init_z_on(const):
                    if ind_SD <= t:
                        return b_tec.var_z[t - ind_SD + 1] == 1
                    else:
                        return b_tec.var_z[len(set_t) + (t - ind_SD + 1)] == 1

                dis.const_z_on = Constraint(rule=init_z_on)

                def init_y_off(const):
                    if ind_SD <= t:
                        return b_tec.var_y[t - ind_SD + 1] == 0
                    else:
                        return b_tec.var_y[len(set_t) + (t - ind_SD + 1)] == 0

                dis.const_y_off = Constraint(rule=init_y_off)

                def init_input_SD(cons):
                    return input[t, main_car] == b_tec.var_size * SD_trajectory[ind_SD - 1]

                dis.const_input_SD = Constraint(rule=init_input_SD)

                def init_output_SD(const, car_output):
                    return output[t, car_output] == alpha1[car_output][0] * input[t, main_car] + \
                           alpha2[car_output][0] * b_tec.var_size * rated_power

                dis.const_output_SD = Constraint(b_tec.set_output_carriers, rule=init_output_SD)

            elif ind > SU_time + SD_time:
                ind_bpx = ind - (SU_time + SD_time)
                dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

                def init_input_on1(const):
                    return input[t, main_car] >= bp_x[ind_bpx - 1] * b_tec.var_size * rated_power
                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return input[t, main_car] <= bp_x[ind_bpx] * b_tec.var_size * rated_power
                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const, car_output):
                    return output[t, car_output] == \
                           alpha1[car_output][ind_bpx - 1] * input[t, main_car] + \
                           alpha2[car_output][ind_bpx - 1] * b_tec.var_size * rated_power
                dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_output_on)

                # min part load relation
                def init_min_partload(const):
                    return input[t, main_car] >= min_part_load * b_tec.var_size * rated_power

                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_SUSD_trajectory = Disjunct(set_t, s_indicators, rule=init_SUSD_trajectories)

        def bind_disjunctions_SUSD(dis, t):
            return [b_tec.dis_SUSD_trajectory[t, k] for k in s_indicators]

        b_tec.disjunction_SUSD_traject = Disjunction(set_t, rule=bind_disjunctions_SUSD)

    # add ramping rates
    if not ramping_rate == 0:
        def init_ramping_down_rate(const, t):
            if t > 1:
                return -ramping_rate <= input[t, main_car] - input[t - 1, main_car]
            else:
                return Constraint.Skip
        b_tec.const_ramping_down_rate = Constraint(set_t, rule=init_ramping_down_rate)

        def init_ramping_up_rate(const, t):
            if t > 1:
                return input[t, main_car] - input[t - 1, main_car] <= ramping_rate
            else:
                return Constraint.Skip
        b_tec.const_ramping_up_rate = Constraint(set_t, rule=init_ramping_up_rate)

    # constraint on input ratios
    def init_input_input(const, t, car_input):
        if car_input == main_car:
            return Constraint.Skip
        else:
            if standby_power == -1:
                return input[t, car_input] == phi[car_input] * input[t, main_car]
            else:
                return input[t, car_input] == phi[car_input] * input[t, main_car] * b_tec.var_x[t]
    b_tec.const_input_input = Constraint(set_t, b_tec.set_input_carriers, rule=init_input_input)

    # size constraint based main carrier input
    def init_size_constraint(const, t):
        return input[t, main_car] <= b_tec.var_size * rated_power
    b_tec.const_size = Constraint(set_t, rule=init_size_constraint)

    return b_tec



def constraints_tec_CONV4(b_tec, tec_data, energyhub):
    """
    Adds constraints to technology blocks for tec_type CONV4, i.e. :math:`output_{car} \leq S>)`

    This technology type resembles a technology with fixed output ratios and no inputs
    Two different performance function fits are possible. The performance
    functions are fitted in ``src.model_construction.technology_performance_fitting``.

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

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    model = energyhub.model

    # DATA OF TECHNOLOGY
    performance_data = tec_data.performance_data
    rated_power = tec_data.fitted_performance.rated_power

    # Full or reduced resolution
    output = b_tec.var_output
    set_t = model.set_t_full

    performance_function_type = performance_data['performance_function_type']


    min_part_load = performance_data['min_part_load']

    # Output ratios
    phi = {}
    main_car = performance_data['main_output_carrier']
    for car in performance_data['output_ratios']:
        phi[car] = performance_data['output_ratios'][car]

    # LINEAR, NO MINIMAL PARTLOAD, THROUGH ORIGIN
    if performance_function_type == 1:
        pass

    # LINEAR, MINIMAL PARTLOAD
    elif performance_function_type == 2:
        global_variables.big_m_transformation_required = 1

        if min_part_load == 0:
            warnings.warn(
                'Having performance_function_type = 2 with no part-load usually makes no sense.')

        # define disjuncts
        s_indicators = range(0, 2)

        def init_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_output_off(const, car_output):
                    return output[t, car_output] == 0
                dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

            else:  # technology on
                def init_min_partload(const):
                    return output[t, main_car] >= min_part_load * b_tec.var_size * rated_power
                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_output = Disjunct(set_t, s_indicators, rule=init_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_output[t, i] for i in s_indicators]
        b_tec.disjunction_output = Disjunction(set_t, rule=bind_disjunctions)

    # constraint on output ratios
    def init_output_output(const, t, car_output):
        if car_output == main_car:
            return Constraint.Skip
        else:
            return output[t, car_output] == phi[car_output] * output[t, main_car]
    b_tec.const_output_output = Constraint(set_t, b_tec.set_output_carriers, rule=init_output_output)

    # size constraint based main carrier output
    def init_size_constraint(const, t):
        return output[t, main_car] <= b_tec.var_size * rated_power
    b_tec.const_size = Constraint(set_t, rule=init_size_constraint)

    return b_tec

def constraints_tec_STOR(b_tec, tec_data, energyhub):
    """
    Adds constraints to technology blocks for tec_type STOR, resembling a storage technology

    The performance
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
    rated_power = tec_data.fitted_performance.rated_power

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
                                  bounds=(b_tec.para_size_min, b_tec.para_size_max * rated_power))

    # Abdditional parameters
    eta_in = coeff['eta_in']
    eta_out = coeff['eta_out']
    eta_lambda = coeff['lambda']
    charge_max = coeff['charge_max']
    discharge_max = coeff['discharge_max']
    ambient_loss_factor = coeff['ambient_loss_factor']

    # Size constraint
    def init_size_constraint(const, t, car):
        return b_tec.var_storage_level[t, car] <= b_tec.var_size * rated_power
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
        return input[t, car] <= charge_max * b_tec.var_size * rated_power
    b_tec.const_max_charge = Constraint(set_t, b_tec.set_input_carriers, rule=init_maximal_charge)

    def init_maximal_discharge(const,t,car):
        return output[t, car] <= discharge_max * b_tec.var_size * rated_power
    b_tec.const_max_discharge = Constraint(set_t, b_tec.set_input_carriers, rule=init_maximal_discharge)

    return b_tec