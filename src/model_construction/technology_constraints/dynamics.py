from pyomo.environ import *
import warnings
from pyomo.gdp import *
import src.global_variables as global_variables
import src.model_construction as mc


def constraints_SUSD_logic(b_tec, tec_data, energyhub):
    """"Add description"""

    model = energyhub.model
    set_t = model.set_t_full

    # New variables
    b_tec.var_y = Var(set_t, domain=NonNegativeReals, bounds=(0, 1))
    b_tec.var_z = Var(set_t, domain=NonNegativeReals, bounds=(0, 1))

    # Collect parameters
    SU_time = tec_data.performance_data['SU_time']
    SD_time = tec_data.performance_data['SD_time']
    min_uptime = tec_data.performance_data['min_uptime']
    min_downtime = tec_data.performance_data['min_downtime'] + SU_time + SD_time
    max_startups = tec_data.performance_data['max_startups']

    # Enforce startup/shutdown logic
    def init_SUSD_logic1(const, t):
        if t == 1:
            return Constraint.Skip
        else:
            return b_tec.var_x[t] - b_tec.var_x[t - 1] == b_tec.var_y[t] - b_tec.var_z[t]

    b_tec.const_SUSD_logic1 = Constraint(set_t, rule=init_SUSD_logic1)

    def init_SUSD_logic2(const, t):
        if t >= min_uptime:
            return b_tec.var_y[t - min_uptime + 1] <= b_tec.var_x[t]
        else:
            return b_tec.var_y[len(set_t) + (t - min_uptime + 1)] <= b_tec.var_x[t]

    b_tec.const_SUSD_logic2 = Constraint(set_t, rule=init_SUSD_logic2)

    def init_SUSD_logic3(const, t):
        if t >= min_downtime:
            return b_tec.var_z[t - min_downtime + 1] <= 1 - b_tec.var_x[t]
        else:
            return b_tec.var_z[len(set_t) + (t - min_downtime + 1)] <= 1 - b_tec.var_x[t]

    b_tec.const_SUSD_logic3 = Constraint(set_t, rule=init_SUSD_logic3)

    # Constrain number of startups
    if not max_startups == -1:
        def init_max_startups(const):
            return sum(b_tec.var_y[t] for t in set_t) <= max_startups

        b_tec.const_max_startups = Constraint(rule=init_max_startups)

    return b_tec


def constraints_fast_SUSD_dynamics(b_tec, tec_data, energyhub):
    """Add description here"""
    model = energyhub.model
    set_t = model.set_t_full
    technology_model = tec_data.technology_model

    # Collect variables
    var_y = b_tec.var_y
    var_z = b_tec.var_z
    input = b_tec.var_input

    # Collect parameters
    SU_load = tec_data.performance_data['SU_load']
    SD_load = tec_data.performance_data['SD_load']
    main_car = tec_data.performance_data['main_input_carrier']

    # SU load limit
    s_indicators = range(0, 2)
    def init_SU_load(dis, t, ind):
        if ind == 0:  # no startup (y=0)
            dis.const_y_off = Constraint(expr=var_y[t] == 0)

        else:  # tech in startup
            dis.const_y_on = Constraint(expr=var_y[t] == 1)

            def init_SU_load_limit(cons, t):
                if technology_model == 'CONV1' or technology_model == 'CONV2':
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
                           <= b_tec.var_size * SU_load
                elif technology_model == 'CONV3':
                    return input[t, main_car] <= b_tec.var_size * SU_load
            dis.const_SU_load_limit = Constraint(set_t, rule=init_SU_load_limit)
    b_tec.dis_SU_load = Disjunct(set_t, s_indicators, rule=init_SU_load)

    def bind_disjunctions_SU_load(dis, t):
        return [b_tec.dis_SU_load[t, i] for i in s_indicators]
    b_tec.disjunction_SU_load = Disjunction(set_t, rule=bind_disjunctions_SU_load)

    #SD load limit
    s_indicators = range(0, 2)
    def init_SD_load(dis, t, ind):
        if ind == 0:  # no shutdown (z=0)
            dis.const_z_off = Constraint(expr=var_z[t] == 0)

        else:  # tech in shutdown
            dis.const_z_on = Constraint(expr=var_z[t] == 1)

            def init_SD_load_limit(cons, t):
                if t == 1:
                    return Constraint.Skip
                else:
                    if technology_model == 'CONV1' or technology_model == 'CONV2':
                        return sum(input[t - 1, car_input] for car_input in b_tec.set_input_carriers)\
                               <= b_tec.var_size * SD_load
                    elif technology_model == 'CONV3':
                        return input[t - 1, main_car] <= b_tec.var_size * SD_load
            dis.const_SD_load_limit = Constraint(set_t, rule=init_SD_load_limit)
    b_tec.dis_SD_load = Disjunct(set_t, s_indicators, rule=init_SD_load)

    def bind_disjunctions_SD_load(dis, t):
        return [b_tec.dis_SD_load[t, i] for i in s_indicators]
    b_tec.disjunction_SD_load = Disjunction(set_t, rule=bind_disjunctions_SD_load)

    return b_tec


def constraints_slow_SUSD_dynamics(alpha1, alpha2, b_tec, tec_data, energyhub):
    """Add description here"""
    model = energyhub.model
    set_t = model.set_t_full
    technology_model = tec_data.technology_model

    # Collect variables
    var_x = b_tec.var_x
    var_y = b_tec.var_y
    var_z = b_tec.var_z
    input = b_tec.var_input
    output = b_tec.var_output

    # Collect parameters
    min_part_load = tec_data.performance_data['min_part_load']
    ramping_rate = tec_data.performance_data['ramping_rate']
    SU_time = tec_data.performance_data['SU_time']
    SD_time = tec_data.performance_data['SD_time']
    min_uptime = tec_data.performance_data['min_uptime']
    min_downtime = tec_data.performance_data['min_downtime'] + SU_time + SD_time
    SU_load = tec_data.performance_data['SU_load']
    SD_load = tec_data.performance_data['SD_load']
    max_startups = tec_data.performance_data['max_startups']
    main_car = tec_data.performance_data['main_input_carrier']
    rated_power = tec_data.fitted_performance.rated_power

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
    s_indicators = range(0, SU_time + SD_time + 2)

    def init_SUSD_trajectories(dis, t, ind):
        if ind == 0:  # technology off
            dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

            def init_y_off(const, i):
                if t < len(set_t) - SU_time or i > SU_time - (len(set_t) - t):
                    return var_y[t - i + SU_time + 1] == 0
                else:
                    return var_y[(t - i + SU_time + 1) - len(set_t)] == 0
            dis.const_y_off = Constraint(range(1, SU_time + 1), rule= init_y_off)

            def init_z_off(const, j):
                if j <= t:
                    return var_z[t - j + 1] == 0
                else:
                    return var_z[len(set_t) + (t - j + 1)] == 0
            dis.const_z_off = Constraint(range(1, SD_time + 1), rule= init_z_off)

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
                    return var_y[t - ind + SU_time + 1] == 1
                else:
                    return var_y[(t - ind + SU_time + 1) - len(set_t)] == 1
            dis.const_y_on = Constraint(rule= init_y_on)

            def init_z_off(const):
                if t < len(set_t) - SU_time or ind > SU_time - (len(set_t) - t):
                    return var_z[t - ind + SU_time + 1] == 0
                else:
                    return var_z[(t - ind + SU_time + 1) - len(set_t)] == 0
            dis.const_z_off = Constraint(rule= init_z_off)

            def init_input_SU(cons):
                if technology_model == 'CONV1' or technology_model == 'CONV2':
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
                           == b_tec.var_size * SU_trajectory[ind - 1]
                elif technology_model == 'CONV3':
                    return input[t, main_car] == b_tec.var_size * SU_trajectory[ind - 1]
            dis.const_input_SU = Constraint(rule=init_input_SU)

            def init_output_SU(const, car_output):
                return output[t, car_output] == alpha1[car_output] * input[t, main_car] + \
                       alpha2[car_output] * b_tec.var_size * rated_power
            dis.const_output_SU = Constraint(b_tec.set_output_carriers, rule=init_output_SU)

        elif ind in range(SU_time + 1, SU_time + SD_time + 1): # technology in shutdown
            ind_SD = ind - SU_time
            dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

            def init_z_on(const):
                if ind_SD <= t:
                    return var_z[t - ind_SD + 1] == 1
                else:
                    return var_z[len(set_t) + (t - ind_SD + 1)] == 1
            dis.const_z_on = Constraint(rule= init_z_on)

            def init_y_off(const):
                if ind_SD <= t:
                    return var_y[t - ind_SD + 1] == 0
                else:
                    return var_y[len(set_t) + (t - ind_SD + 1)] == 0
            dis.const_y_off = Constraint(rule= init_y_off)

            def init_input_SD(cons):
                if technology_model == 'CONV1' or technology_model == 'CONV2':
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
                           == b_tec.var_size * SD_trajectory[ind_SD - 1]
                elif technology_model == 'CONV3':
                    return input[t, main_car] == b_tec.var_size * SD_trajectory[ind_SD - 1]
            dis.const_input_SD = Constraint(rule=init_input_SD)

            def init_output_SD(const, car_output):
                return output[t, car_output] == alpha1[car_output] * input[t, main_car] + \
                       alpha2[car_output] * b_tec.var_size * rated_power
            dis.const_output_SD = Constraint(b_tec.set_output_carriers, rule=init_output_SD)

        elif ind > SU_time + SD_time:
            dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

            def init_input_output_on(const, car_output):
                return output[t, car_output] == \
                       alpha1[car_output] * input[t, main_car] + \
                       alpha2[car_output] * b_tec.var_size * rated_power

            dis.const_input_output_on = Constraint(b_tec.set_output_carriers, rule=init_input_output_on)

            # min part load relation
            def init_min_partload(const):
                return input[t, main_car] >= min_part_load * b_tec.var_size * rated_power

            dis.const_min_partload = Constraint(rule=init_min_partload)

    b_tec.dis_SUSD_trajectory = Disjunct(set_t, s_indicators, rule=init_SUSD_trajectories)

    def bind_disjunctions_SUSD(dis, t):
        return [b_tec.dis_SUSD_trajectory[t, k] for k in s_indicators]
    b_tec.disjunction_SUSD_traject = Disjunction(set_t, rule=bind_disjunctions_SUSD)

    return b_tec