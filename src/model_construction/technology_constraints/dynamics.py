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
    rated_power = tec_data.fitted_performance.rated_power

    # SU load limit
    s_indicators = range(0, 2)
    def init_SU_load(dis, t, ind):
        if ind == 0:  # no startup (y=0)
            dis.const_y_off = Constraint(expr=var_y[t] == 0)

        else:  # tech in startup
            dis.const_y_on = Constraint(expr=var_y[t] == 1)

            def init_SU_load_limit(cons):
                if technology_model == 'CONV1' or technology_model == 'CONV2':
                    return sum(input[t, car_input] for car_input in b_tec.set_input_carriers) \
                           <= b_tec.var_size * SU_load * rated_power
                elif technology_model == 'CONV3':
                    return input[t, main_car] <= b_tec.var_size * SU_load * rated_power
            dis.const_SU_load_limit = Constraint(rule=init_SU_load_limit)

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

            def init_SD_load_limit(cons):
                if t == 1:
                    return Constraint.Skip
                else:
                    if technology_model == 'CONV1' or technology_model == 'CONV2':
                        return sum(input[t - 1, car_input] for car_input in b_tec.set_input_carriers)\
                               <= b_tec.var_size * SD_load * rated_power
                    elif technology_model == 'CONV3':
                        return input[t - 1, main_car] <= b_tec.var_size * SD_load * rated_power
            dis.const_SD_load_limit = Constraint(rule=init_SD_load_limit)
    b_tec.dis_SD_load = Disjunct(set_t, s_indicators, rule=init_SD_load)

    def bind_disjunctions_SD_load(dis, t):
        return [b_tec.dis_SD_load[t, i] for i in s_indicators]
    b_tec.disjunction_SD_load = Disjunction(set_t, rule=bind_disjunctions_SD_load)

    return b_tec
