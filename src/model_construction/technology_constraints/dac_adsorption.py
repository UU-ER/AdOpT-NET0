from pyomo.environ import *
from pyomo.environ import units as u
from pyomo.gdp import *
import src.config_model as m_config

def constraints_tec_dac_adsorption(model, b_tec, tec_data):
    """
    Adds constraints to technology blocks for tec_type DAC_adsorption

    This model is based on Wiegner et al. (2022). Optimal Design and Operation of Solid Sorbent Direct Air Capture
    Processes at Varying Ambient Conditions. Industrial and Engineering Chemistry Research, 2022,
    12649–12667. https://doi.org/10.1021/acs.iecr.2c00681. It resembles operation configuration 1 without water
    spraying.

    The comments on the equations refer to the equation numbers in the paper. All equations can be looked up there.

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    # DATA OF TECHNOLOGY
    fitted_performance = tec_data.fitted_performance

    nr_bp = 3
    ohmic_heating = 1

    # Additional sets
    b_tec.set_pieces = RangeSet(1, nr_bp-1)

    # Additional decision variables
    b_tec.var_modules_on = Var(model.set_t,
                               domain=NonNegativeIntegers,
                               bounds=(b_tec.para_size_min, b_tec.para_size_max))
    b_tec.var_input_total = Var(model.set_t,
                               domain=NonNegativeReals)
    b_tec.var_input_el = Var(model.set_t,
                               domain=NonNegativeReals)
    b_tec.var_input_th = Var(model.set_t,
                               domain=NonNegativeReals)
    b_tec.var_input_ohmic = Var(model.set_t,
                               domain=NonNegativeReals)

    # Additional parameters
    def init_alpha(para, t, ind):
        return fitted_performance['alpha'][t, ind]
    b_tec.para_alpha = Param(model.set_t, b_tec.set_pieces, domain=NonNegativeReals, rule=init_alpha)
    def init_beta(para, t, ind):
        return fitted_performance['beta'][t, ind]
    b_tec.para_beta = Param(model.set_t, b_tec.set_pieces, domain=NonNegativeReals, rule=init_beta)
    def init_b_low(para, t, ind):
        return fitted_performance['b'][t, ind-1]
    b_tec.para_b_low = Param(model.set_t, b_tec.set_pieces, domain=NonNegativeReals, rule=init_b_low)
    def init_b_up(para, t, ind):
        return fitted_performance['b'][t, ind]
    b_tec.para_b_up = Param(model.set_t, b_tec.set_pieces, domain=NonNegativeReals, rule=init_b_up)

    def init_gamma(para, t, ind):
        return fitted_performance['gamma'][t, ind]
    b_tec.para_gamma = Param(model.set_t, b_tec.set_pieces, domain=NonNegativeReals, rule=init_gamma)
    def init_delta(para, t, ind):
        return fitted_performance['delta'][t, ind]
    b_tec.para_delta = Param(model.set_t, b_tec.set_pieces, domain=NonNegativeReals, rule=init_delta)
    def init_a_low(para, t, ind):
        return fitted_performance['a'][t, ind-1]
    b_tec.para_a_low = Param(model.set_t, b_tec.set_pieces, domain=NonNegativeReals, rule=init_a_low)
    def init_a_up(para, t, ind):
        return fitted_performance['a'][t, ind]
    b_tec.para_a_up = Param(model.set_t, b_tec.set_pieces, domain=NonNegativeReals, rule=init_a_up)

    b_tec.para_eta_elth = Param(initialize=fitted_performance['eta_elth'])

    m_config.presolve.big_m_transformation_required = 1

    # Input-Output relationship (eq. 1-5)
    def init_input_output(dis, t, ind):
        # Input-output (eq. 2)
        def init_output(const):
            return b_tec.var_output[t, b_tec.set_output_carriers] == \
                   b_tec.para_alpha[t, ind] * b_tec.var_input_total[t] + b_tec.para_beta[t, ind] * b_tec.var_modules_on[t]
        dis.const_output = Constraint(rule=init_output)
        # Lower bound on the energy input (eq. 5)
        def init_input_low_bound(const):
            return b_tec.para_b_low[t, ind] * b_tec.var_modules_on[t] <= b_tec.var_input_total[t]
        dis.const_input_on1 = Constraint(rule=init_input_low_bound)
        # Upper bound on the energy input (eq. 5)
        def init_input_up_bound(const):
            return b_tec.var_input_total[t] <= b_tec.para_b_up[ind] * b_tec.var_modules_on[t]
        dis.const_input_on2 = Constraint(rule=init_input_up_bound)
    b_tec.dis_input_output = Disjunct(model.set_t, b_tec.set_pieces, rule=init_input_output)
    # Bind disjuncts
    def bind_disjunctions(dis, t):
        return [b_tec.dis_input_output[t, i] for i in b_tec.set_pieces]
    b_tec.disjunction_input_output = Disjunction(model.set_t, rule=bind_disjunctions)

    # Electricity-Heat relationship (eq. 7-10)
    def init_input_input(dis, t, ind):
        # Input-output (eq. 7)
        def init_input(const):
            return b_tec.var_input_el[t] == \
                   b_tec.para_gamma[t, ind] * b_tec.var_input_total[t] + \
                   b_tec.para_delta[t, ind] * b_tec.var_modules_on[t]
        dis.const_output = Constraint(rule=init_input)

        # Lower bound on the energy input (eq. 10)
        def init_input_low_bound(const):
            return b_tec.para_a_low[t, ind] * b_tec.var_modules_on[t] <= b_tec.var_input_total[t]
        dis.const_input_on1 = Constraint(rule=init_input_low_bound)

        # Upper bound on the energy input (eq. 10)
        def init_input_up_bound(const):
            return b_tec.var_input_total[t] <= b_tec.para_a_up[t, ind] * b_tec.var_modules_on[t]
        dis.const_input_on2 = Constraint(rule=init_input_up_bound)
    b_tec.dis_input_input = Disjunct(model.set_t, b_tec.set_pieces, rule=init_input_input)

    # Bind disjuncts
    def bind_disjunctions(dis, t):
        return [b_tec.dis_input_input[t, i] for i in b_tec.set_pieces]
    b_tec.disjunction_input_input = Disjunction(model.set_t, rule=bind_disjunctions)


    # Constraint of number of working modules (eq. 6)
    def init_modules_on(const, t):
        return b_tec.var_modules_on[t] <= b_tec.var_size
    b_tec.const_var_modules_on = Constraint(model.set_t, rule=init_modules_on)

    # Connection thermal and electric energy demand (eq. 11)
    def init_thermal_energy(const, t):
        return b_tec.var_input_th[t] == b_tec.var_input_total[t] - b_tec.var_input_el[t]
    b_tec.const_thermal_energy = Constraint(model.set_t, rule=init_thermal_energy)

    # Account for ohmic heating (eq. 12)
    def init_input_el(const, t):
        return b_tec.var_input[t, 'electricity'] == b_tec.var_input_ohmic[t] + b_tec.var_input_el[t]
    b_tec.const_input_el = Constraint(model.set_t, rule=init_input_el)

    def init_input_th(const, t):
        return b_tec.var_input[t, 'heat'] == b_tec.var_input_th[t] - b_tec.var_input_ohmic[t] * b_tec.para_eta_elth
    b_tec.const_input_th = Constraint(model.set_t, rule=init_input_th)

    # If ohmic heating not allowed, set to zero
    if not ohmic_heating:
        def init_ohmic_heating(const, t):
            return b_tec.var_input_ohmic[t] == 0
        b_tec.const_ohmic_heating = Constraint(model.set_t, rule=init_ohmic_heating)

    return b_tec
