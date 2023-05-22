from pyomo.environ import *
from pyomo.gdp import *
import src.global_variables as global_variables
import src.model_construction as mc


def constraints_tec_dac_adsorption(b_tec, tec_data, energyhub):
    """
    Adds constraints to technology blocks for tec_type DAC_adsorption

    The model resembles as Direct Air Capture technology with a modular setup. It has a heat and electricity input
    and CO2 as an output. The performance is based on data for a generic solid sorbent, as described in the
    article (see below). The performance data is fitted to the ambient temperature and humidity at the respective
    node.

    The model is based on Wiegner et al. (2022). Optimal Design and Operation of Solid Sorbent Direct Air Capture
    Processes at Varying Ambient Conditions. Industrial and Engineering Chemistry Research, 2022,
    12649–12667. https://doi.org/10.1021/acs.iecr.2c00681. It resembles operation configuration 1 without water
    spraying.

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    # Comments on the equations refer to the equation numbers in the paper. All equations can be looked up there.
    # DATA OF TECHNOLOGY
    model = energyhub.model

    performance_data = tec_data.performance_data
    coeff = tec_data.fitted_performance.coefficients
    bounds = tec_data.fitted_performance.bounds
    modelled_with_full_res = tec_data.modelled_with_full_res


    # Full or reduced resolution
    input = b_tec.var_input
    output = b_tec.var_output
    set_t = model.set_t_full

    nr_segments = performance_data['nr_segments']
    ohmic_heating = performance_data['ohmic_heating']

    # Additional sets
    b_tec.set_pieces = RangeSet(1, nr_segments)

    # Additional decision variables
    b_tec.var_modules_on = Var(set_t,
                               domain=NonNegativeIntegers,
                               bounds=(b_tec.para_size_min, b_tec.para_size_max))

    def init_input_total_bounds(bds, t):
        return tuple(bounds['input']['total'][t - 1] * b_tec.para_size_max)
    b_tec.var_input_total = Var(set_t, within=NonNegativeReals, bounds=init_input_total_bounds)

    def init_input_el_bounds(bds, t):
        return tuple(bounds['input']['electricity'][t - 1] * b_tec.para_size_max)
    b_tec.var_input_el = Var(set_t, within=NonNegativeReals, bounds=init_input_el_bounds)

    def init_input_th_bounds(bds, t):
        return tuple(bounds['input']['heat'][t - 1] * b_tec.para_size_max)
    b_tec.var_input_th = Var(set_t, within=NonNegativeReals, bounds=init_input_th_bounds)

    def init_input_ohmic_bounds(bds, t):
        return tuple((el - th for el, th in zip(bounds['input']['electricity'][t - 1] * b_tec.para_size_max,
                                                   bounds['input']['heat'][t - 1] * b_tec.para_size_max)))
    b_tec.var_input_ohmic = Var(set_t, within=NonNegativeReals, bounds=init_input_ohmic_bounds)

    # Additional parameters
    alpha = coeff['alpha']
    beta = coeff['beta']
    b_point = coeff['b']
    gamma = coeff['gamma']
    delta = coeff['delta']
    a_point = coeff['a']
    eta_elth = performance_data['performance']['eta_elth']

    global_variables.big_m_transformation_required = 1

    # Input-Output relationship (eq. 1-5)
    def init_input_output(dis, t, ind):
        # Input-output (eq. 2)
        def init_output(const):
            return output[t, 'CO2'] == \
                   alpha[t-1, ind-1] * b_tec.var_input_total[t] + beta[t-1, ind-1] * b_tec.var_modules_on[t]
        dis.const_output = Constraint(rule=init_output)
        # Lower bound on the energy input (eq. 5)
        def init_input_low_bound(const):
            return b_point[t-1, ind-1] * b_tec.var_modules_on[t] <= b_tec.var_input_total[t]
        dis.const_input_on1 = Constraint(rule=init_input_low_bound)
        # Upper bound on the energy input (eq. 5)
        def init_input_up_bound(const):
            return b_tec.var_input_total[t] <= b_point[t-1, ind] * b_tec.var_modules_on[t]
        dis.const_input_on2 = Constraint(rule=init_input_up_bound)
    b_tec.dis_input_output = Disjunct(set_t, b_tec.set_pieces, rule=init_input_output)
    # Bind disjuncts
    def bind_disjunctions(dis, t):
        return [b_tec.dis_input_output[t, i] for i in b_tec.set_pieces]
    b_tec.disjunction_input_output = Disjunction(set_t, rule=bind_disjunctions)

    # Electricity-Heat relationship (eq. 7-10)
    def init_input_input(dis, t, ind):
        # Input-output (eq. 7)
        def init_input(const):
            return b_tec.var_input_el[t] == \
                   gamma[t-1, ind-1] * b_tec.var_input_total[t] + \
                   delta[t-1, ind-1] * b_tec.var_modules_on[t]
        dis.const_output = Constraint(rule=init_input)

        # Lower bound on the energy input (eq. 10)
        def init_input_low_bound(const):
            return a_point[t-1, ind-1] * b_tec.var_modules_on[t] <= b_tec.var_input_total[t]
        dis.const_input_on1 = Constraint(rule=init_input_low_bound)

        # Upper bound on the energy input (eq. 10)
        def init_input_up_bound(const):
            return b_tec.var_input_total[t] <= a_point[t-1, ind] * b_tec.var_modules_on[t]
        dis.const_input_on2 = Constraint(rule=init_input_up_bound)
    b_tec.dis_input_input = Disjunct(set_t, b_tec.set_pieces, rule=init_input_input)

    # Bind disjuncts
    def bind_disjunctions(dis, t):
        return [b_tec.dis_input_input[t, i] for i in b_tec.set_pieces]
    b_tec.disjunction_input_input = Disjunction(set_t, rule=bind_disjunctions)


    # Constraint of number of working modules (eq. 6)
    def init_modules_on(const, t):
        return b_tec.var_modules_on[t] <= b_tec.var_size
    b_tec.const_var_modules_on = Constraint(set_t, rule=init_modules_on)

    # Connection thermal and electric energy demand (eq. 11)
    def init_thermal_energy(const, t):
        return b_tec.var_input_th[t] == b_tec.var_input_total[t] - b_tec.var_input_el[t]
    b_tec.const_thermal_energy = Constraint(set_t, rule=init_thermal_energy)

    # Account for ohmic heating (eq. 12)
    def init_input_el(const, t):
        return input[t, 'electricity'] == b_tec.var_input_ohmic[t] + b_tec.var_input_el[t]
    b_tec.const_input_el = Constraint(set_t, rule=init_input_el)

    def init_input_th(const, t):
        return input[t, 'heat'] == b_tec.var_input_th[t] - b_tec.var_input_ohmic[t] * eta_elth
    b_tec.const_input_th = Constraint(set_t, rule=init_input_th)

    # If ohmic heating not allowed, set to zero
    if not ohmic_heating:
        def init_ohmic_heating(const, t):
            return b_tec.var_input_ohmic[t] == 0
        b_tec.const_ohmic_heating = Constraint(set_t, rule=init_ohmic_heating)

    return b_tec
