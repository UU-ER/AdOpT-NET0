from pyomo.environ import *
from pyomo.gdp import *
from pyomo.environ import units as u
import src.model_construction as mc
import src.global_variables as global_variables

def constraints_tec_gt(model, b_tec, tec_data):
    """
    Adds constraints to technology blocks for gas turbines

    Hydrogen and Natural Gas Turbines are possible at four different sizes, as indicated by the file names
    of the data. Performance data and the model is taken from Weimann, L., Ellerker, M., Kramer, G. J., &
    Gazzani, M. (2019). Modeling gas turbines in multi-energy systems: A linear model accounting for part-load
    operation, fuel, temperature, and sizing effects. International Conference on Applied Energy.
    https://doi.org/10.46855/energy-proceedings-5280

    A small adaption is made: Natural gas turbines can co-fire hydrogen up to 5% of the energy content


    **Parameter declarations:**

    - :math:`Input_{min}`: Minimal input per turbine

    - :math:`Input_{max}`: Maximal input per turbine

    - :math:`in_{H2max}`: Maximal H2 admixture to fuel (only for natural gas turbines, default is 0.05)

    - :math:`{\\alpha}`: Performance parameter for electricity output

    - :math:`{\\beta}`: Performance parameter for electricity output

    - :math:`{\\epsilon}`: Performance parameter for heat output

    - :math:`f({\\Theta})`: Ambient temperature correction factor

    **Variable declarations:**

    - Total fuel input in :math:`t`: :math:`Input_{tot, t}`

    - Number of turbines on in :math:`t`: :math:`N_{on,t}`

    **Constraint declarations:**

    - Input calculation (For hydrogen turbines, :math:`Input_{NG, t}` is zero, and the second constraint is removed):

      .. math::
        Input_{H2, t} + Input_{NG, t} = Input_{tot, t}

      .. math::
        Input_{H2, t} \leq in_{H2max} Input_{tot, t}

    - Turbines on:

      .. math::
        N_{on, t} \leq S

    - If technology is on:

      .. math::
        Output_{el,t} = ({\\alpha} Input_{tot, t} + {\\beta} * N_{on, t}) *f({\\Theta})

      .. math::
        Output_{th,t} = {\\epsilon} Input_{tot, t} - Output_{el,t}

      .. math::
        Input_{min} * N_{on, t} \leq Input_{tot, t} \leq Input_{max} * N_{on, t}

    - If the technology is off, input and output is set to 0:

      .. math::
         \sum(Output_{t, car}) = 0

      .. math::
         \sum(Input_{t, car}) = 0

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    # DATA OF TECHNOLOGY
    fitted_performance = tec_data.fitted_performance
    performance_data = tec_data.performance_data
    rated_power = fitted_performance['rated_power']

    # Parameter declaration
    b_tec.para_in_min = Param(domain=NonNegativeReals, initialize=performance_data['in_min'])
    b_tec.para_in_max = Param(domain=NonNegativeReals, initialize=performance_data['in_max'])
    b_tec.para_max_H2_admixture = Param(domain=NonNegativeReals, initialize=performance_data['max_H2_admixture'])
    b_tec.para_alpha = Param(domain=Reals, initialize=fitted_performance['alpha'])
    b_tec.para_beta = Param(domain=Reals, initialize=fitted_performance['beta'])
    b_tec.para_epsilon = Param(domain=NonNegativeReals, initialize=fitted_performance['epsilon'])
    f  = fitted_performance['f']

    global_variables.big_m_transformation_required = 1

    # Additional decision variables
    size_max = tec_data.size_max
    def init_input_bounds(bounds, t):
        if len(performance_data['input_carrier']) == 2:
            car = 'gas'
        else:
            car = 'hydrogen'
        return tuple(fitted_performance['input_bounds'][car][t - 1, :] * size_max)
    b_tec.var_total_input = Var(b_tec.set_t, within=NonNegativeReals,
                                bounds=init_input_bounds, units=u.MW)

    b_tec.var_units_on = Var(b_tec.set_t, within=NonNegativeIntegers,
                             bounds=(0, size_max))

    # Calculate total input
    def init_total_input(const, t):
        return b_tec.var_total_input[t] == sum(input[t, car_input]
                                               for car_input in b_tec.set_input_carriers)
    b_tec.const_total_input = Constraint(b_tec.set_t, rule=init_total_input)

    # Constrain hydrogen input
    if len(performance_data['input_carrier']) == 2:
        def init_h2_input(const, t):
            return input[t, 'hydrogen'] <= b_tec.var_total_input[t] * b_tec.para_max_H2_admixture
        b_tec.const_h2_input = Constraint(b_tec.set_t, rule=init_h2_input)

    # LINEAR, MINIMAL PARTLOAD
    s_indicators = range(0, 2)

    def init_input_output(dis, t, ind):
        if ind == 0:  # technology off
            def init_input_off(const, car):
                return input[t, car] == 0
            dis.const_input = Constraint(b_tec.set_input_carriers, rule=init_input_off)

            def init_output_off(const, car):
                return output[t, car] == 0
            dis.const_output_off = Constraint(b_tec.set_output_carriers, rule=init_output_off)

        else:  # technology on
            # input-output relation
            def init_input_output_on_el(const):
                return output[t, 'electricity'] == (b_tec.para_alpha * b_tec.var_total_input[t] + \
                                                              b_tec.para_beta * b_tec.var_units_on[t]) * f[t-1]
            dis.const_input_output_on_el = Constraint(rule=init_input_output_on_el)

            def init_input_output_on_th(const):
                return output[t, 'heat'] == b_tec.para_epsilon * b_tec.var_total_input[t] - \
                       output[t, 'electricity']
            dis.const_input_output_on_th = Constraint(rule=init_input_output_on_th)

            # min part load relation
            def init_min_input(const):
                return b_tec.var_total_input[t] >= \
                       b_tec.para_in_min * b_tec.var_units_on[t]
            dis.const_min_input = Constraint(rule=init_min_input)

            def init_max_input(const):
                return b_tec.var_total_input[t] <= \
                       b_tec.para_in_max * b_tec.var_units_on[t]
            dis.const_max_input = Constraint(rule=init_max_input)

    b_tec.dis_input_output = Disjunct(b_tec.set_t, s_indicators, rule=init_input_output)

    # Bind disjuncts
    def bind_disjunctions(dis, t):
        return [b_tec.dis_input_output[t, i] for i in s_indicators]
    b_tec.disjunction_input_output = Disjunction(b_tec.set_t, rule=bind_disjunctions)

    # Technologies on
    def init_n_on(const, t):
        return b_tec.var_units_on[t] <= b_tec.var_size
    b_tec.const_n_on = Constraint(b_tec.set_t, rule=init_n_on)

    return b_tec
