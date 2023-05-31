from pyomo.environ import *
from pyomo.gdp import *
import src.global_variables as global_variables
import src.model_construction as mc


def constraints_tec_hp(b_tec, tec_data, energyhub):
    """
    Adds constraints to technology blocks for tec_type HP (Heat Pump)

    Three different types of heat pumps are possible: air sourced ('HeatPump_AirSourced'), ground sourced
    ('HeatPump_GroundSourced') and water sourced ('HeatPump_WaterSourced'). Additionally, a heating curve is determined for
    heating for buildings. Then, the application needs to be set to either 'floor_heating' or 'radiator_heating'
    in the data file. Otherwise, the output temperature of the heat pump can also be set to a given temperature.
    The coefficient of performance at full load is calculated in the respective fitting function with the equations
    provided in Ruhnau, O., Hirth, L., & Praktiknjo, A. (2019). Time series of heat demand and
    heat pump efficiency for energy system modeling. Scientific Data, 6(1). https://doi.org/10.1038/s41597-019-0199-y

    The part load behavior is modelled after equation (3) in Xu, Z., Li, H., Xu, W., Shao, S., Wang, Z., Gou, X., Zhao,
    M., & Li, J. (2022). Investigation on the efficiency degradation characterization of low ambient temperature
    air source heat pump under partial load operation. International Journal of Refrigeration,
    133, 99–110. https://doi.org/10.1016/J.IJREFRIG.2021.10.002

    Essentially, the equations for the heat pump model are the same as for generic conversion technology type 1 (with
    time-dependent performance parameter).

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

    min_part_load = performance_data['min_part_load']
    performance_function_type = performance_data['performance_function_type']
    
    if performance_function_type >= 2:
        global_variables.big_m_transformation_required = 1
        
    # Full or reduced resolution
    if global_variables.clustered_data and not modelled_with_full_res:
        input = b_tec.var_input_aux
        output = b_tec.var_output_aux
        set_t = model.set_t_clustered
    else:
        input = b_tec.var_input
        output = b_tec.var_output
        set_t = model.set_t_full

    # Get performance parameters
    alpha1 = coeff['alpha1']
    if performance_function_type == 2:
        alpha2 = coeff['alpha2']
    if performance_function_type == 3:
        bp_x = coeff['bp_x']
        alpha2 = coeff['alpha2']

    if 'min_part_load' in performance_data:
        min_part_load = performance_data['min_part_load']
    else:
        min_part_load = 0

    # LINEAR, NO MINIMAL PARTLOAD, THROUGH ORIGIN
    if performance_function_type == 1:
        def init_input_output(const, t):
            return output[t, 'heat'] == alpha1[t-1] * input[t, 'electricity']
        b_tec.const_input_output = Constraint(set_t, rule=init_input_output)

    # LINEAR, MINIMAL PARTLOAD
    elif performance_function_type == 2:
        # define disjuncts for on/off
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const):
                    return input[t, 'electricity'] == 0
                dis.const_input = Constraint(rule=init_input_off)

                def init_output_off(const):
                    return output[t, 'heat'] == 0
                dis.const_output_off = Constraint(rule=init_output_off)
            else:  # technology on
                # input-output relation
                def init_input_output_on(const):
                    return output[t, 'heat'] == alpha1[t-1] * input[t, 'electricity'] + \
                           alpha2[t-1] * b_tec.var_size * rated_power
                dis.const_input_output_on = Constraint(rule=init_input_output_on)

                # min part load relation
                def init_min_partload(const):
                    return input[t, 'electricity'] >= \
                           min_part_load * b_tec.var_size * rated_power
                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(set_t, rule=bind_disjunctions)

    # PIECEWISE-AFFINE
    elif performance_function_type == 3:
        s_indicators = range(0, 2)

        def init_input_output(dis, t, ind):
            if ind == 0:  # technology off
                def init_input_off(const):
                    return input[t, 'electricity'] == 0
                dis.const_input_off = Constraint(rule=init_input_off)

                def init_output_off(const):
                    return output[t, 'heat'] == 0
                dis.const_output_off = Constraint( rule=init_output_off)

            else:  # piecewise definition
                def init_input_on1(const):
                    return input[t, 'electricity'] >= \
                           bp_x[t-1, ind] * b_tec.var_size * rated_power
                dis.const_input_on1 = Constraint(rule=init_input_on1)

                def init_input_on2(const):
                    return input[t, 'electricity'] <= \
                           bp_x[t-1, ind+1] * b_tec.var_size * rated_power
                dis.const_input_on2 = Constraint(rule=init_input_on2)

                def init_output_on(const):
                    return output[t, 'heat']  == \
                           alpha1[t-1, ind - 1] * input[t, 'electricity'] + \
                           alpha2[t-1, ind - 1] * b_tec.var_size * rated_power
                dis.const_input_output_on = Constraint(rule=init_output_on)

                # min part load relation
                def init_min_partload(const):
                    return input[t, 'electricity'] >= \
                           min_part_load * b_tec.var_size * rated_power
                dis.const_min_partload = Constraint(rule=init_min_partload)

        b_tec.dis_input_output = Disjunct(set_t, s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis, t):
            return [b_tec.dis_input_output[t, i] for i in s_indicators]
        b_tec.disjunction_input_output = Disjunction(set_t, rule=bind_disjunctions)

    # size constraint based on input
    def init_size_constraint(const, t):
        return input[t, 'electricity'] <= b_tec.var_size * rated_power
    b_tec.const_size = Constraint(set_t, rule=init_size_constraint)


    return b_tec
