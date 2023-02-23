import numbers
import numpy as np
from src.model_construction.generic_technology_constraints import *
import src.model_construction as mc
import src.config_model as m_config



def add_technologies(nodename, set_tecsToAdd, model, data, b_node):
    r"""
    Adds all technologies as model blocks to respective node.

    This function initializes parameters and decision variables for all technologies at respective node.
    For each technology, it adds one block indexed by the set of all technologies at the node :math:`S_n`.
    This function adds Sets, Parameters, Variables and Constraints that are common for all technologies.
    For each technology type, individual parts are added. The following technology types are currently available:

    - Type RES: Renewable technology with cap_factor as input. Constructed with \
      :func:`src.model_construction.generic_technology_constraints.constraints_tec_RES`
    - Type CONV1: n inputs -> n output, fuel and output substitution. Constructed with \
      :func:`src.model_construction.generic_technology_constraints.constraints_tec_CONV1`
    - Type CONV2: n inputs -> n output, fuel substitution. Constructed with \
      :func:`src.model_construction.generic_technology_constraints.constraints_tec_CONV2`
    - Type CONV2: n inputs -> n output, no fuel and output substitution. Constructed with \
      :func:`src.model_construction.generic_technology_constraints.constraints_tec_CONV3`
    - Type STOR: Storage technology (1 input -> 1 output). Constructed with \
      :func:`src.model_construction.generic_technology_constraints.constraints_tec_STOR`

    **Set declarations:**

    - Set of input carriers
    - Set of output carriers

    **Parameter declarations:**

    - Min Size
    - Max Size
    - Output max (same as size max)
    - Unit CAPEX (annualized from given data on up-front CAPEX, lifetime and discount rate)
    - Variable OPEX
    - Fixed OPEX

    **Variable declarations:**

    - Size (can be integer or continuous)
    - Input for each input carrier
    - Output for each output carrier
    - CAPEX
    - Variable OPEX
    - Fixed OPEX

    **Constraint declarations**
    - CAPEX, can be linear (for ``capex_model == 1``) or piecewise linear (for ``capex_model == 2``). Linear is defined as:

    .. math::
        CAPEX_{tec} = Size_{tec} * UnitCost_{tec}

    - Variable OPEX: defined per unit of output for the main carrier:

    .. math::
        OPEXvar_{t, tec} = Output_{t, maincarrier} * opex_{var} \forall t \in T

    - Fixed OPEX: defined as a fraction of annual CAPEX:

    .. math::
        OPEXfix_{tec} = CAPEX_{tec} * opex_{fix}

    :param str nodename: name of node for which technology is installed
    :param object b_node: pyomo block for respective node
    :param object model: pyomo model
    :param DataHandle data:  instance of a DataHandle
    :return: model
    """
    def init_technology_block(b_tec, tec):

        # Get options from data
        tec_data = data.technology_data[nodename][tec]
        technology_model = tec_data.technology_model
        existing = tec_data.existing
        size_is_int = tec_data.size_is_int
        size_min = tec_data.size_min
        size_max = tec_data.size_max
        economics = tec_data.economics
        performance_data = tec_data.performance_data

        # PARAMETERS
        if size_is_int:
            unit_size = u.dimensionless
        else:
            unit_size = u.MW
        b_tec.para_size_min = Param(domain=NonNegativeReals, initialize=size_min, units=unit_size)
        b_tec.para_size_max = Param(domain=NonNegativeReals, initialize=size_max, units=unit_size)
        b_tec.para_unit_CAPEX = Param(domain=Reals, initialize=economics.capex_data['unit_capex'],
                                      units=u.EUR/unit_size)

        r = economics.discount_rate
        t = economics.lifetime
        annualization_factor = mc.annualize(r, t)
        b_tec.para_unit_CAPEX_annual = Param(domain=Reals,
                                             initialize= annualization_factor * economics.capex_data['unit_capex'],
                                             units=u.EUR/unit_size)
        b_tec.para_OPEX_variable = Param(domain=Reals, initialize=economics.opex_variable,
                                         units=u.EUR/u.MWh)
        b_tec.para_OPEX_fixed = Param(domain=Reals, initialize=economics.opex_fixed,
                                      units=u.EUR/u.EUR)
        b_tec.para_tec_emissionfactor = Param(domain=Reals, initialize=performance_data['emission_factor'],
                                      units=u.t/u.MWh)

        # SETS
        b_tec.set_input_carriers = Set(initialize=performance_data['input_carrier'])
        b_tec.set_output_carriers = Set(initialize=performance_data['output_carrier'])

        # DECISION VARIABLES
        # Input
        # TODO: if size is integer units do not work
        output_bounds = calculate_output_bounds(tec_data)
        input_bounds = calculate_input_bounds(tec_data)
        if not technology_model == 'RES':
            def init_input_bounds(bounds, t, car):
                return input_bounds[car]
            b_tec.var_input = Var(model.set_t, b_tec.set_input_carriers, within=NonNegativeReals,
                                  bounds=init_input_bounds, units=u.MW)
        # Output
        def init_output_bounds(bounds, t, car):
            return output_bounds[car]
        b_tec.var_output = Var(model.set_t, b_tec.set_output_carriers, within=NonNegativeReals,
                               bounds=init_output_bounds, units=u.MW)

        # Emissions
        b_tec.var_tec_emissions_pos = Var(model.set_t, within=NonNegativeReals, units=u.t)
        b_tec.var_tec_emissions_neg = Var(model.set_t, within=NonNegativeReals, units=u.t)

        # Size
        if size_is_int:
            b_tec.var_size = Var(within=NonNegativeIntegers, bounds=(b_tec.para_size_min, b_tec.para_size_max))
        else:
            b_tec.var_size = Var(within=NonNegativeReals, bounds=(b_tec.para_size_min, b_tec.para_size_max),
                                 units=u.MW)

        # Capex/Opex
        b_tec.var_CAPEX = Var(units=u.EUR)
        b_tec.var_OPEX_variable = Var(model.set_t, units=u.EUR)
        b_tec.var_OPEX_fixed = Var(units=u.EUR)

        # GENERAL CONSTRAINTS
        # Capex
        if economics.capex_model == 1:
            b_tec.const_CAPEX = Constraint(expr=b_tec.var_size * b_tec.para_unit_CAPEX_annual == b_tec.var_CAPEX)
        elif economics.capex_model == 2:
            m_config.presolve.big_m_transformation_required = 1
            # TODO Implement link between bps and data
            b_tec.const_CAPEX = Piecewise(b_tec.var_CAPEX, b_tec.var_size,
                                          pw_pts=bp_x,
                                          pw_constr_type='EQ',
                                          f_rule=bp_y,
                                          pw_repn='SOS2')
        # fixed Opex
        b_tec.const_OPEX_fixed = Constraint(expr=b_tec.var_CAPEX * b_tec.para_OPEX_fixed == b_tec.var_OPEX_fixed)

        # variable Opex
        def init_OPEX_variable(const, t):
            return sum(b_tec.var_output[t, car] for car in b_tec.set_output_carriers) * b_tec.para_OPEX_variable == \
                   b_tec.var_OPEX_variable[t]
        b_tec.const_OPEX_variable = Constraint(model.set_t, rule=init_OPEX_variable)

        # Emissions
        if technology_model == 'RES':
            # Set emissions to zero
            def init_tec_emissions_pos_RES(const, t):
                return b_tec.var_tec_emissions_pos[t] == 0
            b_tec.const_tec_emissions_pos = Constraint(model.set_t, rule=init_tec_emissions_pos_RES)
            def init_tec_emissions_neg_RES(const, t):
                return b_tec.var_tec_emissions_neg[t] == 0
            b_tec.const_tec_emissions_neg = Constraint(model.set_t, rule=init_tec_emissions_neg_RES)
        else:
            # Calculate emissions from emission factor
            def init_tec_emissions_pos(const, t):
                if performance_data['emission_factor'] >= 0:
                    return b_tec.var_input[t, performance_data['main_input_carrier']] \
                           * b_tec.para_tec_emissionfactor \
                           == b_tec.var_tec_emissions_pos[t]
                else:
                    return b_tec.var_tec_emissions_pos[t] == 0
            b_tec.const_tec_emissions = Constraint(model.set_t, rule=init_tec_emissions_pos)

            def init_tec_emissions_neg(const, t):
                if performance_data['emission_factor'] < 0:
                    return b_tec.var_input[t, performance_data['main_input_carrier']] \
                           (-b_tec.para_tec_emissionfactor) == \
                           b_tec.var_tec_emissions_neg[t]
                else:
                    return b_tec.var_tec_emissions_neg[t] == 0
            b_tec.const_tec_emissions_neg = Constraint(model.set_t, rule=init_tec_emissions_neg)


        # TECHNOLOGY TYPES
        if technology_model == 'RES': # Renewable technology with cap_factor as input
            b_tec = constraints_tec_RES(model, b_tec, tec_data)

        elif technology_model == 'CONV1': # n inputs -> n output, fuel and output substitution
            b_tec = constraints_tec_CONV1(model, b_tec, tec_data)

        elif technology_model == 'CONV2': # n inputs -> n output, fuel and output substitution
            b_tec = constraints_tec_CONV2(model, b_tec, tec_data)

        elif technology_model == 'CONV3':  # 1 input -> n outputs, output flexible, linear performance
            b_tec = constraints_tec_CONV3(model, b_tec, tec_data)

        elif technology_model == 'STOR': # Storage technology (1 input -> 1 output)
            if m_config.presolve.clustered_data == 1:
                hourly_order_time_slices = data.k_means_specs['keys']['hourly_order']
            else:
                hourly_order_time_slices = np.arange(1, len(model.set_t)+1)
            b_tec = constraints_tec_STOR(model, b_tec, tec_data, hourly_order_time_slices)

        if m_config.presolve.big_m_transformation_required:
            mc.perform_disjunct_relaxation(b_tec)

        return b_tec

    # Create a new block containing all new technologies.
    if b_node.find_component('tech_blocks_new'):
        b_node.del_component(b_node.tech_blocks_new)
    b_node.tech_blocks_new = Block(set_tecsToAdd, rule=init_technology_block)

    # If it exists, carry over active tech blocks to temporary block
    if b_node.find_component('tech_blocks_active'):
        b_node.tech_blocks_existing = Block(b_node.set_tecsAtNode)
        for tec in b_node.set_tecsAtNode:
            b_node.tech_blocks_existing[tec].transfer_attributes_from(b_node.tech_blocks_active[tec])
        b_node.del_component(b_node.tech_blocks_active)

    # Create a block containing all active technologies at node
    if not set(set_tecsToAdd).issubset(b_node.set_tecsAtNode):
        b_node.set_tecsAtNode.add(set_tecsToAdd)

    def init_active_technology_blocks(bl, tec):
        if tec in set_tecsToAdd:
            bl.transfer_attributes_from(b_node.tech_blocks_new[tec])
        else:
            bl.transfer_attributes_from(b_node.tech_blocks_existing[tec])
    b_node.tech_blocks_active = Block(b_node.set_tecsAtNode, rule=init_active_technology_blocks)

    if b_node.find_component('tech_blocks_new'):
        b_node.del_component(b_node.tech_blocks_new)
    if b_node.find_component('tech_blocks_existing'):
        b_node.del_component(b_node.tech_blocks_existing)
    return b_node



def calculate_input_bounds(tec_data):
    """
    Calculates bounds for technology inputs for each input carrier
    """
    technology_model = tec_data.technology_model
    size_max = tec_data.size_max
    performance_data = tec_data.performance_data

    bounds = {}
    if technology_model == 'CONV3':
        main_car = performance_data['main_input_carrier']
        for c in performance_data['input_carrier']:
            if c == main_car:
                bounds[c] = (0, size_max)
            else:
                bounds[c] = (0, size_max * performance_data['input_ratios'][c])
    else:
        for c in performance_data['input_carrier']:
            bounds[c] = (0, size_max)
    return bounds

def calculate_output_bounds(tec_data):
    """
    Calculates bounds for technology outputs for each input carrier
    """
    technology_model = tec_data.technology_model
    size_is_int = tec_data.size_is_int
    size_max = tec_data.size_max
    performance_data = tec_data.performance_data
    fitted_performance = tec_data.fitted_performance

    bounds = {}

    if technology_model == 'RES':  # Renewable technology with cap_factor as input
        if size_is_int:
            rated_power = fitted_performance['rated_power']
        else:
            rated_power = 1
        cap_factor = fitted_performance['capacity_factor']
        for c in performance_data['output_carrier']:
            max_bound = float(size_max * max(cap_factor) * rated_power)
            bounds[c] = (0, max_bound)

    elif technology_model == 'CONV1':  # n inputs -> n output, fuel and output substitution
        performance_function_type = performance_data['performance_function_type']
        alpha1 = fitted_performance['out']['alpha1']
        for c in performance_data['output_carrier']:
            if performance_function_type == 1:
                max_bound = size_max * alpha1
            if performance_function_type == 2:
                alpha2 = fitted_performance['out']['alpha2']
                max_bound = size_max * (alpha1 + alpha2)
            if performance_function_type == 3:
                alpha2 = fitted_performance['out']['alpha2']
                max_bound = size_max * (alpha1[-1] + alpha2[-1])
            bounds[c] = (0, max_bound)

    elif technology_model == 'CONV2':  # n inputs -> n output, fuel and output substitution
        alpha1 = {}
        alpha2 = {}
        performance_function_type = performance_data['performance_function_type']
        for c in performance_data['performance']['out']:
            alpha1[c] = fitted_performance[c]['alpha1']
            if performance_function_type == 1:
                max_bound = alpha1[c] * size_max
            if performance_function_type == 2:
                alpha2[c] = fitted_performance[c]['alpha2']
                max_bound = size_max * (alpha1[c] + alpha2[c])
            if performance_function_type == 3:
                alpha2[c] = fitted_performance[c]['alpha2']
                max_bound = size_max * (alpha1[c][-1] + alpha2[c][-1])
            bounds[c] = (0, max_bound)

    elif technology_model == 'CONV3':  # 1 input -> n outputs, output flexible, linear performance
        alpha1 = {}
        alpha2 = {}
        performance_function_type = performance_data['performance_function_type']
        # Get performance parameters
        for c in performance_data['performance']['out']:
            alpha1[c] = fitted_performance[c]['alpha1']
            if performance_function_type == 1:
                max_bound = alpha1[c] * size_max
            if performance_function_type == 2:
                alpha2[c] = fitted_performance[c]['alpha2']
                max_bound = size_max * (alpha1[c] + alpha2[c])
            if performance_function_type == 3:
                alpha2[c] = fitted_performance[c]['alpha2']
                max_bound = size_max * (alpha1[c][-1] + alpha2[c][-1])
            bounds[c] = (0, max_bound)

    elif technology_model == 'STOR':  # Storage technology (1 input -> 1 output)
        for c in performance_data['output_carrier']:
            bounds[c] = (0, size_max)

    return bounds