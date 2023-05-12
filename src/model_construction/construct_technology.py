from src.model_construction.technology_constraints import *


def define_size(b_tec, tec_data):
    """
    Defines variables and parameters related to technology size.

    Parameters defined:
    - size min
    - size max
    - size initial (for existing technologies)

    Variables defined:
    - size
    """
    size_is_int = tec_data.size_is_int
    size_min = tec_data.size_min
    existing = tec_data.existing
    decommission = tec_data.decommission
    if existing:
        size_initial = tec_data.size_initial
        size_max = size_initial
    else:
        size_max = tec_data.size_max

    if size_is_int:
        size_domain = NonNegativeIntegers
    else:
        size_domain = NonNegativeReals

    b_tec.para_size_min = Param(domain=NonNegativeReals, initialize=size_min, mutable=True)
    b_tec.para_size_max = Param(domain=NonNegativeReals, initialize=size_max, mutable=True)
    if existing:
        b_tec.para_size_initial = Param(within=size_domain, initialize=size_initial)
    if existing and not decommission:
        # Decommissioning is not possible, size fixed
        b_tec.var_size = Param(within=size_domain, initialize=b_tec.para_size_initial)
    else:
        # Decommissioning is possible, size variable
            b_tec.var_size = Var(within=size_domain, bounds=(b_tec.para_size_min, b_tec.para_size_max))

    return b_tec

def define_capex(b_tec, tec_data, energyhub):
    """
    Defines variables and parameters related to technology capex.

    Parameters defined:
    - unit capex/ breakpoints for capex function

    Variables defined:
    - capex_aux (theoretical CAPEX for existing technologies)
    - CAPEX (actual CAPEX)
    - Decommissioning Costs (for existing technologies)
    """
    configuration = energyhub.configuration

    size_is_int = tec_data.size_is_int
    existing = tec_data.existing
    decommission = tec_data.decommission
    economics = tec_data.economics
    discount_rate = mc.set_discount_rate(configuration, economics)
    capex_model = mc.set_capex_model(configuration, economics)

    # CAPEX auxiliary (used to calculate theoretical CAPEX)
    # For new technologies, this is equal to actual CAPEX
    # For existing technologies it is used to calculate fixed OPEX
    b_tec.var_capex_aux = Var()
    annualization_factor = mc.annualize(discount_rate, economics.lifetime)
    if capex_model == 1:
        b_tec.para_unit_capex = Param(domain=Reals, initialize=economics.capex_data['unit_capex'], mutable=True)
        b_tec.para_unit_capex_annual = Param(domain=Reals, initialize=annualization_factor * economics.capex_data['unit_capex'], mutable=True)
        b_tec.const_capex_aux = Constraint(expr=b_tec.var_size * b_tec.para_unit_capex_annual == b_tec.var_capex_aux)
    elif capex_model == 2:
        b_tec.para_bp_x = Param(domain=Reals, initialize=economics.capex_data['piecewise_capex']['bp_x'])
        b_tec.para_bp_y = Param(domain=Reals, initialize=economics.capex_data['piecewise_capex']['bp_y'])
        b_tec.para_bp_y_annual = Param(domain=Reals, initialize=annualization_factor *
                                                                economics.capex_data['piecewise_capex']['bp_y'])
        global_variables.big_m_transformation_required = 1
        b_tec.const_capex_aux = Piecewise(b_tec.var_capex_aux, b_tec.var_size,
                                          pw_pts=b_tec.para_bp_x,
                                          pw_constr_type='EQ',
                                          f_rule=b_tec.para_bp_y_annual,
                                          pw_repn='SOS2')
    # CAPEX
    if existing and not decommission:
        b_tec.var_capex = Param(domain=Reals, initialize=0)
    else:
        b_tec.var_capex = Var()
        if existing:
            b_tec.para_decommissioning_cost = Param(domain=Reals, initialize=economics.decommission_cost, mutable = True)
            b_tec.const_capex = Constraint(
                expr=b_tec.var_capex == (b_tec.para_size_initial - b_tec.var_size) * b_tec.para_decommissioning_cost)
        else:
            b_tec.const_capex = Constraint(expr=b_tec.var_capex == b_tec.var_capex_aux)
    return b_tec

def define_input(b_tec, tec_data, energyhub):
    """
    Defines input to a technology

    var_input is always in full resolution
    var_input_aux can be in reduced resolution
    """
    # Technology related options
    existing = tec_data.existing
    performance_data = tec_data.performance_data
    fitted_performance = tec_data.fitted_performance
    technology_model = tec_data.technology_model
    modelled_with_full_res = tec_data.modelled_with_full_res

    # set_t and sequence
    set_t = energyhub.model.set_t_full
    if global_variables.clustered_data and not modelled_with_full_res:
        sequence = energyhub.data.k_means_specs.full_resolution['sequence']

    if existing:
        size_initial = tec_data.size_initial
        size_max = size_initial
    else:
        size_max = tec_data.size_max

    rated_power = fitted_performance.rated_power


    if (technology_model == 'RES') or (technology_model == 'CONV4') :
        b_tec.set_input_carriers = Set(initialize=[])
    else:
        b_tec.set_input_carriers = Set(initialize=performance_data['input_carrier'])
        def init_input_bounds(bounds, t, car):
            if global_variables.clustered_data and not modelled_with_full_res:
                return tuple(fitted_performance.bounds['input'][car][sequence[t - 1]-1, :] * size_max * rated_power)
            else:
                return tuple(fitted_performance.bounds['input'][car][t - 1, :] * size_max * rated_power)
        b_tec.var_input = Var(set_t, b_tec.set_input_carriers, within=NonNegativeReals,
                              bounds=init_input_bounds)
    return b_tec

def define_output(b_tec, tec_data, energyhub):
    """
    Defines output to a technology

    var_output is always in full resolution
    """
    # Technology related options
    existing = tec_data.existing
    performance_data = tec_data.performance_data
    fitted_performance = tec_data.fitted_performance
    modelled_with_full_res = tec_data.modelled_with_full_res

    rated_power = fitted_performance.rated_power

    # set_t
    set_t = energyhub.model.set_t_full
    if global_variables.clustered_data and not modelled_with_full_res:
        sequence = energyhub.data.k_means_specs.full_resolution['sequence']

    if existing:
        size_initial = tec_data.size_initial
        size_max = size_initial
    else:
        size_max = tec_data.size_max

    b_tec.set_output_carriers = Set(initialize=performance_data['output_carrier'])

    def init_output_bounds(bounds, t, car):
        if global_variables.clustered_data and not modelled_with_full_res:
            return tuple(fitted_performance.bounds['output'][car][sequence[t - 1]-1, :] * size_max * rated_power)
        else:
            return tuple(fitted_performance.bounds['output'][car][t - 1, :] * size_max * rated_power)
    b_tec.var_output = Var(set_t, b_tec.set_output_carriers, within=NonNegativeReals,
                           bounds=init_output_bounds)
    return b_tec

def define_opex(b_tec, tec_data, energyhub):
    """
    Defines variable and fixed OPEX
    """
    economics = tec_data.economics
    set_t = energyhub.model.set_t_full

    # VARIABLE OPEX
    b_tec.para_opex_variable = Param(domain=Reals, initialize=economics.opex_variable, mutable=True)
    b_tec.var_opex_variable = Var(set_t)
    def init_opex_variable(const, t):
        return sum(b_tec.var_output[t, car] for car in b_tec.set_output_carriers) * b_tec.para_opex_variable == \
               b_tec.var_opex_variable[t]
    b_tec.const_opex_variable = Constraint(set_t, rule=init_opex_variable)

    # FIXED OPEX
    b_tec.para_opex_fixed = Param(domain=Reals, initialize=economics.opex_fixed, mutable=True)
    b_tec.var_opex_fixed = Var()
    b_tec.const_opex_fixed = Constraint(expr=b_tec.var_capex_aux * b_tec.para_opex_fixed == b_tec.var_opex_fixed)
    return b_tec

def define_emissions(b_tec, tec_data, energyhub):
    """
    Defines Emissions
    """

    set_t = energyhub.model.set_t_full
    performance_data = tec_data.performance_data
    technology_model = tec_data.technology_model
    emissions_based_on = tec_data.emissions_based_on

    b_tec.para_tec_emissionfactor = Param(domain=Reals, initialize=performance_data['emission_factor'])
    b_tec.var_tec_emissions_pos = Var(set_t, within=NonNegativeReals)
    b_tec.var_tec_emissions_neg = Var(set_t, within=NonNegativeReals)

    if technology_model == 'RES':
        # Set emissions to zero
        def init_tec_emissions_pos(const, t):
            return b_tec.var_tec_emissions_pos[t] == 0
        b_tec.const_tec_emissions_pos = Constraint(set_t, rule=init_tec_emissions_pos)

        def init_tec_emissions_neg(const, t):
            return b_tec.var_tec_emissions_neg[t] == 0
        b_tec.const_tec_emissions_neg = Constraint(set_t, rule=init_tec_emissions_neg)

    else:

        if emissions_based_on == 'output':
            def init_tec_emissions_pos(const, t):
                if performance_data['emission_factor'] >= 0:
                    return b_tec.var_output[t, performance_data['main_output_carrier']] * \
                           b_tec.para_tec_emissionfactor == \
                           b_tec.var_tec_emissions_pos[t]
                else:
                    return b_tec.var_tec_emissions_pos[t] == 0

            b_tec.const_tec_emissions_pos = Constraint(set_t, rule=init_tec_emissions_pos)

            def init_tec_emissions_neg(const, t):
                if performance_data['emission_factor'] < 0:
                    return b_tec.var_output[t, performance_data['main_output_carrier']] * \
                           (-b_tec.para_tec_emissionfactor) == \
                           b_tec.var_tec_emissions_neg[t]
                else:
                    return b_tec.var_tec_emissions_neg[t] == 0

            b_tec.const_tec_emissions_neg = Constraint(set_t, rule=init_tec_emissions_neg)

        elif emissions_based_on == 'input':
            def init_tec_emissions_pos(const, t):
                if performance_data['emission_factor'] >= 0:
                    return b_tec.var_input[t, performance_data['main_input_carrier']] \
                           * b_tec.para_tec_emissionfactor \
                           == b_tec.var_tec_emissions_pos[t]
                else:
                    return b_tec.var_tec_emissions_pos[t] == 0
            b_tec.const_tec_emissions_pos = Constraint(set_t, rule=init_tec_emissions_pos)

            def init_tec_emissions_neg(const, t):
                if performance_data['emission_factor'] < 0:
                    return b_tec.var_input[t, performance_data['main_input_carrier']] \
                               (-b_tec.para_tec_emissionfactor) == \
                           b_tec.var_tec_emissions_neg[t]
                else:
                    return b_tec.var_tec_emissions_neg[t] == 0
            b_tec.const_tec_emissions_neg = Constraint(set_t, rule=init_tec_emissions_neg)

    return b_tec

def define_auxiliary_vars(b_tec, tec_data, energyhub):
    """
    Defines auxiliary variables, that are required for the modelling of clustered data 
    """
    set_t_clustered = energyhub.model.set_t_clustered
    set_t_full = energyhub.model.set_t_full
    fitted_performance = tec_data.fitted_performance
    technology_model = tec_data.technology_model
    existing = tec_data.existing
    if existing:
        size_initial = tec_data.size_initial
        size_max = size_initial
    else:
        size_max = tec_data.size_max

    rated_power = fitted_performance.rated_power

    sequence = energyhub.data.k_means_specs.full_resolution['sequence']

    def init_input_bounds(bounds, t, car):
            return tuple(fitted_performance.bounds['input'][car][t - 1, :] * size_max * rated_power)
    b_tec.var_input_aux = Var(set_t_clustered, b_tec.set_input_carriers, within=NonNegativeReals,
                          bounds=init_input_bounds)

    b_tec.const_link_full_resolution_input = mc.link_full_resolution_to_clustered(b_tec.var_input_aux,
                                                                               b_tec.var_input,
                                                                               set_t_full,
                                                                               sequence,
                                                                               b_tec.set_input_carriers)

    def init_output_bounds(bounds, t, car):
        return tuple(fitted_performance.bounds['output'][car][t - 1, :] * size_max * rated_power)
    b_tec.var_output_aux = Var(set_t_clustered, b_tec.set_output_carriers, within=NonNegativeReals,
                              bounds=init_output_bounds)

    b_tec.const_link_full_resolution_output = mc.link_full_resolution_to_clustered(b_tec.var_output_aux,
                                                                                  b_tec.var_output,
                                                                                  set_t_full,
                                                                                  sequence,
                                                                                  b_tec.set_output_carriers)
    
    return b_tec

def add_technology(energyhub, nodename, set_tecsToAdd):
    r"""
    Adds all technologies as model blocks to respective node.

    This function initializes parameters and decision variables for all technologies at respective node.
    For each technology, it adds one block indexed by the set of all technologies at the node :math:`S_n`.
    This function adds Sets, Parameters, Variables and Constraints that are common for all technologies.
    For each technology type, individual parts are added. The following technology types of generic technologies
    are currently available
    (all contained in :func:`src.model_construction.technology_constraints.generic_technology_constraints`):

    - Type RES: Renewable technology with cap_factor as input.
    - Type CONV1: n inputs -> n output, fuel and output substitution.
    - Type CONV2: n inputs -> n output, fuel substitution.
    - Type CONV2: n inputs -> n output, no fuel and output substitution.
    - Type STOR: Storage technology (1 input -> 1 output).

    Additionally, the following specific technologies are available:

    - Type DAC_adsorption: Direct Air Capture technology (adsorption).
    - Type Heat_Pump: Three different types of heat pumps
    - Type Gas_Turbine: Different types/sizes of gas turbines

    The following description is true for new technologies. For existing technologies a few adaptions are made
    (see below).

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

    Existing technologies, i.e. existing = 1, can be decommissioned (decommission = 1) or not (decommission = 0).
    For technologies that cannot be decommissioned, the size is fixed to the size given in the technology data.
    For technologies that can be decommissioned, the size can be smaller or equal to the initial size. Reducing the
    size comes at the decommissioning costs specified in the economics of the technology.
    The fixed opex is calculated by determining the capex that the technology would have costed if newly build and
    then taking the respective opex_fixed share of this. This is done with the auxiliary variable var_capex_aux.

    :param str nodename: name of node for which technology is installed
    :param set set_tecsToAdd: list of technologies to add
    :param energyhub EnergyHub: instance of the energyhub
    :return: b_node
    """

    # COLLECT OBJECTS FROM ENERGYHUB
    data = energyhub.data

    def init_technology_block(b_tec, tec):
        print('\t - Adding Technology ' + tec)

        # TECHNOLOGY DATA
        tec_data = data.technology_data[nodename][tec]
        technology_model = tec_data.technology_model
        modelled_with_full_res = tec_data.modelled_with_full_res

        # SIZE
        b_tec = define_size(b_tec, tec_data)
        
        # CAPEX
        b_tec = define_capex(b_tec, tec_data, energyhub)

        # INPUT AND OUTPUT
        b_tec = define_input(b_tec, tec_data, energyhub)
        b_tec = define_output(b_tec, tec_data, energyhub)

        # OPEX
        b_tec = define_opex(b_tec, tec_data, energyhub)

        # EMISSIONS
        b_tec = define_emissions(b_tec, tec_data, energyhub)

        # DEFINE AUXILIARY VARIABLES FOR CLUSTERED DATA
        if global_variables.clustered_data and not modelled_with_full_res:
            b_tec = define_auxiliary_vars(b_tec, tec_data, energyhub)

        # GENERIC TECHNOLOGY CONSTRAINTS
        if technology_model == 'RES': # Renewable technology with cap_factor as input
            b_tec = constraints_tec_RES(b_tec, tec_data, energyhub)

        elif technology_model == 'CONV1': # n inputs -> n output, fuel and output substitution
            b_tec = constraints_tec_CONV1(b_tec, tec_data, energyhub)

        elif technology_model == 'CONV2': # n inputs -> n output, fuel and output substitution
            b_tec = constraints_tec_CONV2(b_tec, tec_data, energyhub)

        elif technology_model == 'CONV3':  # n input -> n outputs, output flexible
            b_tec = constraints_tec_CONV3(b_tec, tec_data, energyhub)

        elif technology_model == 'CONV4':  # no input -> n outputs, fixed output ratios
            b_tec = constraints_tec_CONV4(b_tec, tec_data, energyhub)

        elif technology_model == 'STOR': # Storage technology (1 input -> 1 output)
            b_tec = constraints_tec_STOR(b_tec, tec_data, energyhub)

        # SPECIFIC TECHNOLOGY CONSTRAINTS
        elif technology_model == 'DAC_Adsorption':
            b_tec = constraints_tec_dac_adsorption(b_tec, tec_data, energyhub)

        elif technology_model.startswith('HeatPump_'):  # Heat Pump
            b_tec = constraints_tec_hp(b_tec, tec_data, energyhub)

        elif technology_model.startswith('GasTurbine_'):  # Gas Turbine
            b_tec = constraints_tec_gt(b_tec, tec_data, energyhub)

        if global_variables.big_m_transformation_required:
            mc.perform_disjunct_relaxation(b_tec)

        return b_tec

    # Create a new block containing all new technologies.
    b_node = energyhub.model.node_blocks[nodename]

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