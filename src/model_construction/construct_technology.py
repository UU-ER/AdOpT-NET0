import numpy as np
from pyomo.environ import *
from pyomo.environ import units as u
import src.global_variables as global_variables
import src.model_construction as mc
from src.model_construction.technology_constraints import *


def add_technologies(energyhub, nodename, set_tecsToAdd):
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
    - Type CONV2: n inputs -> n output, fuel substitution. Constructed with
    - Type CONV2: n inputs -> n output, no fuel and output substitution.
    - Type STOR: Storage technology (1 input -> 1 output).

    Additionally, the following specific technologies are available:

    - Type DAC_adsorption: Direct Air Capture technology (adsorption).

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
    :param list set_tecsToAdd: list of technologies to add
    :param EnergyHub energyhub: instance of the energyhub
    :return: b_node
    ----------
    """

    # COLLECT OBJECTS FROM ENERGYHUB
    data = energyhub.data
    model = energyhub.model

    def init_technology_block(b_tec, tec):

        # TECHNOLOGY DATA
        tec_data = data.technology_data[nodename][tec]
        technology_model = tec_data.technology_model
        existing = tec_data.existing
        decommission = tec_data.decommission
        size_is_int = tec_data.size_is_int
        size_min = tec_data.size_min
        size_max = tec_data.size_max
        economics = tec_data.economics
        performance_data = tec_data.performance_data
        fitted_performance = tec_data.fitted_performance

        if existing:
            size_initial = tec_data.size_initial
            size_max = size_initial

        # SIZE
        if size_is_int:
            unit_size = u.dimensionless
        else:
            unit_size = u.MW
        b_tec.para_size_min = Param(domain=NonNegativeReals, initialize=size_min, units=unit_size)
        b_tec.para_size_max = Param(domain=NonNegativeReals, initialize=size_max, units=unit_size)

        if existing:
            b_tec.var_size_initial = Param(within=NonNegativeReals, initialize=size_initial, units=unit_size)

        if existing and not decommission:
            # Decommissioning is not possible, size fixed
            b_tec.var_size = Param(within=NonNegativeReals, initialize=b_tec.var_size_initial, units=unit_size)
        else:
            # Decommissioning is possible, size variable
            if size_is_int:
                b_tec.var_size = Var(within=NonNegativeIntegers, bounds=(b_tec.para_size_min, b_tec.para_size_max))
            else:
                b_tec.var_size = Var(within=NonNegativeReals, bounds=(b_tec.para_size_min, b_tec.para_size_max),
                                     units=u.MW)

        # CAPEX auxilliary (used to calculate theoretical CAPEX)
        # For new technologies, this is equal to actual CAPEX
        # For existing technologies it is used to calculate fixed OPEX
        b_tec.var_CAPEX_aux = Var(units=u.EUR)
        annualization_factor = mc.annualize(economics.discount_rate, economics.lifetime)
        if economics.capex_model == 1:
            b_tec.para_unit_CAPEX = Param(domain=Reals, initialize=economics.capex_data['unit_capex'],
                                          units=u.EUR/unit_size)
            b_tec.para_unit_CAPEX_annual = Param(domain=Reals,
                                                 initialize= annualization_factor * economics.capex_data['unit_capex'],
                                                 units=u.EUR/unit_size)
            b_tec.const_CAPEX_aux = Constraint(expr=b_tec.var_size * b_tec.para_unit_CAPEX_annual == b_tec.var_CAPEX_aux)
        elif economics.capex_model == 2:
            b_tec.para_bp_x = Param(domain=Reals, initialize=economics.capex_data['piecewise_capex']['bp_x'],
                                    units=unit_size)
            b_tec.para_bp_y = Param(domain=Reals, initialize=economics.capex_data['piecewise_capex']['bp_y'],
                                    units=u.EUR/unit_size)
            b_tec.para_bp_y_annual = Param(domain=Reals, initialize=annualization_factor *
                                                                    economics.capex_data['piecewise_capex']['bp_y'],
                                           units=u.EUR/unit_size)
            global_variables.big_m_transformation_required = 1
            b_tec.const_CAPEX_aux = Piecewise(b_tec.var_CAPEX_aux, b_tec.var_size,
                                              pw_pts=b_tec.para_bp_x,
                                              pw_constr_type='EQ',
                                              f_rule=b_tec.para_bp_y,
                                              pw_repn='SOS2')

        # CAPEX
        if existing and not decommission:
            b_tec.var_CAPEX = Param(domain=Reals, initialize=0, units=u.EUR)
        else:
            b_tec.var_CAPEX = Var(units=u.EUR)
            if existing:
                b_tec.para_decommissioning_cost = Param(domain=Reals, initialize=economics.decommission_cost, units=u.EUR/unit_size)
                b_tec.const_CAPEX = Constraint(expr= b_tec.var_CAPEX == (b_tec.var_size_initial - b_tec.var_size) * b_tec.para_decommissioning_cost)
            else:
                b_tec.const_CAPEX = Constraint(expr= b_tec.var_CAPEX == b_tec.var_CAPEX_aux)

        # INPUT
        if technology_model == 'RES':
            b_tec.set_input_carriers = Set(initialize=[])
        else:
            b_tec.set_input_carriers = Set(initialize=performance_data['input_carrier'])
            def init_input_bounds(bounds, t, car):
                return tuple(fitted_performance['input_bounds'][car][t-1,:] * size_max)
            b_tec.var_input = Var(model.set_t, b_tec.set_input_carriers, within=NonNegativeReals,
                                  bounds=init_input_bounds, units=u.MW)

        # OUTPUT
        b_tec.set_output_carriers = Set(initialize=performance_data['output_carrier'])
        def init_output_bounds(bounds, t, car):
            return tuple(fitted_performance['output_bounds'][car][t-1,:] * size_max)
        b_tec.var_output = Var(model.set_t, b_tec.set_output_carriers, within=NonNegativeReals,
                               bounds=init_output_bounds, units=u.MW)

        # VARIABLE OPEX
        b_tec.para_OPEX_variable = Param(domain=Reals, initialize=economics.opex_variable,
                                         units=u.EUR/u.MWh)
        b_tec.var_OPEX_variable = Var(model.set_t, units=u.EUR)
        def init_OPEX_variable(const, t):
            return sum(b_tec.var_output[t, car] for car in b_tec.set_output_carriers) * b_tec.para_OPEX_variable == \
                   b_tec.var_OPEX_variable[t]
        b_tec.const_OPEX_variable = Constraint(model.set_t, rule=init_OPEX_variable)

        # FIXED OPEX
        b_tec.para_OPEX_fixed = Param(domain=Reals, initialize=economics.opex_fixed,
                                      units=u.EUR/u.EUR)
        b_tec.var_OPEX_fixed = Var(units=u.EUR)
        b_tec.const_OPEX_fixed = Constraint(expr=b_tec.var_CAPEX_aux * b_tec.para_OPEX_fixed == b_tec.var_OPEX_fixed)


        # EMISSIONS
        b_tec.para_tec_emissionfactor = Param(domain=Reals, initialize=performance_data['emission_factor'],
                                              units=u.t/u.MWh)
        b_tec.var_tec_emissions_pos = Var(model.set_t, within=NonNegativeReals, units=u.t)
        b_tec.var_tec_emissions_neg = Var(model.set_t, within=NonNegativeReals, units=u.t)

        if technology_model == 'RES':
            # Set emissions to zero
            def init_tec_emissions_pos(const, t):
                return b_tec.var_tec_emissions_pos[t] == 0
            b_tec.const_tec_emissions_pos = Constraint(model.set_t, rule=init_tec_emissions_pos)
            def init_tec_emissions_neg(const, t):
                return b_tec.var_tec_emissions_neg[t] == 0
            b_tec.const_tec_emissions_neg = Constraint(model.set_t, rule=init_tec_emissions_neg)
        elif technology_model == 'DAC_adsorption':
            # Based on output
            def const_tec_emissions_pos(const, t):
                return b_tec.var_tec_emissions_pos[t] == 0
            b_tec.const_tec_emissions_pos = Constraint(model.set_t, rule=const_tec_emissions_pos)
            def init_tec_emissions_neg(const, t):
                return b_tec.var_output[t, performance_data['output_carrier']] \
                           (-b_tec.para_tec_emissionfactor) == \
                       b_tec.var_tec_emissions_neg[t]
            b_tec.const_tec_emissions_neg = Constraint(model.set_t, rule=init_tec_emissions_neg)
        else:
            # Based on input
            def init_tec_emissions_pos(const, t):
                if performance_data['emission_factor'] >= 0:
                    return b_tec.var_input[t, performance_data['main_input_carrier']] \
                           * b_tec.para_tec_emissionfactor \
                           == b_tec.var_tec_emissions_pos[t]
                else:
                    return b_tec.var_tec_emissions_pos[t] == 0
            b_tec.const_tec_emissions_pos = Constraint(model.set_t, rule=init_tec_emissions_pos)

            def init_tec_emissions_neg(const, t):
                if performance_data['emission_factor'] < 0:
                    return b_tec.var_input[t, performance_data['main_input_carrier']] \
                               (-b_tec.para_tec_emissionfactor) == \
                           b_tec.var_tec_emissions_neg[t]
                else:
                    return b_tec.var_tec_emissions_neg[t] == 0
            b_tec.const_tec_emissions_neg = Constraint(model.set_t, rule=init_tec_emissions_neg)


        # GENERIC TECHNOLOGY CONSTRAINTS
        if technology_model == 'RES': # Renewable technology with cap_factor as input
            b_tec = constraints_tec_RES(model, b_tec, tec_data)

        elif technology_model == 'CONV1': # n inputs -> n output, fuel and output substitution
            b_tec = constraints_tec_CONV1(model, b_tec, tec_data)

        elif technology_model == 'CONV2': # n inputs -> n output, fuel and output substitution
            b_tec = constraints_tec_CONV2(model, b_tec, tec_data)

        elif technology_model == 'CONV3':  # 1 input -> n outputs, output flexible, linear performance
            b_tec = constraints_tec_CONV3(model, b_tec, tec_data)

        elif technology_model == 'STOR': # Storage technology (1 input -> 1 output)
            if global_variables.clustered_data == 1:
                hourly_order = data.k_means_specs.full_resolution['hourly_order']
            else:
                hourly_order = np.arange(1, len(model.set_t)+1)
            b_tec = constraints_tec_STOR(model, b_tec, tec_data, hourly_order)

        # SPECIFIC TECHNOLOGY CONSTRAINTS
        elif technology_model == 'DAC_adsorption':
            b_tec = constraints_tec_dac_adsorption(model, b_tec, tec_data)

        elif technology_model.startswith('HeatPump_'):  # Heat Pump
            b_tec = constraints_tec_hp(model, b_tec, tec_data)

        elif technology_model.startswith('GasTurbine_'):  # Gas Turbine
            b_tec = constraints_tec_gt(model, b_tec, tec_data)

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



