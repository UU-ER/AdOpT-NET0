import numbers
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

    - Type 1: Renewable technology with cap_factor as input. Constructed with \
      :func:`src.model_construction.generic_technology_constraints.constraints_tec_type_1`
    - Type 2: n inputs -> n output, fuel and output substitution. Constructed with \
      :func:`src.model_construction.generic_technology_constraints.constraints_tec_type_2`
    - Type 3: n inputs -> n output, fuel and output substitution. Constructed with \
      :func:`src.model_construction.generic_technology_constraints.constraints_tec_type_3`
    - Type 6: Storage technology (1 input -> 1 output). Constructed with \
      :func:`src.model_construction.generic_technology_constraints.constraints_tec_type_6`

    **Set declarations:**

    - Set of input carriers
    - Set of output carriers

    **Parameter declarations:**

    - Min Size
    - Max Size
    - Output max (same as size max)
    - Unit CAPEX
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

        # region Get options from data
        tec_data = data.technology_data[nodename][tec]
        tec_type = tec_data['TechnologyPerf']['tec_type']
        capex_model = tec_data['Economics']['CAPEX_model']
        size_is_integer = tec_data['TechnologyPerf']['size_is_int']
        # endregion

        # region PARAMETERS

        # We need this shit because python does not accept single value in its build-in min function
        if isinstance(tec_data['TechnologyPerf']['size_min'], numbers.Number):
            size_min = tec_data['TechnologyPerf']['size_min']
        else:
            size_min = min(tec_data['TechnologyPerf']['size_min'])
        if isinstance(tec_data['TechnologyPerf']['size_max'], numbers.Number):
            size_max = tec_data['TechnologyPerf']['size_max']
        else:
            size_max = max(tec_data['TechnologyPerf']['size_max'])

        if size_is_integer:
            unit_size = u.dimensionless
        else:
            unit_size = u.MW
        b_tec.para_size_min = Param(domain=NonNegativeReals, initialize=size_min, units=unit_size)
        b_tec.para_size_max = Param(domain=NonNegativeReals, initialize=size_max, units=unit_size)
        b_tec.para_output_max = Param(domain=NonNegativeReals, initialize=size_max, units=u.MW)
        b_tec.para_unit_CAPEX = Param(domain=Reals, initialize=tec_data['Economics']['unit_CAPEX_annual'],
                                      units=u.EUR/unit_size)
        b_tec.para_OPEX_variable = Param(domain=Reals, initialize=tec_data['Economics']['OPEX_variable'],
                                         units=u.EUR/u.MWh)
        b_tec.para_OPEX_fixed = Param(domain=Reals, initialize=tec_data['Economics']['OPEX_fixed'],
                                      units=u.EUR/u.EUR)

        #TODO: check unit tonnes
        b_tec.para_tec_emissionfactor = Param(domain=Reals, initialize=tec_data['TechnologyPerf']['emission_factor'],
                                      units=u.t/u.MWh)

        # endregion

        # region SETS
        b_tec.set_input_carriers = Set(initialize=tec_data['TechnologyPerf']['input_carrier'])
        b_tec.set_output_carriers = Set(initialize=tec_data['TechnologyPerf']['output_carrier'])
        # endregion

        # region DECISION VARIABLES
        # Input
        if not tec_type == 1:
            b_tec.var_input = Var(model.set_t, b_tec.set_input_carriers, within=NonNegativeReals,
                                  bounds=(b_tec.para_size_min, b_tec.para_size_max), units=u.MW)
        # Output
        b_tec.var_output = Var(model.set_t, b_tec.set_output_carriers, within=NonNegativeReals,
                               bounds=(0, b_tec.para_output_max), units=u.MW)

        # Emission
        #TODO: check bounds and units
        b_tec.var_tec_emissions = Var(model.set_t, within=NonNegativeReals, bounds=(0, b_tec.para_size_max), units=u.t)
        b_tec.var_tec_emissions_neg = Var(model.set_t, within=NonNegativeReals, bounds=(0, b_tec.para_size_max), units=u.t)

        # Size
        if size_is_integer:  # size
            b_tec.var_size = Var(within=NonNegativeIntegers, bounds=(b_tec.para_size_min, b_tec.para_size_max))
        else:
            b_tec.var_size = Var(within=NonNegativeReals, bounds=(b_tec.para_size_min, b_tec.para_size_max),
                                 units=u.MW)
        # Capex/Opex
        b_tec.var_CAPEX = Var(units=u.EUR)  # capex
        b_tec.var_OPEX_variable = Var(model.set_t, units=u.EUR)  # variable opex
        b_tec.var_OPEX_fixed = Var(units=u.EUR)  # fixed opex
        # endregion

        # region GENERAL CONSTRAINTS
        # Capex
        if capex_model == 1:
            b_tec.const_CAPEX = Constraint(expr=b_tec.var_size * b_tec.para_unit_CAPEX == b_tec.var_CAPEX)
        elif capex_model == 2:
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
        #TODO: check main input carrier and sum, difference between emissions and neg emissions
        def init_tec_emissions(const, t):
            return sum(b_tec.var_input[t, tec_data['TechnologyPerf']['main_input_carrier']] for t in model.set_t) \
                * b_tec.para_tec_emissionfactor == b_tec.var_tec_emissions
        b_tec.const_tec_emissions = Constraint(rule=init_tec_emissions)
        # def init_tec_emissions_neg(const, t):
        #     return sum(b_tec.var_input[t, tec_data['TechnologyPerf']['main_input_carrier']] for t in model.set_t) \
        #         * b_tec.para_tec_emissionfactor == b_tec.var_tec_emissions
        # b_tec.const_tec_emissions_neg = Constraint(rule=init_tec_emissions_neg)

        # Size constraint
        if tec_type == 1: # we don't need size constraints for renewable technologies
            pass
        elif tec_type == 6: # This is defined in the generic technology constraints
            pass
        else: # in terms of input
            def init_output_constraint(const, t):
                return sum(b_tec.var_input[t, car_input] for car_input in b_tec.set_input_carriers) \
                       <= b_tec.var_size
            b_tec.const_size = Constraint(model.set_t, rule=init_output_constraint)

        # endregion

        # region TECHNOLOGY TYPES
        if tec_type == 1: # Renewable technology with cap_factor as input
            b_tec = constraints_tec_type_1(model, b_tec,tec_data)

        elif tec_type == 2: # n inputs -> n output, fuel and output substitution
            b_tec = constraints_tec_type_2(model, b_tec, tec_data)

        elif tec_type == 3: # n inputs -> n output, fuel and output substitution
            b_tec = constraints_tec_type_3(model, b_tec, tec_data)

        # elif tectype == 4:  # 1 input -> n outputs, output flexible, linear performance
        # elif tectype == 5:  # 1 input -> n outputs, fixed output ratio, linear performance

        elif tec_type == 6: # Storage technology (1 input -> 1 output)
            b_tec = constraints_tec_type_6(model, b_tec, tec_data)

        if m_config.presolve.big_m_transformation_required:
            mc.perform_disjunct_relaxation(b_tec)

        return b_tec

    # Create a new block containing all new technologies. The set of nodes that need to be added
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
