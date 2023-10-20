from ..model_construction import add_technology

import numpy as np
from pyomo.environ import *


def determine_carriers_at_node(energyhub, node):
    """
    Determines carriers that are used at respective node
    """
    # COLLECT OBJECTS FROM ENERGYHUB
    data = energyhub.data
    model = energyhub.model

    # Carriers that need to be considered at node
    carriers = []

    # From time series
    time_series = ['demand', 'production_profile', 'import_limit', 'export_limit']
    for series in time_series:
        for car in data.node_data[node].data[series]:
            if np.any(data.node_data[node].data[series][car]):
                carriers.append(car)

    # From technologies
    for tec in model.set_technologies[node]:
        if 'input_carrier' in data.technology_data[node][tec].performance_data:
            input_carriers = data.technology_data[node][tec].performance_data['input_carrier']
            carriers.extend(input_carriers)
        output_carriers = data.technology_data[node][tec].performance_data['output_carrier']
        carriers.extend(output_carriers)

    # From networks
    if not energyhub.configuration.energybalance.copperplate:
        for netw in model.set_networks:
            # This can be further extended to check if node is connected to network
            for car in model.network_block[netw].set_netw_carrier:
                carriers.append(car)
            if hasattr(model.network_block[netw], 'set_consumed_carriers'):
                for car in model.network_block[netw].set_consumed_carriers:
                    carriers.append(car)

    return carriers

def determine_network_energy_consumption(energyhub):
    """
    Determines if there is network consumption for a network
    """
    # COLLECT OBJECTS FROM ENERGYHUB
    data = energyhub.data
    model = energyhub.model

    # From networks
    # This can be further extended to check if node is connected to network
    network_energy_consumption = 0
    if not energyhub.configuration.energybalance.copperplate:
        for netw in model.set_networks:
            if hasattr(model.network_block[netw], 'set_consumed_carriers'):
                network_energy_consumption = 1
        return network_energy_consumption

def add_nodes(energyhub):
    r"""
    Adds all nodes with respective data to the model

    This function initializes parameters and decision variables for all considered nodes. It also adds all technologies\
    that are installed at the node (see :func:`~add_technologies`). For each node, it adds one block indexed by the \
    set of all nodes. As such, the function constructs:

    node blocks, indexed by :math:`N` > technology blocks, indexed by :math:`Tec_n, n \in N`

    **Set declarations:**

    - Set for all technologies :math:`S_n` at respective node :math:`n` : :math:`S_n, n \in N` (this is a duplicate \
      of a set already initialized in ``self.model.set_technologies``).

    **Parameter declarations:**

    - Demand for each time step
    - Import Prices for each time step
    - Export Prices for each time step
    - Import Limits for each time step
    - Export Limits for each time step
    - Emission Factors for each time step

    **Variable declarations:**

    - Import Flow for each time step
    - Export Flow for each time step
    - Network Inflow for each time step
    - Network Outflow for each time step
    - Cost at node (includes technology costs (CAPEX, OPEX) and import/export costs), see constraint declarations

    **Constraint declarations**

    - Cost at node:

    .. math::
        C_n = \
        \sum_{tec \in Tec_n} CAPEX_{tec} + \
        \sum_{tec \in Tec_n} OPEXfix_{tec} + \
        \sum_{tec \in Tec_n} \sum_{t \in T} OPEXvar_{t, tec} + \\
        \sum_{car \in Car} \sum_{t \in T} import_{t, car} pImport_{t, car} - \
        \sum_{car \in Car} \sum_{t \in T} export_{t, car} pExport_{t, car}

    **Block declarations:**

    - Technologies at node

    :param EnergyHub energyhub: instance of the energyhub
    :return: model
    """

    # COLLECT OBJECTS FROM ENERGYHUB
    data = energyhub.data
    model = energyhub.model

    def init_node_block(b_node, nodename):
        print('_' * 60)
        print('--- Adding Node ' + nodename + '... ---')

        # SETS: Get technologies for each node and make it a set for the block
        carriers = determine_carriers_at_node(energyhub, nodename)
        network_energy_consumption = determine_network_energy_consumption(energyhub)
        b_node.set_carriers = Set(initialize=list(set(carriers)))
        b_node.set_tecsAtNode = Set(initialize=model.set_technologies[nodename])

        set_t = model.set_t_full
        node_data = data.node_data[nodename]

        # PARAMETERS
        # Demand
        def init_demand(para, t, car):
            return node_data.data['demand'][car][t - 1]
        b_node.para_demand = Param(set_t, b_node.set_carriers, rule=init_demand, mutable=True)

        # Generic production profile
        def init_production_profile(para, t, car):
                return node_data.data['production_profile'][car][t - 1]
        b_node.para_production_profile = Param(set_t, b_node.set_carriers, rule=init_production_profile, mutable=True)

        # Import Prices
        def init_import_price(para, t, car):
            if nodename in data.node_data:
                return node_data.data['import_prices'][car][t - 1]
        b_node.para_import_price = Param(set_t, b_node.set_carriers, rule=init_import_price, mutable=True)

        # Export Prices
        def init_export_price(para, t, car):
            if nodename in data.node_data:
                return node_data.data['export_prices'][car][t - 1]
        b_node.para_export_price = Param(set_t, b_node.set_carriers, rule=init_export_price, mutable=True)

        # Import Limit
        def init_import_limit(para, t, car):
            if nodename in data.node_data:
                return node_data.data['import_limit'][car][t - 1]
        b_node.para_import_limit = Param(set_t, b_node.set_carriers, rule=init_import_limit)

        # Export Limit
        def init_export_limit(para, t, car):
            if nodename in data.node_data:
                return node_data.data['export_limit'][car][t - 1]
        b_node.para_export_limit = Param(set_t, b_node.set_carriers, rule=init_export_limit)

        # Emission Factor
        def init_import_emissionfactor(para, t, car):
            if nodename in data.node_data:
                return node_data.data['import_emissionfactors'][car][t - 1]
        b_node.para_import_emissionfactors = Param(set_t, b_node.set_carriers, rule=init_import_emissionfactor, mutable=True)

        def init_export_emissionfactor(para, t, car):
            if nodename in data.node_data:
                return node_data.data['export_emissionfactors'][car][t - 1]
        b_node.para_export_emissionfactors = Param(set_t, b_node.set_carriers, rule=init_export_emissionfactor, mutable=True)

        # DECISION VARIABLES
        # Interaction with network/system boundaries
        def init_import_bounds(var, t, car):
            return (0, b_node.para_import_limit[t, car])
        b_node.var_import_flow = Var(set_t, b_node.set_carriers, bounds=init_import_bounds)

        def init_export_bounds(var, t, car):
            return (0, b_node.para_export_limit[t, car])
        b_node.var_export_flow = Var(set_t, b_node.set_carriers, bounds=init_export_bounds)

        b_node.var_netw_inflow = Var(set_t, b_node.set_carriers)
        b_node.var_netw_outflow = Var(set_t, b_node.set_carriers)

        if network_energy_consumption:
            b_node.var_netw_consumption = Var(set_t, b_node.set_carriers)

        # Generic production profile
        b_node.var_generic_production = Var(set_t, b_node.set_carriers, within=NonNegativeReals)

        # Emissions
        b_node.var_import_emissions_pos = Var(set_t, b_node.set_carriers)
        b_node.var_import_emissions_neg = Var(set_t, b_node.set_carriers)
        b_node.var_export_emissions_pos = Var(set_t, b_node.set_carriers)
        b_node.var_export_emissions_neg = Var(set_t, b_node.set_carriers)
        b_node.var_car_emissions_pos = Var(set_t, within=NonNegativeReals)
        b_node.var_car_emissions_neg = Var(set_t, within=NonNegativeReals)

        # CONSTRAINTS
        # Generic production constraint
        def init_generic_production(const, t, car):
            if node_data.options.production_profile_curtailment[car] == 0:
                return b_node.para_production_profile[t, car] == b_node.var_generic_production[t, car]
            elif node_data.options.production_profile_curtailment[car] == 1:
                return b_node.para_production_profile[t, car] >= b_node.var_generic_production[t, car]
        b_node.const_generic_production = Constraint(set_t, b_node.set_carriers, rule=init_generic_production)

        # Emission constraints
        def init_import_emissions_pos(const, t, car):
            if node_data.data['import_emissionfactors'][car][t - 1] >= 0:
                return b_node.var_import_flow[t, car] * b_node.para_import_emissionfactors[t, car] \
                    == b_node.var_import_emissions_pos[t, car]
            else:
                return 0 == b_node.var_import_emissions_pos[t, car]
        b_node.const_import_emissions_pos = Constraint(set_t, b_node.set_carriers,
                                                       rule=init_import_emissions_pos)

        def init_export_emissions_pos(const, t, car):
            if node_data.data['export_emissionfactors'][car][t - 1] >= 0:
                return b_node.var_export_flow[t, car] * b_node.para_export_emissionfactors[t, car] \
                    == b_node.var_export_emissions_pos[t, car]
            else:
                return 0 == b_node.var_export_emissions_pos[t, car]
        b_node.const_export_emissions_pos = Constraint(set_t, b_node.set_carriers, rule=init_export_emissions_pos)

        def init_import_emissions_neg(const, t, car):
            if node_data.data['import_emissionfactors'][car][t - 1] < 0:
                return b_node.var_import_flow[t, car] * (-b_node.para_import_emissionfactors[t, car]) \
                    == b_node.var_import_emissions_neg[t, car]
            else:
                return 0 == b_node.var_import_emissions_neg[t, car]
        b_node.const_import_emissions_neg = Constraint(set_t, b_node.set_carriers,
                                                       rule=init_import_emissions_neg)

        def init_export_emissions_neg(const, t, car):
            if node_data.data['export_emissionfactors'][car][t - 1] < 0:
                return b_node.var_export_flow[t, car] * (-b_node.para_export_emissionfactors[t, car]) \
                    == b_node.var_export_emissions_neg[t, car]
            else:
                return 0 == b_node.var_export_emissions_neg[t, car]
        b_node.const_export_emissions_neg = Constraint(set_t, b_node.set_carriers,
                                                       rule=init_export_emissions_neg)

        def init_car_emissions_pos(const, t):
            return sum(b_node.var_import_emissions_pos[t, car] + b_node.var_export_emissions_pos[t, car]
                    for car in b_node.set_carriers) \
                   == b_node.var_car_emissions_pos[t]
        b_node.const_car_emissions_pos = Constraint(set_t, rule=init_car_emissions_pos)

        def init_car_emissions_neg(const, t):
            return sum(b_node.var_import_emissions_neg[t, car] + b_node.var_export_emissions_neg[t, car]
                    for car in b_node.set_carriers) == \
                   b_node.var_car_emissions_neg[t]
        b_node.const_car_emissions_neg = Constraint(set_t, rule=init_car_emissions_neg)

         # Define network constraints
        if not energyhub.configuration.energybalance.copperplate:
            def init_netw_inflow(const, t, car):
                return b_node.var_netw_inflow[t,car] == sum(model.network_block[netw].var_inflow[t,car,nodename]
                                                            for netw in model.set_networks
                                                            if car in model.network_block[netw].set_netw_carrier)
            b_node.const_netw_inflow = Constraint(set_t, b_node.set_carriers, rule=init_netw_inflow)

            def init_netw_outflow(const, t, car):
                return b_node.var_netw_outflow[t,car] == sum(model.network_block[netw].var_outflow[t,car,nodename]
                                                            for netw in model.set_networks
                                                            if car in model.network_block[netw].set_netw_carrier)
            b_node.const_netw_outflow = Constraint(set_t, b_node.set_carriers, rule=init_netw_outflow)

            if network_energy_consumption:
                def init_netw_consumption(const, t, car):
                    return b_node.var_netw_consumption[t,car] == sum(model.network_block[netw].var_consumption[t,car,nodename]
                                                                for netw in model.set_networks
                                                                 if data.network_data[netw].energy_consumption and
                                                                     car in model.network_block[netw].set_consumed_carriers)
                b_node.const_netw_consumption = Constraint(set_t, b_node.set_carriers, rule=init_netw_consumption)

        # BLOCKS
        # Add technologies as blocks
        b_node = add_technology(energyhub, nodename, b_node.set_tecsAtNode)

        return b_node

    model.node_blocks = Block(model.set_nodes, rule=init_node_block)

    return model







