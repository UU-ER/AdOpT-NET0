from pyomo.environ import *
from pyomo.environ import units as u
import src.global_variables as global_variables


def add_energybalance(energyhub):
    # TODO: formulate energybalance to include global balance
    """
    Calculates the energy balance for each node and carrier as:

    .. math::
        outputFromTechnologies - inputToTechnologies + \\
        inflowFromNetwork - outflowToNetwork + \\
        imports - exports = demand - genericProductionProfile

    :param EnergyHub energyhub: instance of the energyhub
    :return: model


    """

    # COLLECT OBJECTS FROM ENERGYHUB
    model = energyhub.model

    # Delete previously initialized constraints
    if model.find_component('const_energybalance'):
        model.del_component(model.const_energybalance)
        model.del_component(model.const_energybalance_index)

    # energybalance at each node (always at full resolution)
    set_t = model.set_t_full

    def init_energybalance(const, t, car, node):
        if car in model.node_blocks[node].set_carriers:
            node_block = model.node_blocks[node]
            tec_output = sum(node_block.tech_blocks_active[tec].var_output[t, car] for tec in node_block.set_tecsAtNode if
                             car in node_block.tech_blocks_active[tec].set_output_carriers)
            tec_input = sum(node_block.tech_blocks_active[tec].var_input[t, car] for tec in node_block.set_tecsAtNode if
                            car in node_block.tech_blocks_active[tec].set_input_carriers)
            netw_inflow = node_block.var_netw_inflow[t, car]
            netw_outflow = node_block.var_netw_outflow[t, car]
            if hasattr(node_block, 'var_netw_consumption'):
                netw_consumption = node_block.var_netw_consumption[t, car]
            else:
                netw_consumption = 0
            import_flow = node_block.var_import_flow[t, car]
            export_flow = node_block.var_export_flow[t, car]
            return \
                tec_output - tec_input + \
                netw_inflow - netw_outflow - netw_consumption + \
                import_flow - export_flow == \
                node_block.para_demand[t, car] - node_block.var_generic_production[t, car]
        else:
            return Constraint.Skip
    model.const_energybalance = Constraint(set_t, model.set_carriers, model.set_nodes, rule=init_energybalance)

    return model


def add_emissionbalance(energyhub):
    """
    Calculates the total and the net CO_2 balance.

    :param EnergyHub energyhub: instance of the energyhub
    :return: model
    """
    # COLLECT OBJECTS FROM ENERGYHUB
    model = energyhub.model

    # Emissionbalance is always at full resolution
    set_t = model.set_t_full

    nr_timesteps_averaged = global_variables.averaged_data_specs.nr_timesteps_averaged

    # Delete previously initialized constraints
    if model.find_component('const_emissions_pos'):
        model.del_component(model.const_emissions_pos)
        model.del_component(model.const_emissions_net)
        model.del_component(model.const_emissions_neg)

    # TODO: add unused CO2 to emissions
    # def init_emissionbalance(const, t, car, node):  # emissionbalance at each node
    # model.const_emissionbalance = Constraint(model.set_t, model.set_carriers, model.set_nodes, rule=init_emissionbalance)


    # calculate total emissions from technologies, networks and importing/exporting carriers
    def init_emissions_pos(const):
        from_technologies = sum(sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_tec_emissions_pos[t] *
                                        nr_timesteps_averaged
                                        for t in set_t)
                                    for tec in model.node_blocks[node].set_tecsAtNode)
                                for node in model.set_nodes)
        from_carriers = sum(sum(model.node_blocks[node].var_car_emissions_pos[t] *
                                        nr_timesteps_averaged
                                for t in set_t)
                            for node in model.set_nodes)
        from_networks = sum(sum(model.network_block[netw].var_netw_emissions_pos[t] *
                                        nr_timesteps_averaged
                                for t in set_t)
                            for netw in model.set_networks)
        return from_technologies + from_carriers + from_networks == model.var_emissions_pos
    model.const_emissions_tot = Constraint(rule=init_emissions_pos)

    # calculate negative emissions from technologies and import/export
    def init_emissions_neg(const):
        from_technologies = sum(sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_tec_emissions_neg[t] *
                                        nr_timesteps_averaged
                                    for t in set_t)
                                for tec in model.node_blocks[node].set_tecsAtNode)
                            for node in model.set_nodes)
        from_carriers = sum(sum(model.node_blocks[node].var_car_emissions_neg[t] *
                                        nr_timesteps_averaged
                                for t in set_t)
                            for node in model.set_nodes)
        return from_technologies + from_carriers == model.var_emissions_neg
    model.const_emissions_neg = Constraint(rule=init_emissions_neg)

    model.const_emissions_net = Constraint(expr=model.var_emissions_pos - model.var_emissions_neg == \
                                                model.var_emissions_net)

    return model


def add_system_costs(energyhub):
    """
    Calculates total system costs in three steps.

    - Calculates cost at all nodes as the sum of technology costs, import costs and export revenues
    - Calculates cost of all networks
    - Adds up cost of networks and node costs

    :param EnergyHub energyhub: instance of the energyhub
    :return: model
    """

    # COLLECT OBJECTS FROM ENERGYHUB
    model = energyhub.model

    # Delete previously initialized constraints
    if model.find_component('const_node_cost'):
        model.del_component(model.const_node_cost)
        model.del_component(model.const_netw_cost)
        model.del_component(model.const_cost)

    # Cost is always at full resolution
    set_t = model.set_t_full
    nr_timesteps_averaged = global_variables.averaged_data_specs.nr_timesteps_averaged

    # Cost at each node
    def init_node_cost(const):
        tec_capex = sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_capex
                            for tec in model.node_blocks[node].set_tecsAtNode)
                        for node in model.set_nodes)
        tec_opex_variable = sum(sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_opex_variable[t] *
                                        nr_timesteps_averaged
                                        for tec in model.node_blocks[node].set_tecsAtNode)
                                    for t in set_t)
                                for node in model.set_nodes)
        tec_opex_fixed = sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_opex_fixed
                                for tec in model.node_blocks[node].set_tecsAtNode)
                             for node in model.set_nodes)
        import_cost = sum(sum(sum(model.node_blocks[node].var_import_flow[t, car] *
                                    model.node_blocks[node].para_import_price[t, car] *
                                        nr_timesteps_averaged
                                  for car in model.node_blocks[node].set_carriers)
                              for t in set_t)
                          for node in model.set_nodes)
        export_revenue = sum(sum(sum(model.node_blocks[node].var_export_flow[t, car] *
                                     model.node_blocks[node].para_export_price[t, car] *
                                        nr_timesteps_averaged
                                    for car in model.node_blocks[node].set_carriers)
                                 for t in set_t)
                             for node in model.set_nodes)
        return tec_capex + tec_opex_variable + tec_opex_fixed + import_cost - export_revenue == model.var_node_cost
    model.const_node_cost = Constraint(rule=init_node_cost)

    # Calculates network costs
    def init_netw_cost(const):
        netw_capex = sum(model.network_block[netw].var_capex
                         for netw in model.set_networks)
        netw_opex_variable = sum(sum(model.network_block[netw].var_opex_variable[t] *
                                        nr_timesteps_averaged
                                     for netw in model.set_networks)
                                 for t in set_t)
        netw_opex_fixed = sum(model.network_block[netw].var_opex_fixed
                         for netw in model.set_networks)
        return netw_capex + netw_opex_variable + netw_opex_fixed == \
               model.var_netw_cost
    model.const_netw_cost = Constraint(rule=init_netw_cost)

    def init_total_cost(const):
        return \
            model.var_node_cost + model.var_netw_cost == \
            model.var_total_cost
    model.const_cost = Constraint(rule=init_total_cost)

    return model
