from pyomo.environ import *
from pyomo.environ import units as u

def add_energybalance(model):
    # TODO: formulate energybalance to include global balance
    """
    Calculates the energy balance for each node and carrier as:

    .. math::
        outputFromTechnologies - inputToTechnologies + \\
        inflowFromNetwork - outflowToNetwork + \\
        imports - exports = demand


    """

    # Delete previously initialized constraints
    if model.find_component('const_energybalance'):
        model.const_energybalance.del_component()

    def init_energybalance(const, t, car, node):  # energybalance at each node
        node_block = model.node_blocks[node]
        tec_output = sum(node_block.tech_blocks_active[tec].var_output[t, car] for tec in node_block.set_tecsAtNode if
                         car in node_block.tech_blocks_active[tec].set_output_carriers)
        tec_input = sum(node_block.tech_blocks_active[tec].var_input[t, car] for tec in node_block.set_tecsAtNode if
                        car in node_block.tech_blocks_active[tec].set_input_carriers)
        netw_inflow = node_block.var_netw_inflow[t, car]
        netw_outflow = node_block.var_netw_outflow[t, car]
        netw_consumption = node_block.var_netw_consumption[t, car]
        import_flow = node_block.var_import_flow[t, car]
        export_flow = node_block.var_export_flow[t, car]
        return \
            tec_output - tec_input + \
            netw_inflow - netw_outflow - netw_consumption + \
            import_flow - export_flow == \
            node_block.para_demand[t, car]
    model.const_energybalance = Constraint(model.set_t, model.set_carriers, model.set_nodes, rule=init_energybalance)

    return model

def add_system_costs(model):
    """
    Calculates total system costs in three steps.

    - Calculates cost at all nodes as the sum of technology costs, import costs and export revenues
    - Calculates cost of all networks
    - Adds up cost of networks and node costs
    """
    # Delete previously initialized constraints
    if model.find_component('const_node_cost'):
        model.const_node_cost.del_component()
        model.const_netw_cost.del_component()
        model.const_cost.del_component()

    # Cost at each node
    def init_node_cost(const):
        return sum(
                sum(model.node_blocks[node].tech_blocks_active[tec].var_CAPEX
                       for tec in model.node_blocks[node].set_tecsAtNode) + \
                   sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_OPEX_variable[t]
                           for tec in model.node_blocks[node].set_tecsAtNode) for t in model.set_t) + \
                   sum(model.node_blocks[node].tech_blocks_active[tec].var_OPEX_fixed
                       for tec in model.node_blocks[node].set_tecsAtNode) + \
                   sum(sum(model.node_blocks[node].var_import_flow[t, car] * model.node_blocks[node].para_import_price[t, car]
                           for car in model.set_carriers) for t in model.set_t) - \
                   sum(sum(model.node_blocks[node].var_export_flow[t, car] * model.node_blocks[node].para_export_price[t, car]
                           for car in model.set_carriers) for t in model.set_t) \
                for node in model.set_nodes) == \
               model.var_node_cost
    model.const_node_cost = Constraint(rule=init_node_cost)

    # Calculates network costs
    def init_netw_cost(const):
        return sum(model.network_block[netw].var_cost for netw in model.set_networks) == \
               model.var_netw_cost
    model.const_netw_cost = Constraint(rule=init_netw_cost)

    def init_total_cost(const):
        return \
            model.var_node_cost + model.var_netw_cost == \
            model.var_total_cost
    model.const_cost = Constraint(rule=init_total_cost)

    return model