from pyomo.environ import *
from pyomo.environ import units as u

def add_energybalance(model):
    # TODO: formulate energybalance to include global balance
    """
    output_from_technologies - input_to_technologies + inflow_from_network - outflow_to_network + imports - exports
    == demand
    """
    def init_energybalance(const, t, car, node):  # energybalance at each node
        node_block = model.node_blocks[node]
        tec_output = sum(node_block.tech_blocks[tec].var_output[t, car] for tec in node_block.set_tecsAtNode if
                car in node_block.tech_blocks[tec].set_output_carriers)
        tec_input = sum(node_block.tech_blocks[tec].var_input[t, car] for tec in node_block.set_tecsAtNode if
                car in node_block.tech_blocks[tec].set_input_carriers)
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