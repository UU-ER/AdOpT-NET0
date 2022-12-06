from pyomo.environ import *
from pyomo.environ import units as u

def add_energybalance(model):
    # Make energy balance
    # def network_inflow_init(cons, t, car, node):
    #     return model.network_carrier_blocks[car].var_inflow[t, node]
    # model.cons_network_inflow = Constraint(model.set_t, model.set_carriers, model.set_nodes, rule=network_inflow_init)
    #
    #

    # TODO: formulate energybalance to include global balance
    """
    output_from_technologies - input_to_technologies + inflow_from_network - outflow_to_network + imports - exports
    == demand
    """
    def energybalance(cons, t, car, node):  # energybalance at each node
        node_block = model.node_blocks[node]
        if car in model.network_carrier_blocks:
            inflow = model.network_carrier_blocks[car].var_inflow[t, node]
            outflow = model.network_carrier_blocks[car].var_outflow[t, node]
        else:
            inflow = 0
            outflow = 0
        return \
            sum(node_block.tech_blocks[tec].var_output[t, car] for tec in node_block.s_techs if
                car in node_block.tech_blocks[tec].set_output_carriers) - \
            sum(node_block.tech_blocks[tec].var_input[t, car] for tec in node_block.s_techs if
                car in node_block.tech_blocks[tec].set_input_carriers) + \
            inflow + node_block.import_flow[t, car] - \
            outflow - node_block.export_flow[t, car] == \
            node_block.p_demand[t, car]
    model.cons_energybalance = Constraint(model.set_t, model.set_carriers, model.set_nodes, rule=energybalance)

    return model