#TODO: Add carrier prices for import

from pyomo.environ import *
from pyomo.environ import units as u
from src.construct_technology import add_technologies

def add_nodes(model, data):
    def node_block_rule(b_node, nodename):
        """" Adds all nodes and technologies specified to the model
        - adds decision variables for nodes
        - adds technologies at node (decision vars and constraints)
        - adds energy balance of respective node
        """

        # Quick fix: add a carrier price (high)
        car_price = 100000

        # SETS: Get technologies for each node and make it a set for the block
        b_node.s_techs = Set(initialize=model.set_technologies[nodename])

        # PARAMETERS
        # Demand
        def set_demand(b, t, car):
            if nodename in data.demand.keys():
                return data.demand[nodename][car][t - 1]
        b_node.p_demand = Param(model.set_t, model.set_carriers, rule=set_demand, units=u.MWh)

        # DECISION VARIABLES
        # Interaction with network/system boundaries
        # TODO: Add bounds to variables from max size of networks
        # b_node.network_inflow = Var(model.set_t, model.set_carriers, bounds=(0, 1000), units=u.MWh)
        # b_node.network_outflow = Var(model.set_t, model.set_carriers, bounds=(0, 1000), units=u.MWh)
        b_node.import_flow = Var(model.set_t, model.set_carriers, bounds=(0, 1000), units=u.MWh)
        b_node.export_flow = Var(model.set_t, model.set_carriers, bounds=(0, 1000), units=u.MWh)

        # Create Variable for cost at node
        b_node.cost = Var(units=u.EUR)

        # ADD TECHNOLOGIES
        b_node = add_technologies(nodename, b_node, model, data)

        def calculate_cost_at_node(b_node):  # cost calculation at node
            return sum(b_node.tech_blocks[tec].var_CAPEX for tec in b_node.s_techs) + \
                    sum(sum(b_node.tech_blocks[tec].var_OPEX_variable[t] for tec in b_node.s_techs) for t in model.set_t) + \
                    sum(b_node.tech_blocks[tec].var_OPEX_fixed for tec in b_node.s_techs) + \
                    sum(sum(b_node.import_flow[t, car] * car_price for car in model.set_carriers) for t in model.set_t) == \
                    b_node.cost
        b_node.c_cost = Constraint(rule=calculate_cost_at_node)

    model.node_blocks = Block(model.set_nodes, rule=node_block_rule)

    return model







