from pyomo.environ import *
from src.technology_construction import add_technologies


def add_nodes(model, data):
    def node_block_rule(b_node, node):
        """" Adds all nodes and technologies specified to the model
        - adds decision variables for nodes
        - adds technologies at node (decision vars and constraints)
        - adds energy balance of respective node
        """

        # Rules
        def set_demand(b, t, car):  # build demand as a parameter
            if node in data.demand.keys():
                if car in data.demand[node]:
                    return data.demand[node][car][t - 1]
                else:
                    return 0
            else:
                return 0
        #
        # def calculate_cost_at_node(b_node):  # cost calculation at node
        #     return sum(b_node.tech_blocks[tec].var_CAPEX for tec in b_node.s_techs) == b_node.cost
        #
        # def energybalance(b_node, t, car):  # energybalance at node
        #     return \
        #         sum(b_node.tech_blocks[tec].var_output[t, car] for tec in b_node.s_techs if
        #             car in b_node.tech_blocks[tec].set_output_carriers) - \
        #         sum(b_node.tech_blocks[tec].var_input[t, car] for tec in b_node.s_techs if
        #             car in b_node.tech_blocks[tec].set_input_carriers) + \
        #         b_node.network_inflow[t, car] + b_node.import_flow[t, car] - \
        #         b_node.network_outflow[t, car] - b_node.export_flow[t, car] == \
        #         b_node.p_demand[t, car]

        # SETS: Get technologies for each node and make it a set for the block
        b_node.s_techs = Set(initialize=model.set_technologies[node])

        # PARAMETERS
        b_node.p_demand = Param(model.set_t, model.set_carriers, rule=set_demand)

        # DECISION VARIABLES
        # Interaction with network/system boundaries
        # TODO: Add bounds to variables from data
        b_node.network_inflow = Var(model.set_t, model.set_carriers, bounds=(0, 100))
        b_node.network_outflow = Var(model.set_t, model.set_carriers, bounds=(0, 100))
        b_node.import_flow = Var(model.set_t, model.set_carriers, bounds=(0, 100))
        b_node.export_flow = Var(model.set_t, model.set_carriers, bounds=(0, 100))
        # Create Variable for cost at node
        b_node.cost = Var()

        # ADD TECHNOLOGIES
        b_node = add_technologies(b_node, model, data)



        # def technology_block_rule(b_tec, tec):
        #     """" Adds all technologies at considered node
        #     - Common decision variables and constraints are added first
        #     - Technology specific variables are added second
        #     """
        #
        #     # Rules
        #     def vOpex(b_tec, t):
        #         return b_tec.d_vOpex[t] == sum(b_tec.d_output[t, car] for car in b_tec.s_tecout) * b_tec.vOpex
        #
        #     def outconst(b_tec, t, input):
        #         return b_tec.d_input[t, input] <= b_tec.d_size
        #
        #     # SETS
        #     b_tec.s_tecin = Set(initialize=data.tecin[tec])
        #     b_tec.s_tecout = Set(initialize=data.tecout[tec])
        #
        #     # PARAMETERS
        #     # Settings to get from data # TODO: define this from data
        #     input = data.tecin[tec]
        #     size_is_int = 0
        #     capexModel = 2  # (1, 2)
        #     sizebounds = (0, 100)
        #     bp_x = [0, 5, 100]
        #     bp_y = [0, 15, 250]
        #     b_tec.fOpex = Param(initialize=.01)
        #     b_tec.vOpex = Param(initialize=2)
        #     tectype = 1
        #
        #     # DECISION VARIABLES
        #     if input in ['wind', 'solar']:  # input
        #         b_tec.d_input = Param(m.s_t, b_tec.s_tecin)
        #     else:
        #         b_tec.d_input = Var(m.s_t, b_tec.s_tecin, within=NonNegativeReals)
        #     b_tec.d_output = Var(m.s_t, b_tec.s_tecout, within=NonNegativeReals)  # output
        #     if size_is_int:  # size
        #         b_tec.d_size = Var(within=NonNegativeIntegers, bounds=sizebounds)
        #     else:
        #         b_tec.d_size = Var(within=NonNegativeReals, bounds=sizebounds)
        #
        #     b_tec.d_Capex = Var()  # capex
        #     b_tec.d_vOpex = Var(m.s_t)  # variable opex
        #     b_tec.d_fOpex = Var()  # fixed opex
        #
        #     # CONSTRAINTS
        #     # Capex
        #     if capexModel == 1:
        #         b_tec.c_Capex = Constraint(expr=b_tec.d_size * data.investcost[tec] == b_tec.d_Capex)
        #     elif capexModel == 2:
        #         b_tec.c_Capex = Piecewise(b_tec.d_Capex, b_tec.d_size,
        #                                   pw_pts=bp_x,
        #                                   pw_constr_type='EQ',
        #                                   f_rule=bp_y,
        #                                   pw_repn='SOS2')
        #     # fixed Opex
        #     b_tec.c_fOpex = Constraint(expr=b_tec.d_Capex * b_tec.fOpex == b_tec.d_fOpex)
        #     # variable Opex
        #     b_tec.c_vOpex = Constraint(m.s_t, rule=vOpex)
        #     # Size constraint
        #     b_tec.c_size = Constraint(m.s_t, b_tec.s_tecin, rule=outconst)
        #     # Technology performances
        #     if tectype == 1:  # linear performance, 1 input -> 1 output
        #         def inout1(b_tec, t, input, output):
        #             return b_tec.d_output[t, output] == data.alpha[tec] * b_tec.d_input[t, input]
        #
        #         b_tec.c_performance = Constraint(m.s_t, b_tec.s_tecin, b_tec.s_tecout, rule=inout1)
        #
        # b_node.tech_blocks = Block(b_node.s_techs, rule=technology_block_rule)

    #     b_node.c_cost = Constraint(rule=calculate_cost_at_node)
    #
    #     # Make energy balance
    #     b_node.c_energybalance = Constraint(model.set_t, model.set_carriers, rule=energybalance)
    #
    #     # Quick fix to ensure no heat import
    #     def heat_to_zeroimport1(m, t):
    #         return b_node.network_inflow[t, 'heat'] == 0
    #
    #     b_node.c_import_heat1 = Constraint(model.set_t, rule=heat_to_zeroimport1)
    #
    #     def heat_to_zeroimport2(m, t):
    #         return b_node.import_flow[t, 'heat'] == 0
    #
    #     b_node.c_import_heat2 = Constraint(model.set_t, rule=heat_to_zeroimport2)
    #
    model.node_blocks = Block(model.set_nodes, rule=node_block_rule)

    return model







