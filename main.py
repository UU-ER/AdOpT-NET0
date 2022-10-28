# TODO: Include pvlib, hplib, windpowerlib
from pyomo.environ import *
from gurobipy import GRB
from src.compile_demand import compile_demand
from src.DataHandle import DataHandle
from src.energyhub import energyhub

topology = {}
topology['time'] = range(1, 3)
topology['carriers'] = ['electricity', 'heat', 'natural_gas']
topology['nodes'] = ['onshore', 'offshore']
topology['technologies'] = {}
topology['technologies']['onshore'] = ['FurnaceEl', 'FurnaceNg']
topology['technologies']['offshore'] = ['FurnaceEl']
topology['networks'] = ['electricity']

data = DataHandle(topology)
# Read data
energyhub = energyhub(topology, data)
# Construct equations
energyhub.construct_model()
# Solve model
# energyhub.solve()


# ehub.m.pprint()
# solve = SolverFactory('gurobi_persistent')
# solve.set_instance(ehub.model)
# solution = solve.solve()
# solution.write()
# ehub.model.display()


# Define sets
# ehub.s_nodes = Set(initialize=sets['nodes'])  # Nodes
# ehub.s_carriers = Set(initialize=sets['carriers'])  # Carriers
# ehub.s_t = Set(initialize=sets['time'])  # Timescale
# def tec_node(ehub, node):
#     if node in ehub.s_nodes:
#         return sets['technologies'][node]
# ehub.s_technologies = Set(ehub.s_nodes, initialize=tec_node)  # Technologies

# Define Technologies THIS NEEDS TO GO IN DATAFILE


# Build data required for model construction
#
#
# # Define a node block which is indexed over each node
# def node_block_rule(b_node, node):
#     # Get technologies for each node and make it a set for the block
#     b_node.s_techs = Set(initialize=ehub.s_technologies[node])
#
#     # Create Variables for interaction with network/system boundaries
#     b_node.network_inflow = Var(ehub.s_t, ehub.s_carriers, bounds=(0,100))
#     b_node.network_outflow = Var(ehub.s_t, ehub.s_carriers, bounds=(0,100))
#     b_node.import_flow = Var(ehub.s_t, ehub.s_carriers, bounds=(0,100))
#     b_node.export_flow = Var(ehub.s_t, ehub.s_carriers, bounds=(0,100))
#
#     # Create Variable for cost at node
#     b_node.cost = Var()
#
#     # Define technology block
#     def technology_block_rule(b_tec, tec):
#         # Define set of input/output carriers
#         b_tec.s_tecin = Set(initialize=data.tecin[tec])
#         b_tec.s_tecout = Set(initialize=data.tecout[tec])
#
#         # Define decision variables
#         b_tec.d_input = Var(ehub.s_t, b_tec.s_tecin, within=NonNegativeReals)
#         b_tec.d_output = Var(ehub.s_t, b_tec.s_tecout, within=NonNegativeReals)
#         b_tec.d_size = Var(within=NonNegativeReals)
#         b_tec.d_capex = Var()
#
#         # Define constraints
#         def inout1(b_tec, t, input, output):
#             return b_tec.d_output[t, output] == data.alpha[tec] * b_tec.d_input[t, input]
#         b_tec.c_performance = Constraint(ehub.s_t, b_tec.s_tecin, b_tec.s_tecout, rule=inout1)
#
#         # Size constraint
#         def outconst(b_tec, t, input):
#             return b_tec.d_input[t, input] <= b_tec.d_size
#         b_tec.c_size = Constraint(ehub.s_t, b_tec.s_tecin, rule=outconst)
#
#         # Define Investment Costs
#         b_tec.c_capex = Constraint(expr=b_tec.d_size * data.investcost[tec] == b_tec.d_capex)
#     b_node.tech_blocks = Block(b_node.s_techs, rule=technology_block_rule)
#
#     # energy balance of node
#     def energybalance(b_node, t, car):
#         return \
#         sum(b_node.tech_blocks[tec].d_output[t, car] for tec in b_node.s_techs if car in b_node.tech_blocks[tec].s_tecout) - \
#         sum(b_node.tech_blocks[tec].d_input[t, car] for tec in b_node.s_techs if car in b_node.tech_blocks[tec].s_tecin) + \
#         b_node.network_inflow[t, car] + b_node.import_flow[t, car] - \
#         b_node.network_outflow[t, car] - b_node.export_flow[t, car] == \
#         ehub.p_demand[t, car]
#     b_node.c_energybalance = Constraint(ehub.s_t, ehub.s_carriers, rule=energybalance)
#
#     # Cost at node
#     def capex_calc(b_node):
#         return sum(b_node.tech_blocks[tec].d_capex for tec in b_node.s_techs) == b_node.cost
#     b_node.c_cost = Constraint(rule=capex_calc)
#
#     # Quick fix to ensure no heat import
#     def heat_to_zeroimport1(ehub, t):
#         return b_node.network_inflow[t, 'heat'] == 0
#     b_node.c_import_heat1 = Constraint(ehub.s_t, rule=heat_to_zeroimport1)
#
#     def heat_to_zeroimport2(ehub, t):
#         return b_node.import_flow[t, 'heat'] == 0
#     b_node.c_import_heat2 = Constraint(ehub.s_t, rule=heat_to_zeroimport2)
#
# ehub.node_blocks = Block(ehub.s_nodes, rule=node_block_rule)
# ehub.node_blocks.pprint()
#
# # ehub.s_nodes_used = Set(initialize=['onshore'])
# # deactivate = ehub.s_nodes - ehub.s_nodes_used
# # deactivate.pprint()
# # ehub.node_blocks[ehub.s_nodes - ehub.s_nodes_used].deactivate()
#
#
#
# def cost_objective(ehub):
#     return sum(ehub.node_blocks[n].cost for n in ehub.s_nodes)
# ehub.objective = Objective(rule=cost_objective, sense=minimize)
# ehub.objective.pprint()
#
# solve = SolverFactory('gurobi_persistent')
# solve.set_instance(ehub)
# solution = solve.solve()
# solution.write()
# # ehub.display()
#
# ehub.node_blocks['offshore'].deactivate()
# ehub.objective.clear()
# ehub.objective = Objective(rule=cost_objective, sense=minimize)
#
# solve = SolverFactory('gurobi_persistent')
# solve.set_instance(ehub)
# solution = solve.solve()
# solution.write()
# ehub.display()
