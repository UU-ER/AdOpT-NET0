from pyomo.environ import *
from gurobipy import GRB
import numpy as np
from src.compile_demand import compile_demand


# Initialize model
ehub = ConcreteModel()

# Define sets
ehub.s_tecs = Set(initialize = ['FurnaceNg', 'FurnaceEl'])  # Technologies
ehub.s_car  = Set(initialize = ['electricity', 'heat', 'natural_gas'])  # Carriers
ehub.s_t    = RangeSet(1,8760)  # Timescale

# Define Technologies THIS NEEDS TO GO IN DATAFILE
investcost = dict() # Define CAPEX
investcost['FurnaceNg'] = 100
investcost['FurnaceEl'] = 100

alpha = dict() # Define performance, input, output
alpha['FurnaceNg'] = 0.7
alpha['FurnaceEl']  = 0.9

tecin = dict() # Input of technologies
tecin['FurnaceNg'] = ['natural_gas']
tecin['FurnaceEl'] = ['electricity']

tecout = dict() # Output of technologies
tecout['FurnaceNg'] = ['heat']
tecout['FurnaceEl'] = ['heat']

demand = dict() # Demands
demand['heat'] = np.ones(8760) * 60

# Build data required for model construction
ehub = compile_demand(ehub, demand)

# Build technology model blocks
def technologymodels_type1(b, tec):
    # Define input/output sets
    b.s_tecin = Set(initialize= tecin[tec])
    b.s_tecout = Set(initialize= tecout[tec])

    # Define variables
    b.d_input = Var(ehub.s_t, b.s_tecin)
    b.d_output = Var(ehub.s_t, b.s_tecout)
    b.d_size = Var(within=NonNegativeReals)
    b.d_capex = Var()

    # Define constraints
    def inout1(b, t, input, output):
        return b.d_output[t,output] == alpha[tec] * b.d_input[t,input]
    b.c_performance1 = Constraint(ehub.s_t, b.s_tecin, b.s_tecout, rule=inout1)

    # Define Investment Costs
    b.c_capex = Constraint(expr=b.d_size * investcost[tec] == b.d_capex)

ehub.technologymodels = Block(ehub.s_tecs, rule=technologymodels_type1)


# Define energy balance
ehub.d_export = Var(ehub.s_t, ehub.s_car)
ehub.d_import = Var(ehub.s_t, ehub.s_car)


# def sum_all_inputs(m, car):
#     return sum(ehub.Tec_in[car,t] for t in m.in_tec_comb[car]) == m.insum[car]
# m.sumIn = Constraint(m.car, rule=sum_all_inputs)


def energybalance(ehub, t, car):
        return \
            sum(ehub.technologymodels[tec].d_output[t, car] for tec in ehub.s_tecs if car in ehub.technologymodels[tec].s_tecout) \
            - sum(ehub.technologymodels[tec].d_input[t, car] for tec in ehub.s_tecs if car in ehub.technologymodels[tec].s_tecin) \
            + ehub.d_import[t, car] \
            - ehub.d_export[t, car] \
            == ehub.p_demand[t, car]
ehub.c_energybalance = Constraint(ehub.s_t, ehub.s_car, rule=energybalance)
ehub.c_energybalance.pprint()


# Dirty Fix to Run
def heatimport(ehub, t):
    return ehub.d_import[t, 'heat'] == 0

ehub.c_heatimport = Constraint(ehub.s_t, rule=heatimport)
ehub.c_heatimport.pprint()

# def totalcost(ehub, t):
#     return sum(ehub.technologymodels[t].c_capex)
ehub.objective = Objective(expr=ehub.technologymodels['FurnaceNg'].d_capex + ehub.technologymodels['FurnaceEl'].d_capex, sense=minimize)
ehub.objective.pprint()

solve = SolverFactory('gurobi_persistent')
solve.set_instance(ehub)
solution = solve.solve()
solution.write()
#
# # Initialize ehub model
# ehub = ConcreteModel()
# ehub.m1 = technologies[s_technologies[0]]
# ehub.m2 = technologies[s_technologies[1]]


#
# model.T = RangeSet(0, 23)
# maxSize = 1000
# model.s = Var(within=NonNegativeReals, bounds=(0,maxSize), initialize=0)
# model.p_out = Var(model.T, within=NonNegativeReals)
# model.p_in = Var(model.T, within=NonNegativeReals)
#
# def inout(model, t):
#     return model.p_out[t] == 0.7 * model.p_in[t]
# model.c_performance = Constraint(model.T, rule=inout)
#
# def outconst(model, t):
#     return model.p_out[t] <= model.s
# model.c_size = Constraint(model.T, rule=outconst)
#
# model.profit = Objective(expr = 1000*sum(model.p_out[t] for t in model.T) - 30*sum(model.p_in[t] for t in model.T), sense=maximize)
#
# solve = SolverFactory('gurobi_persistent')
# solve.set_instance(model)
# solution = solve.solve()
# solution.write()
#
# model.c_size2 = Constraint(expr = model.s <= 800)
# solve.add_constraint(model.c_size2)
# solution = solve.solve()
# solution.write()
#
# solve.remove_constraint(model.c_size2)
# solution = solve.solve()
# solution.write()
#
# # solution = solver.solve(model, tee=True)


