from pyomo.environ import *
# from gurobipy import GRB
# from src.compile_demand import compile_demand
# from src.DataHandle import DataHandle


# How to formulate hierarchical models with blocks
m = ConcreteModel()

m.t = RangeSet(1,2)
m.tec = Set(initialize=['Tec1','Tec2'])


def time_block_rule(b_t):


    def tec_block_rule(b_tec):
        b_tec.var_input = Var()
        b_tec.var_output = Var()
        def inout(b_tec):
            return b_tec.var_output == 0.7 * b_tec.var_input
        b_tec.c_perf = Constraint(rule=inout)
    b_t.tecBlock = Block(m.tec, rule=tec_block_rule)


m.time_block = Block(m.t, rule=time_block_rule)

m.time_block.pprint()

# Set definitions
m.nodes = Set(initialize=['onshore','offshore'])
m.tecs = Set(m.nodes, dimen=1, initialize=['Tec1','Tec2'])

m.tecs['onshore'].pprint()

