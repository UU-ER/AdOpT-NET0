from pyomo.environ import *

def compile_demand(model, demand):
    def demand_init(model, t, car, node):  # build demand as a parameter
        if node in demand.keys():
            if car in demand[node]:
                return demand[node][car][t - 1]
            else:
                return 0
        else:
            return 0
    model.para_demand = Param(model.set_t, model.set_carriers, model.set_nodes, initialize=demand_init)
    return model