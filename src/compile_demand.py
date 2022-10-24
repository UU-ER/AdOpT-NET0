from pyomo.environ import *

def compile_demand(model, demand):
    def demand_init(model, t, car):  # build demand as a parameter
        if car in demand:
            return demand[car][t - 1]
        else:
            return 0
    model.p_demand = Param(model.s_t, model.s_car, initialize=demand_init)
    return model