from pyomo.environ import *
from pyomo.environ import units as u

def add_emissionbalance(model):

    """
    Calculates the total and the net CO_2 balance.


    """
    # def init_emissionbalance(const, t, car, node):  # emissionbalance at each node
    #TODO: add unused CO2 to emissions

    # model.const_emissionbalance = Constraint(model.set_t, model.set_carriers, model.set_nodes, rule=init_emissionbalance)

    return model