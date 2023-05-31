from pyomo.environ import *
from pyomo.environ import units as u
import src.config_model as m_config
import numpy as np


def add_globals(model, data):
    r"""
        Adds all nodes with respective data to the model

        This function initializes parameters and decision variables that are on a global level. These include the total
        costs, emissions and carbon tax or carbon subsidy

    """


    # DEFINE VARIABLES
    # Global cost variables
    model.var_node_cost = Var()
    model.var_netw_cost = Var()
    model.var_total_cost = Var()
    model.var_carbon_revenue = Var()
    model.var_carbon_cost = Var()

    # Global Emission variables
    model.var_emissions_pos = Var()
    model.var_emissions_neg = Var()
    model.var_emissions_net = Var()

    # Parameters

    def init_carbon_subsidy(para, t):
        return data.global_data['carbon_prices']['subsidy'][t - 1]
    model.para_carbon_subsidy = Param(model.set_t, rule=init_carbon_subsidy, units=u.EUR / u.ton)

    def init_carbon_tax(para, t):
        return data.global_data['carbon_prices']['tax'][t - 1]
    model.para_carbon_tax = Param(model.set_t, rule=init_carbon_tax, units=u.EUR / u.ton)

    return model
