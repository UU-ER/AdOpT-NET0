from pyomo.environ import *
from pyomo.environ import units as u
import numpy as np
import src.global_variables as global_variables


def add_globals(energyhub):
    r"""
        Adds all nodes with respective data to the model

        This function initializes parameters and decision variables that are on a global level. These include the total
        costs, emissions and carbon tax or carbon subsidy

    """

    data = energyhub.data
    model = energyhub.model

    # DEFINE SETS
    # Nodes, Carriers, Technologies, Networks
    topology = data.topology
    model.set_nodes = Set(initialize=topology.nodes)
    model.set_carriers = Set(initialize=topology.carriers)

    def tec_node(set, node):
        if data.technology_data:
            return data.technology_data[node].keys()
        else:
            return Set.Skip

    model.set_technologies = Set(model.set_nodes, initialize=tec_node)
    model.set_networks = Set(initialize=data.network_data.keys())

    # Time Frame
    model.set_t_full = RangeSet(1, len(data.topology.timesteps))

    if global_variables.clustered_data == 1:
        model.set_t_clustered = RangeSet(1, len(data.topology.timesteps_clustered))

    # DEFINE VARIABLES
    # Global cost variables
    model.var_node_cost = Var()
    model.var_netw_cost = Var()
    model.var_total_cost = Var()

    # Global Emission variables
    model.var_emissions_pos = Var()
    model.var_emissions_neg = Var()
    model.var_emissions_net = Var()

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
    model.para_carbon_subsidy = Param(model.set_t_full, rule=init_carbon_subsidy)

    def init_carbon_tax(para, t):
        return data.global_data['carbon_prices']['tax'][t - 1]
    model.para_carbon_tax = Param(model.set_t_full, rule=init_carbon_tax)

    return model
