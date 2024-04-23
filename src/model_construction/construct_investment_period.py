from pyomo.environ import Set, RangeSet, Var


def construct_investment_period_block(b_period, data):

    # PREPROCESSING
    investment_period = b_period.index()
    config = data["config"]
    topology = data["topology"]
    network_data = data["network_data"]

    # SETS
    b_period.set_networks = Set(initialize=network_data.keys())
    b_period.set_t_full = RangeSet(1, len(topology["time_index"]["full"]))
    if config["optimization"]["typicaldays"]["N"]["value"] != 0:
        b_period.set_t_clustered = RangeSet(1, len(topology["time_index"]["clustered"]))
    else:
        b_period.set_t_clustered = RangeSet(1, len(topology["time_index"]["full"]))

    # VARIABLES
    b_period.var_node_cost = Var()
    b_period.var_netw_cost = Var()
    b_period.var_total_cost = Var()
    b_period.var_carbon_revenue = Var()
    b_period.var_carbon_cost = Var()
    b_period.var_emissions_pos = Var()
    b_period.var_emissions_neg = Var()
    b_period.var_emissions_net = Var()

    # PARAMETERS

    # CONSTRAINTS

    return b_period
