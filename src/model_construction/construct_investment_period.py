from pyomo.environ import Set, RangeSet, Var


def construct_investment_period_block(b_period, data):
    """
    SETS
    - set_networks
    - set_t_full
    - set_t_clustered

    VARIABLES
    Cost Variables
    - var_cost_capex_tecs: Total Capex of technologies for respective investment period
    - var_cost_capex_netws: Total Capex of networks for respective investment period
    - var_cost_opex_tecs: Total Opex (fixed and variable) of technologies for respective investment period
    - var_cost_opex_netws: Total Opex (fixed and variable) of networks for respective investment period
    - var_cost_tecs: Total technology costs
    - var_cost_netws: Total network costs
    - var_cost_imports: Total import costs
    - var_cost_exports: Total export costs
    - var_cost_violation: Total violation cost
    - var_carbon_revenue: Total carbon revenues from negative emission technologies
    - var_carbon_cost: Total carbon cost from technologies, networks and imports/exports
    - var_cost_total: Total annualized cost for respective investment period
    Emission Variables
    - var_emissions_pos: Positive emissions from technologies, networks and imports/exports
    - var_emissions_neg: Negative emissions from technologies and imports/exports
    - var_emissions_net: Net emissions in investment period

    :param Block b_period: Pyomo block to holding respective investment period
    :param dict data: Data of investment period
    :return Block b_period:
    """

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
    b_period.var_cost_capex_tecs = Var()
    b_period.var_cost_capex_netws = Var()
    b_period.var_cost_opex_tecs = Var()
    b_period.var_cost_opex_netws = Var()
    b_period.var_cost_tecs = Var()
    b_period.var_cost_netws = Var()
    b_period.var_cost_imports = Var()
    b_period.var_cost_exports = Var()
    b_period.var_cost_violation = Var()
    b_period.var_carbon_revenue = Var()
    b_period.var_carbon_cost = Var()
    b_period.var_cost_total = Var()

    b_period.var_emissions_pos = Var()
    b_period.var_emissions_neg = Var()
    b_period.var_emissions_net = Var()

    return b_period
