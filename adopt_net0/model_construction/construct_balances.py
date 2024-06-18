import pyomo.environ as pyo

from ..utilities import get_set_t, get_hour_factors, get_nr_timesteps_averaged


def delete_all_balances(model):
    """
    Deletes all balances (required if they need to be reconstructed)

    :param model: pyomo model
    :return: pyomo model
    """

    if model.find_component("block_network_constraints"):
        model.del_component(model.block_network_constraints)
    if model.find_component("const_energybalance"):
        model.del_component(model.const_energybalance)
    if model.find_component("const_emissions_tot"):
        model.del_component(model.const_emissions_tot)
        model.del_component(model.const_emissions_neg)
        model.del_component(model.const_emissions_net)
    if model.find_component("block_costbalance"):
        model.del_component(model.block_costbalance)
    if model.find_component("const_npv"):
        model.del_component(model.const_npv)
    if model.find_component("const_emissions"):
        model.del_component(model.const_emissions)

    return model


def construct_network_constraints(model, config: dict):
    """
    Construct the network constraints to calculate nodal in- and outflow

    .. math::
      outflowToNetwork = \sum(outflow \forall arcs at node)

    .. math::
      inflowFromNetwork = \sum(inflow \forall arcs at node)\\

    :param model: pyomo model
    :param dict config: dict containing model information
    :return: pyomo model
    """

    def init_network_constraints(b_netw_const, period):
        """Pyomo rule to generate network constraint block"""

        b_period = model.periods[period]

        set_t = get_set_t(config, b_period)

        def init_netw_inflow(const, node, car, t):
            if car in b_period.node_blocks[node].set_carriers:
                return b_period.node_blocks[node].var_netw_inflow[t, car] == sum(
                    b_period.network_block[netw].var_inflow[t, car, node]
                    for netw in b_period.set_networks
                    if car in b_period.network_block[netw].set_netw_carrier
                )
            else:
                return pyo.Constraint.Skip

        b_netw_const.const_netw_inflow = pyo.Constraint(
            model.set_nodes, model.set_carriers, set_t, rule=init_netw_inflow
        )

        def init_netw_outflow(const, node, car, t):

            if car in b_period.node_blocks[node].set_carriers:
                return b_period.node_blocks[node].var_netw_outflow[t, car] == sum(
                    b_period.network_block[netw].var_outflow[t, car, node]
                    for netw in b_period.set_networks
                    if car in b_period.network_block[netw].set_netw_carrier
                )
            else:
                return pyo.Constraint.Skip

        b_netw_const.const_netw_outflow = pyo.Constraint(
            model.set_nodes, model.set_carriers, set_t, rule=init_netw_outflow
        )

        def init_netw_consumption(const, node, car, t):

            if (b_period.node_blocks[node].find_component("var_consumption")) and (
                car in b_period.node_blocks[node].set_carriers
            ):
                return b_period.node_blocks[node].var_netw_consumption[t, car] == sum(
                    b_period.network_block[netw].var_consumption[t, car, node]
                    for netw in b_period.set_networks
                    if (b_period.network_block[netw].find_component("var_consumption"))
                    and car in b_period.network_block[netw].set_consumed_carriers
                )
            else:
                return pyo.Constraint.Skip

        b_netw_const.const_netw_consumption = pyo.Constraint(
            model.set_nodes, model.set_carriers, set_t, rule=init_netw_consumption
        )

    model.block_network_constraints = pyo.Block(
        model.set_periods, rule=init_network_constraints
    )

    return model


def construct_nodal_energybalance(model, config: dict):
    """
    Calculates the energy balance for each node and carrier

    .. math::
        outputFromTechnologies - inputToTechnologies + \\
        inflowFromNetwork - outflowToNetwork + \\
        imports - exports = demand - genericProductionProfile

    :param model: pyomo model
    :param dict config: dict containing model information
    :return: pyomo model
    """

    def init_energybalance(b_ebalance, period):
        b_period = model.periods[period]

        set_t = get_set_t(config, b_period)

        # Violation variables and costs
        if config["energybalance"]["violation"]["value"] > 0:
            b_period.var_violation = pyo.Var(
                set_t,
                model.set_carriers,
                model.set_nodes,
                domain=pyo.NonNegativeReals,
            )
            b_period.var_cost_violation = pyo.Var()

        def init_energybalance(const, t, car, node):
            if car in b_period.node_blocks[node].set_carriers:
                node_block = b_period.node_blocks[node]
                tec_output = sum(
                    node_block.tech_blocks_active[tec].var_output_tot[t, car]
                    for tec in node_block.set_technologies
                    if car in node_block.tech_blocks_active[tec].set_output_carriers_all
                )

                tec_input = sum(
                    node_block.tech_blocks_active[tec].var_input_tot[t, car]
                    for tec in node_block.set_technologies
                    if car in node_block.tech_blocks_active[tec].set_input_carriers_all
                )

                netw_inflow = node_block.var_netw_inflow[t, car]

                netw_outflow = node_block.var_netw_outflow[t, car]

                if hasattr(node_block, "var_netw_consumption"):
                    netw_consumption = node_block.var_netw_consumption[t, car]
                else:
                    netw_consumption = 0

                import_flow = node_block.var_import_flow[t, car]

                export_flow = node_block.var_export_flow[t, car]

                if config["energybalance"]["violation"]["value"] > 0:
                    violation = b_period.var_violation[t, car, node]
                else:
                    violation = 0
                return (
                    tec_output
                    - tec_input
                    + netw_inflow
                    - netw_outflow
                    - netw_consumption
                    + import_flow
                    - export_flow
                    + violation
                    == node_block.para_demand[t, car]
                    - node_block.var_generic_production[t, car]
                )
            else:
                return pyo.Constraint.Skip

        b_ebalance.const_energybalance = pyo.Constraint(
            set_t, model.set_carriers, model.set_nodes, rule=init_energybalance
        )

        return b_ebalance

    model.block_energybalance = pyo.Block(model.set_periods, rule=init_energybalance)

    return model


def construct_global_energybalance(model, config):
    """
    Calculates the global energy balance for each carrier summed over all nodes

    .. math::
        outputFromTechnologies - inputToTechnologies + \\
        imports - exports = demand - genericProductionProfile

    :param model: pyomo model
    :param dict config: dict containing model information
    :return: pyomo model
    """

    def init_energybalance(b_ebalance, period):
        b_period = model.periods[period]

        set_t = get_set_t(config, b_period)

        # Violation variables and costs
        if config["energybalance"]["violation"]["value"] >= 0:
            b_period.var_violation = pyo.Var(
                set_t,
                model.set_carriers,
                model.set_nodes,
                domain=pyo.NonNegativeReals,
            )
            b_period.var_cost_violation = pyo.Var()

        def init_energybalance_global(const, t, car):
            tec_output = sum(
                sum(
                    b_period.node_blocks[node]
                    .tech_blocks_active[tec]
                    .var_output_tot[t, car]
                    for tec in b_period.node_blocks[node].set_technologies
                    if car in b_period.node_blocks[node].set_carriers
                    and b_period.node_blocks[node]
                    .tech_blocks_active[tec]
                    .set_output_carriers_all
                )
                for node in model.set_nodes
            )

            tec_input = sum(
                sum(
                    b_period.node_blocks[node]
                    .tech_blocks_active[tec]
                    .var_input_tot[t, car]
                    for tec in b_period.node_blocks[node].set_technologies
                    if car in b_period.node_blocks[node].set_carriers
                    and b_period.node_blocks[node]
                    .tech_blocks_active[tec]
                    .set_input_carriers_all
                )
                for node in model.set_nodes
            )

            import_flow = sum(
                b_period.node_blocks[node].var_import_flow[t, car]
                for node in model.set_nodes
                if car in b_period.node_blocks[node].set_carriers
            )

            export_flow = sum(
                b_period.node_blocks[node].var_export_flow[t, car]
                for node in model.set_nodes
                if car in b_period.node_blocks[node].set_carriers
            )

            demand = sum(
                b_period.node_blocks[node].para_demand[t, car]
                for node in model.set_nodes
                if car in b_period.node_blocks[node].set_carriers
            )

            gen_prod = sum(
                b_period.node_blocks[node].var_generic_production[t, car]
                for node in model.set_nodes
                if car in b_period.node_blocks[node].set_carriers
            )

            if config["energybalance"]["violation"]["value"] > 0:
                violation = sum(
                    b_period.var_violation[t, car, node]
                    for node in model.set_nodes
                    if car in b_period.node_blocks[node].set_carriers
                )
            else:
                violation = 0

            return (
                tec_output - tec_input + import_flow - export_flow + violation
                == demand - gen_prod
            )

        model.set_used_carriers = pyo.Set(
            initialize=list(
                set().union(
                    *[
                        b_period.node_blocks[node].set_carriers
                        for node in model.set_nodes
                    ]
                )
            )
        )

        b_ebalance.const_energybalance = pyo.Constraint(
            set_t, model.set_used_carriers, rule=init_energybalance_global
        )

        return b_ebalance

    model.block_energybalance = pyo.Block(model.set_periods, rule=init_energybalance)

    return model


def construct_emission_balance(model, data):
    """
    Calculates the total postive and negative emissions as well as the net emissions

    .. math::
        E_{pos, tot} = E_{pos, technologies} + E_{pos, carriers} + E_{pos,
        networks}

    .. math::
        E_{neg, tot} = E_{neg, technologies} + E_{neg, carriers}

    .. math::
        E_{net} = E_{pos, tot} - E_{neg, tot}


    :param model: pyomo model
    :param data: DataHandle
    :return: pyomo model
    """
    config = data.model_config

    def init_emissionbalance(b_emissionbalance, period):
        b_period = model.periods[period]
        set_t = get_set_t(config, b_period)
        hour_factors = get_hour_factors(config, data, period)
        nr_timesteps_averaged = get_nr_timesteps_averaged(config)

        # calculate total emissions from technologies, networks and importing/exporting carriers
        def init_emissions_pos(const):
            from_technologies = sum(
                sum(
                    sum(
                        b_period.node_blocks[node]
                        .tech_blocks_active[tec]
                        .var_tec_emissions_pos[t]
                        * nr_timesteps_averaged
                        * hour_factors[t - 1]
                        for t in set_t
                    )
                    for tec in b_period.node_blocks[node].set_technologies
                )
                for node in model.set_nodes
            )
            from_carriers = sum(
                sum(
                    b_period.node_blocks[node].var_car_emissions_pos[t]
                    * nr_timesteps_averaged
                    * hour_factors[t - 1]
                    for t in set_t
                )
                for node in model.set_nodes
            )
            if not config["energybalance"]["copperplate"]["value"]:
                from_networks = sum(
                    sum(
                        sum(
                            b_period.network_block[netw].var_netw_emissions_pos[t, node]
                            * nr_timesteps_averaged
                            * hour_factors[t - 1]
                            for t in set_t
                        )
                        for node in model.set_nodes
                    )
                    for netw in b_period.set_networks
                )
            else:
                from_networks = 0
            return (
                from_technologies + from_carriers + from_networks
                == b_period.var_emissions_pos
            )

        b_emissionbalance.const_emissions_tot = pyo.Constraint(rule=init_emissions_pos)

        def init_emissions_neg(const):
            from_technologies = sum(
                sum(
                    sum(
                        b_period.node_blocks[node]
                        .tech_blocks_active[tec]
                        .var_tec_emissions_neg[t]
                        * nr_timesteps_averaged
                        * hour_factors[t - 1]
                        for t in set_t
                    )
                    for tec in b_period.node_blocks[node].set_technologies
                )
                for node in model.set_nodes
            )
            from_carriers = sum(
                sum(
                    b_period.node_blocks[node].var_car_emissions_neg[t]
                    * nr_timesteps_averaged
                    * hour_factors[t - 1]
                    for t in set_t
                )
                for node in model.set_nodes
            )
            return from_technologies + from_carriers == b_period.var_emissions_neg

        b_emissionbalance.const_emissions_neg = pyo.Constraint(rule=init_emissions_neg)

        b_emissionbalance.const_emissions_net = pyo.Constraint(
            expr=b_period.var_emissions_pos - b_period.var_emissions_neg
            == b_period.var_emissions_net
        )

        return b_emissionbalance

    model.block_emissionbalance = pyo.Block(
        model.set_periods, rule=init_emissionbalance
    )

    return model


def construct_import_costs(b_period, data, period: str):
    """
    Calculates the total import costs for an investment period

    .. math::
        C_{import} = \sum(p_{t, import} * F_{t, import})

    :param b_period: pyomo block for period
    :param data: DataHandle
    :param str period: investment period to calculate import cost for
    :return: pyomo constraint
    """
    config = data.model_config

    set_t = get_set_t(config, b_period)
    hour_factors = get_hour_factors(config, data, period)
    nr_timesteps_averaged = get_nr_timesteps_averaged(config)

    def init_cost_import(const):
        return b_period.var_cost_imports == sum(
            sum(
                sum(
                    b_period.node_blocks[node].var_import_flow[t, car]
                    * b_period.node_blocks[node].para_import_price[t, car]
                    * nr_timesteps_averaged
                    * hour_factors[t - 1]
                    for car in b_period.node_blocks[node].set_carriers
                )
                for t in set_t
            )
            for node in b_period.node_blocks
        )

    return pyo.Constraint(rule=init_cost_import)


def construct_export_costs(b_period, data, period):
    """
    Calculates the total export costs for an investment period

    .. math::
        C_{export} = \sum(p_{t, export} * F_{t, export})

    :param b_period: pyomo block for period
    :param data: DataHandle
    :param str period: investment period to calculate export cost for
    :return: pyomo constraint
    """
    config = data.model_config

    set_t = get_set_t(config, b_period)
    hour_factors = get_hour_factors(config, data, period)
    nr_timesteps_averaged = get_nr_timesteps_averaged(config)

    def init_cost_export(const):
        return b_period.var_cost_exports == -sum(
            sum(
                sum(
                    b_period.node_blocks[node].var_export_flow[t, car]
                    * b_period.node_blocks[node].para_export_price[t, car]
                    * nr_timesteps_averaged
                    * hour_factors[t - 1]
                    for car in b_period.node_blocks[node].set_carriers
                )
                for t in set_t
            )
            for node in b_period.node_blocks
        )

    return pyo.Constraint(rule=init_cost_export)


def construct_system_cost(model, data):
    """
    Aggregates costs per investment period

    - Total capex of technologies
    - Total capex of networks
    - Total opex of technologies
    - Total opex of networks
    - Total cost of technologies (sum of opex and capex)
    - Total cost of networks (sum of opex and capex)
    - Total import costs
    - Total export costs
    - Total costs from violations of the energy balance
    - Carbon costs and revenues
    - Total cost per investment period as a sum of technology, network, import,
      export, violation and carbon costs

    :param model: pyomo model
    :param dict config: dict containing model information
    :return: pyomo model
    """
    config = data.model_config

    def init_period_cost(b_period_cost, period):
        b_period = model.periods[period]
        set_t = get_set_t(config, b_period)
        hour_factors = get_hour_factors(config, data, period)
        nr_timesteps_averaged = get_nr_timesteps_averaged(config)

        # Capex Tecs
        def init_cost_capex_tecs(const):
            return b_period.var_cost_capex_tecs == sum(
                sum(
                    b_period.node_blocks[node].tech_blocks_active[tec].var_capex_tot
                    for tec in b_period.node_blocks[node].set_technologies
                )
                for node in model.set_nodes
            )

        b_period_cost.const_capex_tecs = pyo.Constraint(rule=init_cost_capex_tecs)

        # Capex Networks
        def init_cost_capex_netws(const):
            if not config["energybalance"]["copperplate"]["value"]:
                return b_period.var_cost_capex_netws == sum(
                    b_period.network_block[netw].var_capex
                    for netw in b_period.set_networks
                )
            else:
                return b_period.var_cost_capex_netws == 0

        b_period_cost.const_capex_netw = pyo.Constraint(rule=init_cost_capex_netws)

        # Opex Tecs
        def init_cost_opex_tecs(const):
            tec_opex_variable = sum(
                sum(
                    sum(
                        b_period.node_blocks[node]
                        .tech_blocks_active[tec]
                        .var_opex_variable_tot[t]
                        * nr_timesteps_averaged
                        * hour_factors[t - 1]
                        for tec in b_period.node_blocks[node].set_technologies
                    )
                    for t in set_t
                )
                for node in model.set_nodes
            )

            tec_opex_fixed = sum(
                sum(
                    b_period.node_blocks[node]
                    .tech_blocks_active[tec]
                    .var_opex_fixed_tot
                    for tec in b_period.node_blocks[node].set_technologies
                )
                for node in model.set_nodes
            )

            return b_period.var_cost_opex_tecs == tec_opex_fixed + tec_opex_variable

        b_period_cost.const_opex_tecs = pyo.Constraint(rule=init_cost_opex_tecs)

        # Opex Networks
        def init_cost_opex_netws(const):
            if not config["energybalance"]["copperplate"]["value"]:
                netw_opex_variable = sum(
                    sum(
                        b_period.network_block[netw].var_opex_variable[t]
                        * nr_timesteps_averaged
                        * hour_factors[t - 1]
                        for netw in b_period.set_networks
                    )
                    for t in set_t
                )
                netw_opex_fixed = sum(
                    b_period.network_block[netw].var_opex_fixed
                    for netw in b_period.set_networks
                )
                return (
                    b_period.var_cost_opex_netws == netw_opex_fixed + netw_opex_variable
                )
            else:
                return b_period.var_cost_opex_netws == 0

        b_period_cost.const_opex_netw = pyo.Constraint(rule=init_cost_opex_netws)

        # Total technology costs
        def init_cost_tecs(const):
            return (
                b_period.var_cost_tecs
                == b_period.var_cost_capex_tecs + b_period.var_cost_opex_tecs
            )

        b_period_cost.const_cost_tecs = pyo.Constraint(rule=init_cost_tecs)

        # Total network costs
        def init_cost_netw(const):
            return (
                b_period.var_cost_netws
                == b_period.var_cost_capex_netws + b_period.var_cost_opex_netws
            )

        b_period_cost.const_cost_netws = pyo.Constraint(rule=init_cost_netw)

        # Total import/export cost
        b_period_cost.const_cost_import = construct_import_costs(b_period, data, period)
        b_period_cost.const_cost_export = construct_export_costs(b_period, data, period)

        # Total violation cost
        def init_violation_cost(const):
            if config["energybalance"]["violation"]["value"] >= 0:
                return (
                    b_period.var_cost_violation
                    == sum(
                        sum(
                            sum(
                                b_period.var_violation[t, car, node]
                                * hour_factors[t - 1]
                                for t in set_t
                            )
                            for car in model.set_carriers
                        )
                        for node in model.set_nodes
                    )
                    * config["energybalance"]["violation"]["value"]
                )
            else:
                return b_period.var_cost_violation == 0

        b_period_cost.const_violation_cost = pyo.Constraint(rule=init_violation_cost)

        # Emission cost and revenues (if applicable)
        def init_carbon_revenue(const):
            revenue_carbon_from_technologies = sum(
                sum(
                    sum(
                        b_period.node_blocks[node]
                        .tech_blocks_active[tec]
                        .var_tec_emissions_neg[t]
                        * nr_timesteps_averaged
                        * hour_factors[t - 1]
                        * b_period.node_blocks[node].para_carbon_subsidy[t]
                        for t in set_t
                    )
                    for tec in b_period.node_blocks[node].set_technologies
                )
                for node in model.set_nodes
            )
            return revenue_carbon_from_technologies == b_period.var_carbon_revenue

        b_period_cost.const_revenue_carbon = pyo.Constraint(rule=init_carbon_revenue)

        def init_carbon_cost(const):
            cost_carbon_from_technologies = sum(
                sum(
                    sum(
                        b_period.node_blocks[node]
                        .tech_blocks_active[tec]
                        .var_tec_emissions_pos[t]
                        * nr_timesteps_averaged
                        * hour_factors[t - 1]
                        * b_period.node_blocks[node].para_carbon_tax[t]
                        for t in set_t
                    )
                    for tec in b_period.node_blocks[node].set_technologies
                )
                for node in model.set_nodes
            )
            cost_carbon_from_carriers = sum(
                sum(
                    b_period.node_blocks[node].var_car_emissions_pos[t]
                    * nr_timesteps_averaged
                    * b_period.node_blocks[node].para_carbon_tax[t]
                    * hour_factors[t - 1]
                    for t in set_t
                )
                for node in model.set_nodes
            )
            if not config["energybalance"]["copperplate"]["value"]:
                cost_carbon_from_networks = sum(
                    sum(
                        sum(
                            b_period.network_block[netw].var_netw_emissions_pos[t, node]
                            * nr_timesteps_averaged
                            * hour_factors[t - 1]
                            * b_period.node_blocks[node].para_carbon_tax[t]
                            for t in set_t
                        )
                        for node in model.set_nodes
                    )
                    for netw in b_period.set_networks
                )
            else:
                cost_carbon_from_networks = 0
            return (
                cost_carbon_from_technologies
                + cost_carbon_from_carriers
                + cost_carbon_from_networks
                == b_period.var_carbon_cost
            )

        b_period_cost.const_cost_carbon = pyo.Constraint(rule=init_carbon_cost)

        def init_total_cost(const):
            return (
                b_period.var_cost_tecs
                + b_period.var_cost_netws
                + b_period.var_cost_imports
                + b_period.var_cost_exports
                + b_period.var_cost_violation
                + b_period.var_carbon_cost
                - b_period.var_carbon_revenue
                == b_period.var_cost_total
            )

        b_period_cost.const_cost = pyo.Constraint(rule=init_total_cost)

        return b_period_cost

    model.block_costbalance = pyo.Block(model.set_periods, rule=init_period_cost)

    return model


def construct_global_balance(model):
    """
    Calculates total npv and total emissions over all investment periods
    :param model: pyomo model
    :return: pyomo model
    """

    def init_npv(const):
        return (
            sum(model.periods[period].var_cost_total for period in model.set_periods)
            == model.var_npv
        )

    model.const_npv = pyo.Constraint(rule=init_npv)

    def init_emissions(const):
        return (
            sum(model.periods[period].var_emissions_net for period in model.set_periods)
            == model.var_emissions_net
        )

    model.const_emissions = pyo.Constraint(rule=init_emissions)

    return model
