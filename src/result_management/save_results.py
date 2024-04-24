import h5py

from .utilities import *


def get_summary(model, solution, folder_path, model_info):
    """
    Retrieves all variable values relevant for the summary of an optimization run.

    :param energyhub: EnergyHub
    :param folder_path: folder path of optimization run
    :return:
    """
    # SUMMARY: create dictionary
    summary_dict = {}

    # summary: retrieve / calculate cost variables
    summary_dict["total_npv"] = model.var_npv.value
    summary_dict["cost_capex_tecs"] = sum(
        model.periods[period].var_cost_capex_tecs.value for period in model.set_periods
    )

    summary_dict["cost_capex_netws"] = sum(
        model.periods[period].var_cost_capex_netws.value for period in model.set_periods
    )
    summary_dict["cost_opex_tecs"] = sum(
        model.periods[period].var_cost_opex_tecs.value for period in model.set_periods
    )
    summary_dict["cost_opex_netws"] = sum(
        model.periods[period].var_cost_opex_netws.value for period in model.set_periods
    )
    summary_dict["cost_tecs"] = sum(
        model.periods[period].var_cost_tecs.value for period in model.set_periods
    )
    summary_dict["cost_netws"] = sum(
        model.periods[period].var_cost_netws.value for period in model.set_periods
    )
    summary_dict["cost_imports"] = sum(
        model.periods[period].var_cost_imports.value for period in model.set_periods
    )
    summary_dict["cost_exports"] = sum(
        model.periods[period].var_cost_exports.value for period in model.set_periods
    )
    summary_dict["violation_cost"] = sum(
        model.periods[period].var_cost_violation.value for period in model.set_periods
    )
    summary_dict["carbon_revenue"] = sum(
        model.periods[period].var_carbon_revenue.value for period in model.set_periods
    )
    summary_dict["carbon_cost"] = sum(
        model.periods[period].var_carbon_cost.value for period in model.set_periods
    )
    summary_dict["total_cost"] = sum(
        model.periods[period].var_cost_total.value for period in model.set_periods
    )

    summary_dict["emissions_pos"] = sum(
        model.periods[period].var_emissions_pos.value for period in model.set_periods
    )
    summary_dict["emissions_neg"] = sum(
        model.periods[period].var_emissions_neg.value for period in model.set_periods
    )
    summary_dict["emissions_net"] = sum(
        model.periods[period].var_emissions_net.value for period in model.set_periods
    )

    # summary: retrieve / calculate solver status
    summary_dict["time_total"] = solution.solver(0).wallclock_time
    summary_dict["lb"] = solution.problem(0).lower_bound
    summary_dict["ub"] = solution.problem(0).upper_bound
    summary_dict["absolute gap"] = (
        solution.problem(0).upper_bound - solution.problem(0).lower_bound
    )

    # summary: retrieve / calculate run specs
    summary_dict["objective"] = model_info["config"]["optimization"]["objective"][
        "value"
    ]
    summary_dict["solver_status"] = solution.solver.termination_condition.value
    summary_dict["pareto_point"] = model_info["pareto_point"]
    summary_dict["monte_carlo_run"] = model_info["monte_carlo_run"]

    # Fixme: averaging algorithm
    # time_stage = get_time_stage(energyhub)
    # summary_dict["time_stage"] = time_stage
    summary_dict["time_stamp"] = str(folder_path)

    return summary_dict


def write_optimization_results_to_h5(model, solution, folder_path, model_info, data):
    """
    Collects the results from the model blocks and writes them to an HDF5 file using the h5py library.
    The summary results are returned in a dictionary format for further processing into an excel in the energyhub.
    Overhead (calculation of variables) are placed in the utilities file.

    :param energyhub:
    :return: summary_dict
    """

    # create the results h5 file in the results folder
    h5_file_path = os.path.join(folder_path, "optimization_results.h5")
    with h5py.File(h5_file_path, mode="w") as f:

        config = model_info["config"]
        summary_dict = get_summary(model, solution, folder_path, model_info)

        # SUMMARY [g]: convert dictionary to h5 datasets
        summary = f.create_group("summary")
        for key in summary_dict:
            summary.create_dataset(key, data=summary_dict[key])

        # Topology Information
        topology = f.create_group("topology")
        topology.create_dataset("nodes", data=list(model.set_nodes))
        topology.create_dataset("carriers", data=list(model.set_carriers))

        aggregation_type = "full"

        for period in model.set_periods:
            g_period = f.create_group(period)

            # TIME-INDEPENDENT RESULTS (design) [g]
            g_design = g_period.create_group("design")

            # TIME-INDEPENDENT RESULTS: NETWORKS [g] > within: specific network [g] > within: specific arc of network[g]
            networks_design = g_design.create_group("networks")

            b_period = model.periods[period]
            set_t = b_period.set_t_full

            if not config["energybalance"]["copperplate"]["value"]:
                for netw_name in b_period.set_networks:
                    netw_specific_group = networks_design.create_group(netw_name)
                    b_netw = b_period.network_block[netw_name]
                    data.network_data[aggregation_type][period][
                        netw_name
                    ].write_netw_design_results_to_group(netw_specific_group, b_netw)

            # TIME-INDEPENDENT RESULTS: NODES [g]
            nodes_design = g_design.create_group("nodes")

            # TIME-INDEPENDENT RESULTS: NODES: specific node [g] within: specific technology [g]
            for node_name in model.set_nodes:
                node_specific_group = nodes_design.create_group(node_name)
                b_node = b_period.node_blocks[node_name]

                for tec_name in b_node.set_technologies:
                    tec_group = node_specific_group.create_group(tec_name)
                    b_tec = b_node.tech_blocks_active[tec_name]
                    data.technology_data[aggregation_type][period][node_name][
                        tec_name
                    ].write_tec_design_results_to_group(tec_group, b_tec)

            # TIME-DEPENDENT RESULTS (operation) [g]
            operation = f.create_group("operation")

            # TIME-DEPENDENT RESULTS: NETWORKS [g] > within: specific network [g] > within: specific arc of network [g]
            networks_operation = operation.create_group("networks")

            if not config["energybalance"]["copperplate"]["value"]:
                for netw_name in b_period.set_networks:
                    netw_specific_group = networks_operation.create_group(netw_name)
                    b_netw = b_period.network_block[netw_name]
                    data.network_data[aggregation_type][period][
                        netw_name
                    ].write_netw_operation_results_to_group(netw_specific_group, b_netw)

            # TECHNOLOGY OPERATION [g] > within: node > specific technology [g]
            tec_operation_group = operation.create_group("technology_operation")
            for node_name in model.set_nodes:
                node_specific_group = tec_operation_group.create_group(node_name)
                b_node = b_period.node_blocks[node_name]

                for tec_name in b_node.set_technologies:
                    tec_group = node_specific_group.create_group(tec_name)
                    b_tec = b_node.tech_blocks_active[tec_name]
                    data.technology_data[aggregation_type][period][node_name][
                        tec_name
                    ].write_tec_operation_results_to_group(tec_group, b_tec)

            # ENERGY BALANCE [g] > within: node > specific carrier [g]
            ebalance_group = operation.create_group("energy_balance")
            for node_name in model.set_nodes:
                node_specific_group = ebalance_group.create_group(node_name)
                b_node = b_period.node_blocks[node_name]

                for car in b_node.set_carriers:
                    car_group = node_specific_group.create_group(car)
                    node_data = b_node

                    technology_inputs = [
                        sum(
                            node_data.tech_blocks_active[tec].var_input[t, car].value
                            for tec in node_data.set_technologies
                            if car
                            in node_data.tech_blocks_active[tec].set_input_carriers
                        )
                        for t in set_t
                    ]
                    car_group.create_dataset(
                        "technology_inputs", data=technology_inputs
                    )
                    technology_outputs = [
                        sum(
                            node_data.tech_blocks_active[tec].var_output[t, car].value
                            for tec in node_data.set_technologies
                            if car
                            in node_data.tech_blocks_active[tec].set_output_carriers
                        )
                        for t in set_t
                    ]
                    car_group.create_dataset(
                        "technology_outputs", data=technology_outputs
                    )
                    car_group.create_dataset(
                        "generic_production",
                        data=[
                            node_data.var_generic_production[t, car].value
                            for t in set_t
                        ],
                    )
                    car_group.create_dataset(
                        "network_inflow",
                        data=[
                            0 if x is None else x
                            for x in [
                                node_data.var_netw_inflow[t, car].value for t in set_t
                            ]
                        ],
                    )
                    car_group.create_dataset(
                        "network_outflow",
                        data=[
                            0 if x is None else x
                            for x in [
                                node_data.var_netw_outflow[t, car].value for t in set_t
                            ]
                        ],
                    )
                    if hasattr(node_data, "var_netw_consumption"):
                        network_consumption = [
                            node_data.var_netw_consumption[t, car].value for t in set_t
                        ]
                        car_group.create_dataset(
                            "network_consumption", data=network_consumption
                        )
                    car_group.create_dataset(
                        "import",
                        data=[node_data.var_import_flow[t, car].value for t in set_t],
                    )
                    car_group.create_dataset(
                        "export",
                        data=[node_data.var_export_flow[t, car].value for t in set_t],
                    )
                    car_group.create_dataset(
                        "demand", data=[node_data.para_demand[t, car] for t in set_t]
                    )

    return summary_dict
