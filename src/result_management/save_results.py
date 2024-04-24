import h5py

from .utilities import *


def get_summary(energyhub, folder_path):
    """
    Retrieves all variable values relevant for the summary of an optimization run.

    :param energyhub: EnergyHub
    :param folder_path: folder path of optimization run
    :return:
    """
    model = energyhub.model

    # SUMMARY: create dictionary
    summary_dict = {}

    # summary: retrieve / calculate cost variables
    summary_dict["total_costs"] = model.var_total_cost.value
    summary_dict["carbon_costs"] = model.var_carbon_cost.value
    summary_dict["carbon_revenues"] = model.var_carbon_revenue.value
    tec_costs = calculate_tec_cost(energyhub)
    summary_dict["technology_cost"] = tec_costs
    summary_dict["network_cost"] = model.var_netw_cost.value
    import_costs = calculate_import_costs(energyhub)
    summary_dict["import_cost"] = import_costs
    export_revenues = calculate_export_revenues(energyhub)
    summary_dict["export_revenues"] = export_revenues
    violation_cost = calculate_violation_cost(energyhub)
    summary_dict["violation_cost"] = violation_cost

    # summary: retrieve / calculate emission variables
    summary_dict["net_emissions"] = model.var_emissions_net.value
    summary_dict["positive_emissions"] = model.var_emissions_pos.value
    summary_dict["negative_emissions"] = model.var_emissions_neg.value
    net_emissions_from_tecs = calculate_net_emissions_from_tecs(energyhub)
    summary_dict["net_emissions_from_technologies"] = net_emissions_from_tecs
    net_emissions_from_cars = calculate_net_emissions_from_carriers(energyhub)
    summary_dict["net_emissions_from_carriers"] = net_emissions_from_cars
    net_emissions_from_netw = calculate_net_emissions_from_netw(energyhub)
    summary_dict["net_emissions_from_networks"] = net_emissions_from_netw

    # summary: retrieve / calculate solver status
    summary_dict["time_total"] = energyhub.solution.solver(0).wallclock_time
    summary_dict["lb"] = energyhub.solution.problem(0).lower_bound
    summary_dict["ub"] = energyhub.solution.problem(0).upper_bound
    summary_dict["absolute gap"] = (
        energyhub.solution.problem(0).upper_bound
        - energyhub.solution.problem(0).lower_bound
    )

    # summary: retrieve / calculate run specs
    summary_dict["objective"] = config["optimization"]["objective"]["value"]
    summary_dict["solver_status"] = (
        energyhub.solution.solver.termination_condition.value
    )
    summary_dict["pareto_point"] = energyhub.model_information.pareto_point
    summary_dict["monte_carlo_run"] = energyhub.model_information.monte_carlo_run

    time_stage = get_time_stage(energyhub)
    summary_dict["time_stage"] = time_stage
    summary_dict["time_stamp"] = str(folder_path)

    return summary_dict


def write_optimization_results_to_h5(model, folder_path):
    """
    Collects the results from the model blocks and writes them to an HDF5 file using the h5py library.
    The summary results are returned in a dictionary format for further processing into an excel in the energyhub.
    Overhead (calculation of variables) are placed in the utilities file.

    :param energyhub:
    :return: summary_dict
    """
    set_t = model.set_t_full

    # create the results h5 file in the results folder
    h5_file_path = os.path.join(folder_path, "optimization_results.h5")
    with h5py.File(h5_file_path, mode="w") as f:

        summary_dict = get_summary(energyhub, folder_path)

        # SUMMARY [g]: convert dictionary to h5 datasets
        summary = f.create_group("summary")
        for key in summary_dict:
            summary.create_dataset(key, data=summary_dict[key])

        # Topology Information
        topology = f.create_group("topology")
        topology.create_dataset("nodes", data=list(model.set_nodes))
        topology.create_dataset("carriers", data=list(model.set_carriers))

        # TIME-INDEPENDENT RESULTS (design) [g]
        design = f.create_group("design")

        # TIME-INDEPENDENT RESULTS: NETWORKS [g] > within: specific network [g] > within: specific arc of network[g]
        networks_design = design.create_group("networks")

        if not config["energybalance"]["copperplate"]["value"]:
            for netw_name in model.set_networks:
                netw_specific_group = networks_design.create_group(netw_name)
                b_netw = model.network_block[netw_name]
                energyhub.data.network_data[
                    netw_name
                ].write_netw_design_results_to_group(netw_specific_group, b_netw)

        # TIME-INDEPENDENT RESULTS: NODES [g]
        nodes_design = design.create_group("nodes")

        # TIME-INDEPENDENT RESULTS: NODES: specific node [g] within: specific technology [g]
        for node_name in model.set_nodes:
            node_specific_group = nodes_design.create_group(node_name)

            for tec_name in model.node_blocks[node_name].set_tecsAtNode:
                tec_group = node_specific_group.create_group(tec_name)
                b_tec = model.node_blocks[node_name].tech_blocks_active[tec_name]
                energyhub.data.technology_data[node_name][
                    tec_name
                ].write_tec_design_results_to_group(tec_group, b_tec)

        # TIME-DEPENDENT RESULTS (operation) [g]
        operation = f.create_group("operation")

        # TIME-DEPENDENT RESULTS: NETWORKS [g] > within: specific network [g] > within: specific arc of network [g]
        networks_operation = operation.create_group("networks")

        if not config["energybalance"]["copperplate"]["value"]:
            for netw_name in model.set_networks:
                netw_specific_group = networks_operation.create_group(netw_name)
                b_netw = model.network_block[netw_name]
                energyhub.data.network_data[
                    netw_name
                ].write_netw_operation_results_to_group(netw_specific_group, b_netw)

        # TECHNOLOGY OPERATION [g] > within: node > specific technology [g]
        tec_operation_group = operation.create_group("technology_operation")
        for node_name in model.set_nodes:
            node_specific_group = tec_operation_group.create_group(node_name)
            for tec_name in model.node_blocks[node_name].set_tecsAtNode:
                tec_group = node_specific_group.create_group(tec_name)
                b_tec = model.node_blocks[node_name].tech_blocks_active[tec_name]
                energyhub.data.technology_data[node_name][
                    tec_name
                ].write_tec_operation_results_to_group(tec_group, b_tec)

        # ENERGY BALANCE [g] > within: node > specific carrier [g]
        ebalance_group = operation.create_group("energy_balance")
        for node_name in model.set_nodes:
            node_specific_group = ebalance_group.create_group(node_name)
            for car in model.node_blocks[node_name].set_carriers:
                car_group = node_specific_group.create_group(car)
                node_data = model.node_blocks[node_name]
                technology_inputs = [
                    sum(
                        node_data.tech_blocks_active[tec].var_input[t, car].value
                        for tec in node_data.set_tecsAtNode
                        if car in node_data.tech_blocks_active[tec].set_input_carriers
                    )
                    for t in set_t
                ]
                car_group.create_dataset("technology_inputs", data=technology_inputs)
                technology_outputs = [
                    sum(
                        node_data.tech_blocks_active[tec].var_output[t, car].value
                        for tec in node_data.set_tecsAtNode
                        if car in node_data.tech_blocks_active[tec].set_output_carriers
                    )
                    for t in set_t
                ]
                car_group.create_dataset("technology_outputs", data=technology_outputs)
                car_group.create_dataset(
                    "generic_production",
                    data=[
                        node_data.var_generic_production[t, car].value for t in set_t
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
                    "demand", data=[node_data.para_demand[t, car].value for t in set_t]
                )

    return summary_dict
