import h5py
from pathlib import Path
import os

from pyomo.environ import ConcreteModel
from ..logger import log_event
from ..utilities import get_set_t


def get_summary(model, solution, folder_path: Path, model_info: dict) -> dict:
    """
    Retrieves all variable values relevant for the summary of an optimization run.

    These variables and their values are written to a dictionary.

    :param model: the model for which you want to obtain the results summary.
    :param solution: Pyomo solver results
    :param Path folder_path: folder path of optimization run
    :param dict model_info: information of the last solve done by the model
    :return: a dictionary containing the most important model results (i.e., summary_dict)
    :rtype: dict
    """
    # SUMMARY: create dictionary
    summary_dict = {}

    # summary: retrieve / calculate cost variables
    # Fixme: algorithms
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
    try:
        int(solution.solver(0).wallclock_time)
        time = solution.solver(0).wallclock_time
    except:
        time = 0
    summary_dict["time_total"] = time
    summary_dict["lb"] = solution.problem(0).lower_bound
    summary_dict["ub"] = solution.problem(0).upper_bound
    summary_dict["absolute gap"] = (
        solution.problem(0).upper_bound - solution.problem(0).lower_bound
    )
    summary_dict["solver_status"] = solution.solver.termination_condition.value

    # summary: retrieve / calculate run specs
    summary_dict["objective"] = model_info["config"]["optimization"]["objective"][
        "value"
    ]
    summary_dict["pareto_point"] = model_info["pareto_point"]
    summary_dict["monte_carlo_run"] = model_info["monte_carlo_run"]
    summary_dict["time_stage"] = model_info["time_stage"]

    summary_dict["case"] = model_info["config"]["reporting"]["case_name"]["value"]

    summary_dict["time_stamp"] = str(folder_path)

    return summary_dict


def write_optimization_results_to_h5(model, solution, model_info: dict, data) -> dict:
    """
    Collects the results from the model blocks and writes them to an HDF5 file

    Saving to HDF5 files is done using the h5py library.
    The summary results are returned in a dictionary format for exporting to Excel.
    Overhead (calculation of variables) are placed in the utilities file.

    :param ConcreteModel model: the model for which you want to save the results to an HDF5 file.
    :param solution: Pyomo solver results
    :param dict model_info: information of the last solve done by the model
    :param data: DataHandle object containing all data read in by the DataHandle class.
    :return: a dictionary containing the most important model results (i.e., summary_dict)
    :rtype: dict
    """

    def read_value_from_parameter(para) -> list:
        """
        Reads parameter values from a pyomo parameter

        :param para: pyomo parameter
        :return: values as a list
        :rtype: list
        """
        if para.mutable:
            return [para[t, car].value for t in set_t]
        else:
            return [para[t, car] for t in set_t]

    config = model_info["config"]
    folder_path = model_info["result_folder_path"]

    # LOG
    log_event(f"Writing results to {folder_path}")

    # create the results h5 file in the results folder
    h5_file_path = os.path.join(folder_path, "optimization_results.h5")
    with h5py.File(h5_file_path, mode="w") as f:

        summary_dict = get_summary(model, solution, folder_path, model_info)

        # SUMMARY [g]: convert dictionary to h5 datasets
        summary = f.create_group("summary")
        for key in summary_dict:
            if summary_dict[key] is None:
                value = -1
            else:
                value = summary_dict[key]
            summary.create_dataset(key, data=value)

        # TIME AGGREGATION INFORMATION [g]:
        # K-means specs
        k_means_specs = f.create_group("k_means_specs")
        for investment_period in data.k_means_specs:
            k_means_specs_period = k_means_specs.create_group(investment_period)
            for key in data.k_means_specs[investment_period]:
                k_means_specs_period.create_dataset(
                    key, data=data.k_means_specs[investment_period][key]
                )

        # Topology Information
        topology = f.create_group("topology")
        topology.create_dataset("nodes", data=list(model.set_nodes))
        topology.create_dataset("periods", data=list(model.set_periods))
        topology.create_dataset("carriers", data=list(model.set_carriers))

        # TIME-INDEPENDENT RESULTS (design) [g]
        g_design = f.create_group("design")

        # TIME-INDEPENDENT RESULTS: NETWORKS [g] > within: specific network [g] > within: specific arc of network[g]
        networks_design = g_design.create_group("networks")

        for period in model.set_periods:
            g_period_netw_design = networks_design.create_group(period)

            b_period = model.periods[period]
            set_t = get_set_t(config, b_period)

            if not config["energybalance"]["copperplate"]["value"]:
                for netw_name in b_period.set_networks:
                    netw_specific_group = g_period_netw_design.create_group(netw_name)
                    b_netw = b_period.network_block[netw_name]
                    data.network_data[period][netw_name].write_results_netw_design(
                        netw_specific_group, b_netw
                    )

        # TIME-INDEPENDENT RESULTS: NODES [g]
        nodes_design = g_design.create_group("nodes")
        for period in model.set_periods:
            g_period_node_design = nodes_design.create_group(period)

            # TIME-INDEPENDENT RESULTS: NODES: specific node [g] within: specific technology [g]
            for node_name in model.set_nodes:
                node_specific_group = g_period_node_design.create_group(node_name)
                b_node = b_period.node_blocks[node_name]

                for tec_name in b_node.set_technologies:
                    tec_group = node_specific_group.create_group(tec_name)
                    b_tec = b_node.tech_blocks_active[tec_name]
                    data.technology_data[period][node_name][
                        tec_name
                    ].write_results_tec_design(tec_group, b_tec)

        # TIME-DEPENDENT RESULTS (operation) [g]
        operation = f.create_group("operation")

        # TIME-DEPENDENT RESULTS: NETWORKS [g] > within: specific network [g] > within: specific arc of network [g]
        networks_operation = operation.create_group("networks")

        for period in model.set_periods:
            g_period_netw_operation = networks_operation.create_group(period)

            if not config["energybalance"]["copperplate"]["value"]:
                for netw_name in b_period.set_networks:
                    netw_specific_group = g_period_netw_operation.create_group(
                        netw_name
                    )
                    b_netw = b_period.network_block[netw_name]
                    data.network_data[period][netw_name].write_results_netw_operation(
                        netw_specific_group, b_netw
                    )

        # TECHNOLOGY OPERATION [g] > within: node > specific technology [g]
        tec_operation_group = operation.create_group("technology_operation")
        for period in model.set_periods:
            g_period_tec_operation = tec_operation_group.create_group(period)

            for node_name in model.set_nodes:
                node_specific_group = g_period_tec_operation.create_group(node_name)
                b_node = b_period.node_blocks[node_name]

                for tec_name in b_node.set_technologies:
                    tec_group = node_specific_group.create_group(tec_name)
                    b_tec = b_node.tech_blocks_active[tec_name]
                    data.technology_data[period][node_name][
                        tec_name
                    ].write_results_tec_operation(tec_group, b_tec)

        # ENERGY BALANCE [g] > within: node > specific carrier [g]
        ebalance_group = operation.create_group("energy_balance")

        for period in model.set_periods:
            g_period_ebalance = ebalance_group.create_group(period)

            for node_name in model.set_nodes:
                node_specific_group = g_period_ebalance.create_group(node_name)
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
                        "import_price",
                        data=read_value_from_parameter(node_data.para_import_price),
                    )
                    car_group.create_dataset(
                        "export",
                        data=[node_data.var_export_flow[t, car].value for t in set_t],
                    )
                    car_group.create_dataset(
                        "export_price",
                        data=read_value_from_parameter(node_data.para_export_price),
                    )
                    car_group.create_dataset(
                        "demand", data=[node_data.para_demand[t, car] for t in set_t]
                    )

    return summary_dict
