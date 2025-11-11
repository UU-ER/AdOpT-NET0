import json
import shutil
from pathlib import Path

from pyomo.environ import SolverFactory

from adopt_net0.data_preprocessing.template_creation import create_empty_network_matrix


def get_gurobi_parameters(solveroptions: dict):
    """
    Initiates the gurobi solver and defines solver parameters

    :param dict solveroptions: dict with solver parameters
    :return: Gurobi Solver
    """

    solver = SolverFactory(solveroptions["solver"]["value"], solver_io="python")
    solver.options["TimeLimit"] = solveroptions["timelim"]["value"] * 3600
    solver.options["MIPGap"] = solveroptions["mipgap"]["value"]
    solver.options["MIPFocus"] = solveroptions["mipfocus"]["value"]
    solver.options["Threads"] = solveroptions["threads"]["value"]
    solver.options["NodefileStart"] = solveroptions["nodefilestart"]["value"]
    solver.options["Method"] = solveroptions["method"]["value"]
    solver.options["Heuristics"] = solveroptions["heuristics"]["value"]
    solver.options["Presolve"] = solveroptions["presolve"]["value"]
    solver.options["BranchDir"] = solveroptions["branchdir"]["value"]
    solver.options["LPWarmStart"] = solveroptions["lpwarmstart"]["value"]
    solver.options["IntFeasTol"] = solveroptions["intfeastol"]["value"]
    solver.options["FeasibilityTol"] = solveroptions["feastol"]["value"]
    solver.options["Cuts"] = solveroptions["cuts"]["value"]
    solver.options["NumericFocus"] = solveroptions["numericfocus"]["value"]

    return solver


def get_glpk_parameters(solveroptions: dict):
    """
    Initiates the glpk solver and defines solver parameters

    :param dict solveroptions: dict with solver parameters
    :return: Gurobi Solver
    """
    solver = SolverFactory("glpk")
    solver.options["tol"] = 1e-9
    solver.options["mipgap"] = 1e-9

    return solver


def get_set_t(config: dict, model_block):
    """
    Returns the correct set_t for different clustering options

    :param dict config: config dict
    :param model_block: pyomo block holding set_t_full and set_t_clustered
    :return: set_t
    """
    if config["optimization"]["typicaldays"]["N"]["value"] == 0:
        return model_block.set_t_full
    elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
        return model_block.set_t_clustered
    elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
        return model_block.set_t_full


def get_data_for_investment_period(
    data, investment_period: str, aggregation_model: str
) -> dict:
    """
    Gets data from DataHandle for specific investement_period. Writes it to a dict.

    :param data: data to use
    :param str investment_period: investment period
    :param str aggregation_model: aggregation type
    :return: data of respective investment period
    :rtype: dict
    """
    data_period = {}
    data_period["period_name"] = investment_period
    data_period["topology"] = data.topology
    data_period["technology_data"] = data.technology_data[investment_period]
    data_period["time_series"] = data.time_series[aggregation_model].loc[
        :, investment_period
    ]
    data_period["network_data"] = data.network_data[investment_period]
    data_period["energybalance_options"] = data.energybalance_options[investment_period]
    data_period["config"] = data.model_config
    if data.model_config["optimization"]["typicaldays"]["N"]["value"] != 0:
        data_period["k_means_specs"] = data.k_means_specs[investment_period]
        # data_period["averaged_specs"] = data.averaged_specs[investment_period]
    if data.model_config["performance"]["pressure"]["pressure_on"]["value"] == 1:
        data_period["compressor_data"] = data.compressor_data[investment_period]

    # Hour multiplication factors
    if data.model_config["optimization"]["typicaldays"]["N"]["value"] == 0:
        data_period["hour_factors"] = [1] * len(
            data_period["topology"]["time_index"]["full"]
        )
    elif data.model_config["optimization"]["typicaldays"]["method"]["value"] == 1:
        data_period["hour_factors"] = data_period["k_means_specs"]["factors"]
    elif data.model_config["optimization"]["typicaldays"]["method"]["value"] == 2:
        data_period["hour_factors"] = [1] * len(
            data_period["topology"]["time_index"]["full"]
        )

    # Nr timesteps averaged
    if data.model_config["optimization"]["timestaging"]["value"] != 0:
        data_period["nr_timesteps_averaged"] = data.model_config["optimization"][
            "timestaging"
        ]["value"]
    else:
        data_period["nr_timesteps_averaged"] = 1

    return data_period


def determine_flow_existing_compressors(self, compressor, b_period, node):
    """
    Determines the flow capacity of an existing compressor connection by returning
    the minimum available capacity between the output and input components

    :param compressor: tuple with carrier, component1, component 2
    :param b_period: pyomo block data for period
    :param node: pyomo block data for node
    :return float: minimum capacity between input and output component
    """
    component_output_bound = float("inf")
    component_input_bound = float("inf")
    period_name = b_period.name.split("[")[-1].rstrip("]")
    type_component = [compressor.output_type, compressor.input_type]

    if type_component[0] == "Technology":
        var_output = (
            b_period.node_blocks[node]
            .tech_blocks_active[compressor.output_component]
            .var_output
        )
        component_output_bound = max(var_output[idx].ub for idx in var_output)
    elif type_component[0] == "Network":
        component_output_bound = next(
            iter(
                b_period.network_block[
                    compressor.output_component
                ].para_size_initial.values()
            )
        )
    elif type_component[0] == "Import":
        component_output_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Import limit"]
        )

    elif type_component[0] == "Generic production":
        component_output_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Generic production"]
        )

    if type_component[1] == "Technology":
        var_output = (
            b_period.node_blocks[node]
            .tech_blocks_active[compressor.input_component]
            .var_output
        )
        component_input_bound = max(var_output[idx].ub for idx in var_output)
    elif type_component[1] == "Network":
        component_input_bound = next(
            iter(
                b_period.network_block[
                    compressor.input_component
                ].para_size_initial.values()
            )
        )
    elif type_component[1] == "Demand":
        component_input_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Demand"]
        )
    elif type_component[1] == "Export":
        component_input_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Export limit"]
        )

    size = min(component_output_bound, component_input_bound)

    return size


def installed_capacities_existing(m, interval, prev_interval, casepath):
    """
    Transfer installed capacities from a previous interval to define minimum capacities
    for the next brownfield simulation, updating both technologies and networks. Installed
    compressor capacities for the networks are calculated from the existing network and
    technology capacities, as for all simulations with existing networks.

    This function performs two main tasks:

    1. **Technologies** — For each node, it reads the installed technology sizes from
       the previous interval's solved model and writes them into the corresponding
       `Technologies.json` file of the current interval.

       - The sum of the new or existing capacities of a technology in the previous run
       are stored under the `"existing"` key in the JSON file.

    2. **Networks** — For each network, it determines whether the network was active in
       the previous interval (based on arc sizes). If active, it:

       - Adds the network name to the `"existing"` list in `Networks.json`.
       - Copies `distance.csv` and `connection.csv` from the "new" topology folder to
         the "existing" topology folder (if not already present).
       - Writes a `size.csv` file with the current arc sizes.

       If inactive, it removes the network from `"existing"` in `Networks.json` and,
       if the existing folder exists, overwrites `size.csv` with a zero matrix.

    Parameters
    ----------
    m : dict
        Model dictionary containing interval-specific pyomo model objects. The previous
        interval model is accessed via `m[prev_interval]`.
    interval : str
        Name of the current interval (e.g., `"2030"` or `"Interval_1"`).
    prev_interval : str
        Name of the previous interval from which existing capacities are taken.
    casepath : str or pathlib.Path
        Base path to the case directory containing case study data, including the
        `node_data` and `network_topology` subfolders.
    """

    casepath = Path(casepath)
    prev_model = (
        m[prev_interval]
        .model[m[prev_interval].info_solving_algorithms["aggregation_model"]]
        .periods[prev_interval]
    )

    # Technologies
    for node in prev_model.node_blocks:
        b_node_prev = prev_model.node_blocks[node]

        size_tecs_existing = {}

        # Loop through all technologies
        for tec_name in b_node_prev.set_technologies:

            # Standalone existing technology (no new counterpart)
            if tec_name.endswith("_existing"):
                base_tec_name = tec_name.replace("_existing", "")
                if base_tec_name not in b_node_prev.set_technologies:
                    prev_existing_size = (
                        b_node_prev.tech_blocks_active[tec_name].var_size.value or 0
                    )
                    if prev_existing_size > 1e-6:
                        size_tecs_existing[base_tec_name] = prev_existing_size
                continue  # Skip processing it as a "new" technology

            # New technology case
            prev_tec_size = b_node_prev.tech_blocks_active[tec_name].var_size.value or 0

            existing_tec_name = tec_name + "_existing"
            prev_existing_size = 0

            # Add existing capacity to new capacity
            if existing_tec_name in b_node_prev.set_technologies:
                prev_existing_size = (
                    b_node_prev.tech_blocks_active[existing_tec_name].var_size.value
                    or 0
                )

            if prev_tec_size + prev_existing_size > 1e-6:
                size_tecs_existing[tec_name] = prev_tec_size + prev_existing_size

        # Read the JSON technology file
        json_tec_file_path = (
            casepath / interval / "node_data" / node / "Technologies.json"
        )
        with open(json_tec_file_path, "r") as json_tec_file:
            json_tec = json.load(json_tec_file)

        json_tec["existing"] = size_tecs_existing
        with open(json_tec_file_path, "w") as json_tec_file:
            json.dump(json_tec, json_tec_file, indent=4)

        print(node, size_tecs_existing)

    # Networks
    for network in prev_model.network_block:
        # --- Define paths ---
        folder_topology_existing = (
            casepath / interval / "network_topology" / "existing" / network
        )
        folder_topology_new = casepath / interval / "network_topology" / "new" / network
        json_netw_file_path = casepath / interval / "Networks.json"

        # --- Build network size matrix ---
        netw_size_matrix = create_empty_network_matrix(list(prev_model.node_blocks))
        for arc in prev_model.network_block[network].set_arcs:
            netw_size_matrix.loc[arc] = (
                prev_model.network_block[network].arc_block[arc].var_size.value
            )

        active_network = netw_size_matrix.values.sum() > 0

        # --- Read JSON once ---
        with open(json_netw_file_path, "r") as f:
            json_netw = json.load(f)

        if active_network:
            # Add network to 'existing' if not already there
            if network not in json_netw["existing"]:
                json_netw["existing"].append(network)

            # Create folder and copy files if not yet present
            if not folder_topology_existing.exists():
                folder_topology_existing.mkdir(parents=True, exist_ok=True)
                for fname in ["distance.csv", "connection.csv"]:
                    src = folder_topology_new / fname
                    dst = folder_topology_existing / fname
                    if src.exists():
                        shutil.copy(src, dst)
                    else:
                        print(f"Warning: {src} not found, skipping copy.")

        else:
            # Remove inactive network from 'existing' if present
            if network in json_netw["existing"]:
                json_netw["existing"].remove(network)

        # --- Always overwrite size.csv (zero matrix if inactive) ---
        if folder_topology_existing.exists():
            netw_size_matrix.index.name = ""
            netw_size_matrix.to_csv(
                folder_topology_existing / "size.csv",
                sep=";",
                decimal=".",
                float_format="%.6f",
            )

        # --- Save JSON back ---
        with open(json_netw_file_path, "w") as f:
            json.dump(json_netw, f, indent=4)
