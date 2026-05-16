import json
import shutil
import warnings
from pathlib import Path

import pandas as pd

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


def installed_capacities_existing(
    m,
    interval,
    prev_interval,
    casepath,
    intervals_between_years=None,
    interval_index=None,
):
    """
    Transfer installed capacities from a previous interval to define minimum capacities
    for the next brownfield simulation, updating both technologies and networks. Installed
    compressor capacities for the networks are calculated from the existing network and
    technology capacities, as for all simulations with existing networks.

    This function performs two main tasks:

    1. **Technologies**
       - If `intervals_between_years` is provided, a lifetime check is performed.
       Technologies whose remaining lifetime has reached zero are not carried forward.
       The updated remaining lifetimes are written under the `"remaining_lifetime"` key.

        — For each node, it reads the installed technology sizes from
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
    intervals_between_years : list or None
        Full list of years between consecutive intervals. If None, no lifetime check
        is performed.
    interval_index : int or None
        Index of the current interval in the intervals list (i.e. `i` from the loop).
        Used to select the correct value from `intervals_between_years`.
    """

    casepath = Path(casepath)
    years_this_step = (
        intervals_between_years[interval_index - 1]
        if intervals_between_years is not None
        else None
    )
    prev_model = (
        m[prev_interval]
        .model[m[prev_interval].info_solving_algorithms["aggregation_model"]]
        .periods[prev_interval]
    )

    # Technologies
    for node in prev_model.node_blocks:
        b_node_prev = prev_model.node_blocks[node]

        size_tecs_existing = {}
        remaining_lifetime_dict = {}
        vintage_sizes_dict = {}

        # Read remaining_lifetime and vintage_sizes from the previous interval's JSON
        prev_remaining_lifetimes = {}
        prev_vintage_sizes = {}
        if years_this_step is not None:
            prev_json_path = (
                casepath.parent
                / ("Case_" + prev_interval)
                / prev_interval
                / "node_data"
                / node
                / "Technologies.json"
            )
            with open(prev_json_path) as f:
                prev_json = json.load(f)
                prev_remaining_lifetimes = prev_json.get("remaining_lifetime", {})
                prev_vintage_sizes = prev_json.get("vintage_sizes", {})

        # First pass: identify expired _existing technologies
        # (new investments are always alive — they were just built in prev_interval)
        expired_tecs = set()
        if years_this_step is not None:
            for tec_name in b_node_prev.set_technologies:
                if not tec_name.endswith("_existing"):
                    continue
                new_remaining = check_component_remaining_lifetime(
                    m,
                    tec_name,
                    years_this_step,
                    prev_interval,
                    prev_remaining_lifetimes,
                    node=node,
                )
                if new_remaining is not None and new_remaining <= 0:
                    expired_tecs.add(tec_name)

        # Second pass: build carry-forward sizes and lifetime tracking
        for tec_name in b_node_prev.set_technologies:

            # --- Standalone existing technology (no new counterpart in this interval) ---
            if tec_name.endswith("_existing"):
                if tec_name in expired_tecs:
                    continue
                base_tec_name = tec_name.replace("_existing", "")
                if base_tec_name not in b_node_prev.set_technologies:
                    prev_existing_size = (
                        b_node_prev.tech_blocks_active[tec_name].var_size.value or 0
                    )
                    if prev_existing_size > 1e-6:
                        size_tecs_existing[base_tec_name] = prev_existing_size
                        if years_this_step is not None:
                            ex_remaining = check_component_remaining_lifetime(
                                m,
                                tec_name,
                                years_this_step,
                                prev_interval,
                                prev_remaining_lifetimes,
                                node=node,
                            )
                            if ex_remaining is not None:
                                remaining_lifetime_dict[tec_name] = ex_remaining
                                vintage_sizes_dict[tec_name] = prev_existing_size
                continue  # Always skip further processing for _existing techs

            # --- New technology (has a fresh investment variable in prev_interval) ---
            new_size = b_node_prev.tech_blocks_active[tec_name].var_size.value or 0
            existing_tec_name = tec_name + "_existing"

            # Resolve surviving existing capacity (with sub-vintage breakdown)
            surviving_existing_size = 0
            existing_vintage_min_remaining = None

            if (
                existing_tec_name in b_node_prev.set_technologies
                and existing_tec_name not in expired_tecs
            ):
                merged_existing_size = (
                    b_node_prev.tech_blocks_active[existing_tec_name].var_size.value
                    or 0
                )

                if (
                    years_this_step is not None
                    and merged_existing_size > 1e-6
                    and prev_vintage_sizes
                ):
                    # Sub-vintage A: previous interval's new investment (key = tec_name)
                    # Sub-vintage B: older vintages aggregated (key = existing_tec_name)
                    sv_a_size = prev_vintage_sizes.get(tec_name, 0)
                    sv_b_size = prev_vintage_sizes.get(existing_tec_name, 0)

                    sv_a_rl = prev_remaining_lifetimes.get(tec_name)
                    sv_b_rl = prev_remaining_lifetimes.get(existing_tec_name)

                    sv_a_alive = sv_a_rl is None or (sv_a_rl - years_this_step) > 0
                    sv_b_alive = sv_b_rl is None or (sv_b_rl - years_this_step) > 0

                    surviving_existing_size = (sv_a_size if sv_a_alive else 0) + (
                        sv_b_size if sv_b_alive else 0
                    )

                    expired_portion = merged_existing_size - surviving_existing_size
                    if expired_portion > 1e-6:
                        warnings.warn(
                            f"Node '{node}', technology '{tec_name}': "
                            f"{expired_portion:.3f} of existing capacity expired and is not carried forward.",
                            UserWarning,
                            stacklevel=2,
                        )

                    # Min remaining lifetime of surviving sub-vintages
                    surviving_rls = []
                    if sv_a_alive and sv_a_rl is not None:
                        surviving_rls.append(sv_a_rl - years_this_step)
                    if sv_b_alive and sv_b_rl is not None:
                        surviving_rls.append(sv_b_rl - years_this_step)
                    existing_vintage_min_remaining = (
                        min(surviving_rls) if surviving_rls else None
                    )

                else:
                    # No sub-vintage info — fall back to single-vintage check
                    surviving_existing_size = merged_existing_size
                    if years_this_step is not None:
                        ex_remaining = check_component_remaining_lifetime(
                            m,
                            existing_tec_name,
                            years_this_step,
                            prev_interval,
                            prev_remaining_lifetimes,
                            node=node,
                        )
                        existing_vintage_min_remaining = ex_remaining

            # Warn if both vintages are active (same technology name, different investment periods)
            if new_size > 1e-6 and surviving_existing_size > 1e-6:
                warnings.warn(
                    f"Node '{node}', technology '{tec_name}': both new ({new_size:.3f}) "
                    f"and existing ({surviving_existing_size:.3f}) vintages are active. "
                    "Sizes are merged. Consider using distinct technology names per investment period.",
                    UserWarning,
                    stacklevel=2,
                )

            total_size = new_size + surviving_existing_size
            if total_size > 1e-6:
                size_tecs_existing[tec_name] = total_size

            if years_this_step is not None:
                # New investment vintage: always starts from full economics lifetime
                if new_size > 1e-6:
                    comp_data = (
                        m[prev_interval]
                        .data.technology_data[prev_interval][node]
                        .get(tec_name)
                    )
                    if comp_data is not None:
                        full_lt = comp_data.economics.get("lifetime")
                        if full_lt is not None:
                            remaining_lifetime_dict[tec_name] = (
                                full_lt - years_this_step
                            )
                            vintage_sizes_dict[tec_name] = new_size

                # Surviving existing vintage: carry forward with minimum remaining lifetime
                if (
                    surviving_existing_size > 1e-6
                    and existing_vintage_min_remaining is not None
                ):
                    remaining_lifetime_dict[existing_tec_name] = (
                        existing_vintage_min_remaining
                    )
                    vintage_sizes_dict[existing_tec_name] = surviving_existing_size

        # Write the JSON technology file for the current interval
        json_tec_file_path = (
            casepath / interval / "node_data" / node / "Technologies.json"
        )
        with open(json_tec_file_path, "r") as json_tec_file:
            json_tec = json.load(json_tec_file)

        json_tec["existing"] = size_tecs_existing
        if years_this_step is not None:
            json_tec["remaining_lifetime"] = remaining_lifetime_dict
            json_tec["vintage_sizes"] = vintage_sizes_dict
        with open(json_tec_file_path, "w") as json_tec_file:
            json.dump(json_tec, json_tec_file, indent=4)

    # Networks
    json_netw_file_path = casepath / interval / "Networks.json"

    with open(json_netw_file_path, "r") as f:
        json_netw = json.load(f)

    # Read remaining_lifetime and vintage_sizes from the previous interval's Networks.json
    prev_remaining_lifetimes_netw = {}
    prev_vintage_sizes_netw = {}
    if years_this_step is not None:
        prev_netw_json_path = (
            casepath.parent
            / ("Case_" + prev_interval)
            / prev_interval
            / "Networks.json"
        )
        with open(prev_netw_json_path) as f:
            prev_netw_json = json.load(f)
            prev_remaining_lifetimes_netw = prev_netw_json.get("remaining_lifetime", {})
            prev_vintage_sizes_netw = prev_netw_json.get("vintage_sizes", {})

    remaining_lifetime_netw_dict = {}
    vintage_sizes_netw_dict = {}

    # First pass: identify expired _existing networks
    expired_netws = set()
    if years_this_step is not None:
        for network in prev_model.network_block:
            if not network.endswith("_existing"):
                continue
            new_remaining = check_component_remaining_lifetime(
                m,
                network,
                years_this_step,
                prev_interval,
                prev_remaining_lifetimes_netw,
            )
            if new_remaining is not None and new_remaining <= 0:
                expired_netws.add(network)

    def _write_size_csv(matrix, path):
        matrix.index.name = ""
        matrix.to_csv(path, sep=";", decimal=".", float_format="%.6f")

    def _read_size_csv(path):
        return pd.read_csv(path, sep=";", index_col=0)

    def _ensure_existing_folder(folder, folder_new):
        """Create existing topology folder and copy distance/connection CSVs if missing."""
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            for fname in ["distance.csv", "connection.csv"]:
                src = folder_new / fname
                if src.exists():
                    shutil.copy(src, folder / fname)
                else:
                    warnings.warn(
                        f"Warning: {src} not found, skipping copy.",
                        UserWarning,
                        stacklevel=2,
                    )

    # Second pass: build carry-forward arc sizes and lifetime tracking
    for network in prev_model.network_block:
        base_name = (
            network.replace("_existing", "")
            if network.endswith("_existing")
            else network
        )

        folder_topology_existing = (
            casepath / interval / "network_topology" / "existing" / base_name
        )
        folder_topology_new = (
            casepath / interval / "network_topology" / "new" / base_name
        )

        # Build arc size matrix for this specific network block
        netw_size_matrix = create_empty_network_matrix(list(prev_model.node_blocks))
        for arc in prev_model.network_block[network].set_arcs:
            netw_size_matrix.loc[arc] = (
                prev_model.network_block[network].arc_block[arc].var_size.value or 0
            )

        # --- Standalone existing network (no new counterpart in this interval) ---
        if network.endswith("_existing"):
            if network in expired_netws:
                if base_name in json_netw["existing"]:
                    json_netw["existing"].remove(base_name)
                if folder_topology_existing.exists():
                    zero_matrix = create_empty_network_matrix(
                        list(prev_model.node_blocks)
                    )
                    _write_size_csv(zero_matrix, folder_topology_existing / "size.csv")
                continue

            if base_name not in prev_model.network_block:
                # Truly standalone: base network not invested in this interval
                active = netw_size_matrix.values.sum() > 1e-6
                if active:
                    if base_name not in json_netw["existing"]:
                        json_netw["existing"].append(base_name)
                    _ensure_existing_folder(
                        folder_topology_existing, folder_topology_new
                    )
                    _write_size_csv(
                        netw_size_matrix, folder_topology_existing / "size.csv"
                    )
                    if years_this_step is not None:
                        ex_remaining = check_component_remaining_lifetime(
                            m,
                            network,
                            years_this_step,
                            prev_interval,
                            prev_remaining_lifetimes_netw,
                        )
                        if ex_remaining is not None:
                            remaining_lifetime_netw_dict[network] = ex_remaining
                            vintage_sizes_netw_dict[network] = float(
                                netw_size_matrix.values.sum()
                            )
                            _write_size_csv(
                                netw_size_matrix,
                                folder_topology_existing / "size_existing.csv",
                            )
                else:
                    if base_name in json_netw["existing"]:
                        json_netw["existing"].remove(base_name)
            continue  # Always skip further processing for _existing networks

        # --- New network (has investment variable in prev_interval) ---
        new_size_matrix = netw_size_matrix.copy()
        existing_netw_name = network + "_existing"

        # Resolve surviving existing arc sizes (with sub-vintage breakdown)
        nodes = list(prev_model.node_blocks)
        surviving_existing_matrix = create_empty_network_matrix(nodes)
        existing_vintage_min_remaining = None

        if (
            existing_netw_name in prev_model.network_block
            and existing_netw_name not in expired_netws
        ):
            merged_existing_matrix = create_empty_network_matrix(nodes)
            for arc in prev_model.network_block[existing_netw_name].set_arcs:
                merged_existing_matrix.loc[arc] = (
                    prev_model.network_block[existing_netw_name]
                    .arc_block[arc]
                    .var_size.value
                    or 0
                )
            merged_sum = float(merged_existing_matrix.values.sum())

            if (
                years_this_step is not None
                and merged_sum > 1e-6
                and prev_vintage_sizes_netw
            ):
                # Sub-vintage A: previous interval's new investment (key = network base name)
                # Sub-vintage B: older vintages aggregated (key = existing_netw_name)
                sv_a_total = prev_vintage_sizes_netw.get(network, 0)
                sv_b_total = prev_vintage_sizes_netw.get(existing_netw_name, 0)

                sv_a_rl = prev_remaining_lifetimes_netw.get(network)
                sv_b_rl = prev_remaining_lifetimes_netw.get(existing_netw_name)

                sv_a_alive = sv_a_rl is None or (sv_a_rl - years_this_step) > 0
                sv_b_alive = sv_b_rl is None or (sv_b_rl - years_this_step) > 0

                # Reconstruct surviving arc matrix from per-vintage CSV files
                surviving_existing_matrix = create_empty_network_matrix(nodes)
                if sv_a_alive and sv_a_total > 1e-6:
                    sv_a_csv = folder_topology_existing / "size_new.csv"
                    if sv_a_csv.exists():
                        surviving_existing_matrix = surviving_existing_matrix.add(
                            _read_size_csv(sv_a_csv), fill_value=0
                        )
                if sv_b_alive and sv_b_total > 1e-6:
                    sv_b_csv = folder_topology_existing / "size_existing.csv"
                    if sv_b_csv.exists():
                        surviving_existing_matrix = surviving_existing_matrix.add(
                            _read_size_csv(sv_b_csv), fill_value=0
                        )

                expired_portion = merged_sum - float(
                    surviving_existing_matrix.values.sum()
                )
                if expired_portion > 1e-6:
                    warnings.warn(
                        f"Network '{network}': {expired_portion:.3f} of existing arc "
                        "capacity expired and is not carried forward.",
                        UserWarning,
                        stacklevel=2,
                    )

                surviving_rls = []
                if sv_a_alive and sv_a_rl is not None:
                    surviving_rls.append(sv_a_rl - years_this_step)
                if sv_b_alive and sv_b_rl is not None:
                    surviving_rls.append(sv_b_rl - years_this_step)
                existing_vintage_min_remaining = (
                    min(surviving_rls) if surviving_rls else None
                )

            else:
                # No sub-vintage info — fall back to single-vintage check
                surviving_existing_matrix = merged_existing_matrix
                if years_this_step is not None:
                    ex_remaining = check_component_remaining_lifetime(
                        m,
                        existing_netw_name,
                        years_this_step,
                        prev_interval,
                        prev_remaining_lifetimes_netw,
                    )
                    existing_vintage_min_remaining = ex_remaining

        # Warn if both vintages are active
        new_sum = float(new_size_matrix.values.sum())
        surviving_existing_sum = float(surviving_existing_matrix.values.sum())
        if new_sum > 1e-6 and surviving_existing_sum > 1e-6:
            warnings.warn(
                f"Network '{network}': both new ({new_sum:.3f}) and existing "
                f"({surviving_existing_sum:.3f}) vintages are active. Arc sizes are merged. "
                "Consider using distinct network names per investment period.",
                UserWarning,
                stacklevel=2,
            )

        total_matrix = new_size_matrix + surviving_existing_matrix
        active_network = float(total_matrix.values.sum()) > 1e-6

        if active_network:
            if base_name not in json_netw["existing"]:
                json_netw["existing"].append(base_name)
            _ensure_existing_folder(folder_topology_existing, folder_topology_new)
            _write_size_csv(total_matrix, folder_topology_existing / "size.csv")

            if years_this_step is not None:
                # New investment vintage: always starts from full economics lifetime
                if new_sum > 1e-6:
                    comp_data = (
                        m[prev_interval].data.network_data[prev_interval].get(network)
                    )
                    if comp_data is not None:
                        full_lt = comp_data.economics.get("lifetime")
                        if full_lt is not None:
                            remaining_lifetime_netw_dict[network] = (
                                full_lt - years_this_step
                            )
                            vintage_sizes_netw_dict[network] = new_sum
                            _write_size_csv(
                                new_size_matrix,
                                folder_topology_existing / "size_new.csv",
                            )

                # Surviving existing vintage: carry forward with minimum remaining lifetime
                if (
                    surviving_existing_sum > 1e-6
                    and existing_vintage_min_remaining is not None
                ):
                    remaining_lifetime_netw_dict[existing_netw_name] = (
                        existing_vintage_min_remaining
                    )
                    vintage_sizes_netw_dict[existing_netw_name] = surviving_existing_sum
                    _write_size_csv(
                        surviving_existing_matrix,
                        folder_topology_existing / "size_existing.csv",
                    )
        else:
            if base_name in json_netw["existing"]:
                json_netw["existing"].remove(base_name)
            if folder_topology_existing.exists():
                zero_matrix = create_empty_network_matrix(list(prev_model.node_blocks))
                _write_size_csv(zero_matrix, folder_topology_existing / "size.csv")

    if years_this_step is not None:
        json_netw["remaining_lifetime"] = remaining_lifetime_netw_dict
        json_netw["vintage_sizes"] = vintage_sizes_netw_dict
    with open(json_netw_file_path, "w") as f:
        json.dump(json_netw, f, indent=4)


def check_component_remaining_lifetime(
    m, name, years_this_step, prev_interval, prev_remaining_lifetimes, node=None
):
    """
    Computes the remaining lifetime of a technology or network after subtracting the
    years elapsed between two consecutive intervals.

    If the component's remaining lifetime was already tracked in the previous interval's
    JSON (via `prev_remaining_lifetimes`), that value is used as the starting point.
    Otherwise, the original lifetime from the component's economics data is used
    (i.e. first transition for this component).

    Returns None if no lifetime is defined for the component (no check performed).
    Returns the new remaining lifetime, which may be zero or negative if expired.

    :param dict m: Model dictionary containing interval-specific ModelHub objects.
    :param str name: Name of the technology or network to check.
    :param int years_this_step: Number of years between the previous and current interval.
    :param str prev_interval: Name of the previous interval.
    :param dict prev_remaining_lifetimes: Remaining lifetimes read from the previous
        interval's JSON under the `"remaining_lifetime"` key.
    :param str | None node: Node name. If provided, data is read from technology_data;
        if None, data is read from network_data.
    :return float | None: New remaining lifetime, or None if no lifetime is defined.
    """
    if name in prev_remaining_lifetimes:
        prev_remaining = prev_remaining_lifetimes[name]
    elif (
        name.endswith("_existing")
        and name.replace("_existing", "") in prev_remaining_lifetimes
    ):
        prev_remaining = prev_remaining_lifetimes[name.replace("_existing", "")]
    else:
        if node is not None:
            component_data = (
                m[prev_interval].data.technology_data[prev_interval][node].get(name)
            )
        else:
            component_data = m[prev_interval].data.network_data[prev_interval].get(name)

        if component_data is None:
            return None
        prev_remaining = component_data.economics.get("lifetime")
        if prev_remaining is None:
            return None  # No lifetime defined → skip check

    return prev_remaining - years_this_step
