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


def _expire_vintages(old_vintages, old_rls, years_this_step, label):
    """
    Filter vintage sizes by remaining lifetime, subtracting years_this_step.

    :param dict old_vintages: {interval_name: size} from previous interval
    :param dict old_rls: {interval_name: remaining_lifetime} from previous interval
    :param int years_this_step: years elapsed between intervals
    :param str label: description for warning messages
    :return: (surviving_vintages, surviving_rls) — nested dicts keyed by interval name
    """
    surviving = {}
    surviving_rls = {}
    for vintage_interval, vsize in old_vintages.items():
        vrl = old_rls.get(vintage_interval)
        if vrl is None:
            surviving[vintage_interval] = vsize
        else:
            new_vrl = vrl - years_this_step
            if new_vrl > 0:
                surviving[vintage_interval] = vsize
                surviving_rls[vintage_interval] = new_vrl
            else:
                warnings.warn(
                    f"{label}, vintage '{vintage_interval}': {vsize:.3f} expired and is not carried forward."
                )
    return surviving, surviving_rls


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
    for the next brownfield simulation, updating both technologies and networks.

    .. note::
        Compressor capacities are not set explicitly by this function. They will be
        calculated from the existing network and technology capacities in the next
        optimization, as for all simulations with existing components and pressure levels.

    The function operates in two stages:

    **Stage 1 — Lifetime check (all components)**

    Before carrying any capacity forward, each component's investment vintages are
    checked against their remaining lifetime. Every investment period is tracked as
    a separate entry in the nested ``vintage_sizes`` and ``remaining_lifetime`` dicts
    (both written to ``Technologies.json`` and ``Networks.json``). Any vintage whose
    remaining lifetime has dropped to zero or below is expired and excluded from the
    carry-over. Only surviving vintages proceed to Stage 2.

    If ``intervals_between_years`` is ``None``, this stage is skipped and all
    capacities are carried forward unchanged (no lifetime check performed).

    **Stage 2 — Carry-over**

    1. **Technologies** — For each node, it reads the installed technology sizes from
       the previous interval's solved model and writes them into the corresponding
       `Technologies.json` file of the current interval.

       - The sum of the new or existing capacities of a technology in the previous run
         are stored under the ``"existing"`` key in the JSON file.
       - If lifetime tracking is active, ``vintage_sizes`` and ``remaining_lifetime``
         are also written to ``Technologies.json``.

    2. **Networks** — For each network, it determines whether the network was active in
       the previous interval (based on arc sizes).
       If active, it:

       - Adds the network name to the ``"existing"`` list in ``Networks.json``.
       - Copies ``distance.csv`` and ``connection.csv`` from the "new" topology folder to
         the "existing" topology folder (if not already present).
       - Writes a ``size.csv`` file with the current arc sizes. If lifetime tracking
         is active, one ``size_{interval}.csv`` per surviving vintage is also written,
         and ``vintage_sizes`` and ``remaining_lifetime`` are added to ``Networks.json``.

       If inactive, it removes the network from ``"existing"`` in ``Networks.json`` and,
       if the existing folder exists, overwrites ``size.csv`` with a zero matrix.

    Parameters
    ----------
    m : dict
        Model dictionary containing interval-specific ModelHub objects. The previous
        interval model is accessed via ``m[prev_interval]``.
    interval : str
        Name of the current interval (e.g., ``"Interval_2"``).
    prev_interval : str
        Name of the previous interval from which existing capacities are taken.
    casepath : str or pathlib.Path
        Path to the current interval's case directory (e.g. ``.../Case_Interval_2``),
        containing the ``node_data`` and ``network_topology`` subfolders. The previous
        interval's data is read from ``casepath.parent``.
    intervals_between_years : list of int or None
        List of years between each pair of consecutive intervals, of length
        ``len(intervals) - 1``. Only the element at ``interval_index - 1`` is used
        (the gap between ``prev_interval`` and ``interval``). If ``None``, no lifetime
        check is performed.
    interval_index : int or None
        Zero-based index of the current interval in the intervals list (i.e. ``i``
        from the loop, minimum value ``1``). Used to select the correct step from
        ``intervals_between_years`` via ``intervals_between_years[interval_index - 1]``.
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

    # ─── Technologies ────────────────────────────────────────────────────────
    for node in prev_model.node_blocks:
        b_node_prev = prev_model.node_blocks[node]
        all_tecs = set(b_node_prev.set_technologies)

        size_tecs_existing = {}
        remaining_lifetime_dict = {}
        vintage_sizes_dict = {}

        # Read tracking dicts from previous interval's JSON
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

        # Unique base technology names (strip _existing suffix)
        base_tecs = {
            t.replace("_existing", "") if t.endswith("_existing") else t
            for t in all_tecs
        }

        for base_tec in base_tecs:
            existing_tec = base_tec + "_existing"
            new_size = (
                b_node_prev.tech_blocks_active[base_tec].var_size.value or 0.0
                if base_tec in all_tecs
                else 0.0
            )
            existing_size = (
                b_node_prev.tech_blocks_active[existing_tec].var_size.value or 0.0
                if existing_tec in all_tecs
                else 0.0
            )

            if years_this_step is None:
                total_size = new_size + existing_size
                if total_size > 1e-6:
                    size_tecs_existing[base_tec] = total_size
                continue

            # N-vintage tracking
            old_vintages = prev_vintage_sizes.get(base_tec, {})
            old_rls = prev_remaining_lifetimes.get(base_tec, {})

            if old_vintages:
                surviving, surviving_rls = _expire_vintages(
                    old_vintages,
                    old_rls,
                    years_this_step,
                    f"Node '{node}', technology '{base_tec}'",
                )
            elif existing_size > 1e-6:
                # First transition for pre-existing capacity: initialize tracking
                ex_remaining = check_component_remaining_lifetime(
                    m,
                    existing_tec if existing_tec in all_tecs else base_tec,
                    years_this_step,
                    prev_interval,
                    prev_remaining_lifetimes,
                    node=node,
                )
                init_key = f"{prev_interval}_initial"
                surviving = {}
                surviving_rls = {}
                if ex_remaining is None or ex_remaining > 0:
                    surviving[init_key] = existing_size
                    if ex_remaining is not None:
                        surviving_rls[init_key] = ex_remaining
                else:
                    warnings.warn(
                        f"Node '{node}', technology '{base_tec}': existing capacity "
                        f"{existing_size:.3f} expired at first transition."
                    )
            else:
                surviving = {}
                surviving_rls = {}

            # Add new investment vintage (always starts from full economics lifetime)
            if new_size > 1e-6:
                comp_data = (
                    m[prev_interval]
                    .data.technology_data[prev_interval][node]
                    .get(base_tec)
                )
                full_lt = (
                    _get_component_lifetime(comp_data.economics)
                    if comp_data is not None
                    else None
                )
                surviving[prev_interval] = new_size
                if full_lt is not None:
                    surviving_rls[prev_interval] = full_lt - years_this_step

            total_size = sum(surviving.values())
            if total_size > 1e-6:
                size_tecs_existing[base_tec] = total_size
                vintage_sizes_dict[base_tec] = dict(surviving)
                remaining_lifetime_dict[base_tec] = dict(surviving_rls)

        # Write Technologies.json for current interval
        json_tec_file_path = (
            casepath / interval / "node_data" / node / "Technologies.json"
        )
        with open(json_tec_file_path) as f:
            json_tec = json.load(f)
        json_tec["existing"] = size_tecs_existing
        if years_this_step is not None:
            json_tec["remaining_lifetime"] = remaining_lifetime_dict
            json_tec["vintage_sizes"] = vintage_sizes_dict
        else:
            json_tec.pop("remaining_lifetime", None)
            json_tec.pop("vintage_sizes", None)
        with open(json_tec_file_path, "w") as f:
            json.dump(json_tec, f, indent=4)

    # ─── Networks ────────────────────────────────────────────────────────────
    json_netw_file_path = casepath / interval / "Networks.json"
    with open(json_netw_file_path) as f:
        json_netw = json.load(f)

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
    nodes = list(prev_model.node_blocks)

    def _write_size_csv(matrix, path):
        matrix.index.name = ""
        matrix.to_csv(path, sep=";", decimal=".", float_format="%.6f")

    def _read_size_csv(path):
        return pd.read_csv(path, sep=";", index_col=0)

    def _ensure_existing_folder(folder, folder_new):
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            for fname in ["distance.csv", "connection.csv"]:
                src = folder_new / fname
                if src.exists():
                    shutil.copy(src, folder / fname)
                else:
                    warnings.warn(f"Warning: {src} not found, skipping copy.")

    # Collect unique base network names and their presence flags
    base_networks = {}
    for network in prev_model.network_block:
        base = (
            network.replace("_existing", "")
            if network.endswith("_existing")
            else network
        )
        if base not in base_networks:
            base_networks[base] = {"has_new": False, "has_existing": False}
        if network.endswith("_existing"):
            base_networks[base]["has_existing"] = True
        else:
            base_networks[base]["has_new"] = True

    for base_name, presence in base_networks.items():
        existing_netw_name = base_name + "_existing"

        folder_topology_existing = (
            casepath / interval / "network_topology" / "existing" / base_name
        )
        folder_topology_new = (
            casepath / interval / "network_topology" / "new" / base_name
        )
        prev_folder_existing = (
            casepath.parent
            / ("Case_" + prev_interval)
            / prev_interval
            / "network_topology"
            / "existing"
            / base_name
        )

        # Arc size matrices from previous model
        new_matrix = create_empty_network_matrix(nodes)
        if presence["has_new"]:
            for arc in prev_model.network_block[base_name].set_arcs:
                new_matrix.loc[arc] = (
                    prev_model.network_block[base_name].arc_block[arc].var_size.value
                    or 0
                )
        new_sum = float(new_matrix.values.sum())

        existing_matrix = create_empty_network_matrix(nodes)
        if presence["has_existing"]:
            for arc in prev_model.network_block[existing_netw_name].set_arcs:
                existing_matrix.loc[arc] = (
                    prev_model.network_block[existing_netw_name]
                    .arc_block[arc]
                    .var_size.value
                    or 0
                )
        existing_sum = float(existing_matrix.values.sum())

        if years_this_step is None:
            total_matrix = new_matrix + existing_matrix
            active = float(total_matrix.values.sum()) > 1e-6
            if active:
                if base_name not in json_netw["existing"]:
                    json_netw["existing"].append(base_name)
                _ensure_existing_folder(folder_topology_existing, folder_topology_new)
                _write_size_csv(total_matrix, folder_topology_existing / "size.csv")
            else:
                if base_name in json_netw["existing"]:
                    json_netw["existing"].remove(base_name)
                if folder_topology_existing.exists():
                    _write_size_csv(
                        create_empty_network_matrix(nodes),
                        folder_topology_existing / "size.csv",
                    )
            continue

        # N-vintage tracking
        old_vintages = prev_vintage_sizes_netw.get(base_name, {})
        old_rls = prev_remaining_lifetimes_netw.get(base_name, {})

        # vintage_matrices: arc DataFrames for vintages known in this step
        surviving_vintages = {}
        surviving_rls = {}
        total_matrix = create_empty_network_matrix(nodes)
        vintage_matrices = {}

        if old_vintages:
            surviving_vintages, surviving_rls = _expire_vintages(
                old_vintages, old_rls, years_this_step, f"Network '{base_name}'"
            )
            for vintage_interval in surviving_vintages:
                src_csv = prev_folder_existing / f"size_{vintage_interval}.csv"
                if src_csv.exists():
                    vdf = _read_size_csv(src_csv)
                    total_matrix = total_matrix.add(vdf, fill_value=0)
                    vintage_matrices[vintage_interval] = vdf
                else:
                    warnings.warn(
                        f"Network '{base_name}': vintage CSV '{src_csv.name}' not found. "
                        "Arc sizes for this vintage lost."
                    )
        elif existing_sum > 1e-6:
            # First transition for pre-existing network: initialize tracking
            ex_remaining = check_component_remaining_lifetime(
                m,
                existing_netw_name,
                years_this_step,
                prev_interval,
                prev_remaining_lifetimes_netw,
            )
            init_key = f"{prev_interval}_initial"
            if ex_remaining is None or ex_remaining > 0:
                surviving_vintages[init_key] = existing_sum
                if ex_remaining is not None:
                    surviving_rls[init_key] = ex_remaining
                total_matrix = existing_matrix.copy()
                vintage_matrices[init_key] = existing_matrix.copy()
            else:
                warnings.warn(
                    f"Network '{base_name}': existing capacity expired at first transition."
                )

        # Add new investment vintage
        if new_sum > 1e-6:
            comp_data = m[prev_interval].data.network_data[prev_interval].get(base_name)
            full_lt = (
                _get_component_lifetime(comp_data.economics)
                if comp_data is not None
                else None
            )
            surviving_vintages[prev_interval] = new_sum
            if full_lt is not None:
                surviving_rls[prev_interval] = full_lt - years_this_step
            total_matrix = total_matrix.add(new_matrix, fill_value=0)
            vintage_matrices[prev_interval] = new_matrix.copy()

        active = float(total_matrix.values.sum()) > 1e-6

        if active:
            if base_name not in json_netw["existing"]:
                json_netw["existing"].append(base_name)
            _ensure_existing_folder(folder_topology_existing, folder_topology_new)
            _write_size_csv(total_matrix, folder_topology_existing / "size.csv")

            # Write per-vintage CSVs to current interval's existing folder
            for vintage_interval in surviving_vintages:
                dst_csv = folder_topology_existing / f"size_{vintage_interval}.csv"
                if vintage_interval in vintage_matrices:
                    _write_size_csv(vintage_matrices[vintage_interval], dst_csv)
                else:
                    src_csv = prev_folder_existing / f"size_{vintage_interval}.csv"
                    if src_csv.exists() and not dst_csv.exists():
                        shutil.copy(src_csv, dst_csv)

            vintage_sizes_netw_dict[base_name] = dict(surviving_vintages)
            remaining_lifetime_netw_dict[base_name] = dict(surviving_rls)
        else:
            if base_name in json_netw["existing"]:
                json_netw["existing"].remove(base_name)
            if folder_topology_existing.exists():
                _write_size_csv(
                    create_empty_network_matrix(nodes),
                    folder_topology_existing / "size.csv",
                )

    if years_this_step is not None:
        json_netw["remaining_lifetime"] = remaining_lifetime_netw_dict
        json_netw["vintage_sizes"] = vintage_sizes_netw_dict
    else:
        json_netw.pop("remaining_lifetime", None)
        json_netw.pop("vintage_sizes", None)
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
        prev_remaining = _get_component_lifetime(component_data.economics)
        if prev_remaining is None:
            return None  # No lifetime defined → skip check

    return prev_remaining - years_this_step


def _get_component_lifetime(economics):
    """
    Return the relevant lifetime for a component's economics dict.
    Prefers ``technical_lifetime`` if defined, falls back to ``lifetime``.
    Returns ``None`` if neither is defined.
    """
    lt = economics.get("technical_lifetime")
    return lt if lt is not None else economics.get("lifetime")
