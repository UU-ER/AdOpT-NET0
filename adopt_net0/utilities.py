import json
import shutil
import warnings
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
         The updated remaining lifetimes are written under the ``"remaining_lifetime"`` key.
       - For each node, reads installed technology (new and existing) sizes from the previous interval's
         solved model and writes them into the corresponding `Technologies.json` of the
         current interval under the ``"existing"`` key. The sum of the new and existing
         capacities of a technology in the previous interval is stored as a single value
         under that key.

    2. **Networks** — For each network, it determines whether the network was active in
       the previous interval (based on arc sizes). If active, it:

       - Adds the network name to the ``"existing"`` list in `Networks.json`.
       - Copies ``distance.csv`` and ``connection.csv`` from the "new" topology folder to
         the "existing" topology folder (if not already present).
       - Writes a ``size.csv`` file with the current arc sizes.

       If inactive, it removes the network from ``"existing"`` in `Networks.json` and,
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
        Path to the current interval's case directory (e.g. ``".../Case_Interval_2"``).
    intervals_between_years : list or None
        Full list of years between consecutive intervals. If ``None``, no lifetime check
        is performed and a warning is emitted.
    interval_index : int or None
        Index of the current interval in the intervals list (i.e. ``i`` from the loop).
        Used to select the correct value from ``intervals_between_years``.
    """

    casepath = Path(casepath)

    if intervals_between_years is None:
        warnings.warn(
            "intervals_between_years is not defined. No lifetime check will be performed."
        )

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

        # Read remaining_lifetime from the previous interval's JSON (for carry-over lookup)
        prev_remaining_lifetimes = {}
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
                prev_remaining_lifetimes = json.load(f).get("remaining_lifetime", {})

        # First pass: identify expired _existing technologies.
        # New investments may also expire if their lifetime < years_this_step — checked in second pass.
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

        # Second pass: carry forward sizes and compute remaining lifetimes.
        for tec_name in b_node_prev.set_technologies:

            # --- Standalone existing technology (no new counterpart) ---
            if tec_name.endswith("_existing"):
                if tec_name in expired_tecs:
                    continue
                base_tec_name = tec_name.replace("_existing", "")
                if base_tec_name not in b_node_prev.set_technologies:
                    size = b_node_prev.tech_blocks_active[tec_name].var_size.value or 0
                    if size > 1e-6:
                        size_tecs_existing[base_tec_name] = size
                        if years_this_step is not None:
                            rl = check_component_remaining_lifetime(
                                m,
                                tec_name,
                                years_this_step,
                                prev_interval,
                                prev_remaining_lifetimes,
                                node=node,
                            )
                            if rl is not None:
                                remaining_lifetime_dict[tec_name] = rl
                continue  # Always skip further processing for _existing techs

            # --- New technology ---
            new_size = b_node_prev.tech_blocks_active[tec_name].var_size.value or 0
            existing_tec_name = tec_name + "_existing"
            existing_size = 0

            if (
                existing_tec_name in b_node_prev.set_technologies
                and existing_tec_name not in expired_tecs
            ):
                existing_size = (
                    b_node_prev.tech_blocks_active[existing_tec_name].var_size.value
                    or 0
                )

            total_size = new_size + existing_size
            if total_size <= 1e-6:
                continue

            # Compute remaining lifetimes and check expiry.
            # New investment starts from full economics lifetime.
            # Existing carry-over uses the tracked remaining lifetime.
            if years_this_step is not None:
                lifetimes = []

                if new_size > 1e-6:
                    comp_data = (
                        m[prev_interval]
                        .data.technology_data[prev_interval][node]
                        .get(tec_name)
                    )
                    if comp_data is not None:
                        full_lt = comp_data.economics.get("lifetime")
                        if full_lt is not None:
                            lifetimes.append(full_lt - years_this_step)

                if existing_size > 1e-6:
                    ex_rl = check_component_remaining_lifetime(
                        m,
                        existing_tec_name,
                        years_this_step,
                        prev_interval,
                        prev_remaining_lifetimes,
                        node=node,
                    )
                    if ex_rl is not None:
                        lifetimes.append(ex_rl)

                if lifetimes and min(lifetimes) <= 0:
                    continue  # all vintages expired

                if lifetimes:
                    remaining_lifetime_dict[tec_name] = min(lifetimes)
                    if len(lifetimes) > 1:
                        warnings.warn(
                            f"Node '{node}', technology '{tec_name}': both new "
                            f"({new_size:.3f}) and existing ({existing_size:.3f}) "
                            "vintages active. Using minimum remaining lifetime "
                            f"({min(lifetimes)} years). Consider distinct technology "
                            "names per investment period for precise lifetime tracking."
                        )

            size_tecs_existing[tec_name] = total_size

        json_tec_file_path = (
            casepath / interval / "node_data" / node / "Technologies.json"
        )
        with open(json_tec_file_path, "r") as f:
            json_tec = json.load(f)
        json_tec["existing"] = size_tecs_existing
        if years_this_step is not None:
            json_tec["remaining_lifetime"] = remaining_lifetime_dict
        with open(json_tec_file_path, "w") as f:
            json.dump(json_tec, f, indent=4)

    # -------------------------------------------------------------------------
    # Networks
    # -------------------------------------------------------------------------
    json_netw_file_path = casepath / interval / "Networks.json"
    with open(json_netw_file_path, "r") as f:
        json_netw = json.load(f)

    prev_remaining_lifetimes_netw = {}
    remaining_lifetime_netw_dict = {}
    if years_this_step is not None:
        prev_netw_lifetime_path = (
            casepath.parent
            / ("Case_" + prev_interval)
            / prev_interval
            / "Networks_lifetime.json"
        )
        if prev_netw_lifetime_path.exists():
            with open(prev_netw_lifetime_path) as f:
                prev_remaining_lifetimes_netw = json.load(f).get(
                    "remaining_lifetime", {}
                )

    # First pass: identify expired _existing networks.
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

    # Second pass: carry forward arc sizes and compute remaining lifetimes.
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

        netw_size_matrix = create_empty_network_matrix(list(prev_model.node_blocks))
        for arc in prev_model.network_block[network].set_arcs:
            netw_size_matrix.loc[arc] = (
                prev_model.network_block[network].arc_block[arc].var_size.value or 0
            )

        # --- Standalone existing network (no new counterpart) ---
        if network.endswith("_existing"):
            if network in expired_netws:
                if base_name in json_netw["existing"]:
                    json_netw["existing"].remove(base_name)
                if folder_topology_existing.exists():
                    zero = create_empty_network_matrix(list(prev_model.node_blocks))
                    zero.index.name = ""
                    zero.to_csv(
                        folder_topology_existing / "size.csv",
                        sep=";",
                        decimal=".",
                        float_format="%.6f",
                    )
                continue

            if base_name not in prev_model.network_block:
                active = netw_size_matrix.values.sum() > 1e-6
                if active:
                    if base_name not in json_netw["existing"]:
                        json_netw["existing"].append(base_name)
                    if not folder_topology_existing.exists():
                        folder_topology_existing.mkdir(parents=True, exist_ok=True)
                        for fname in ["distance.csv", "connection.csv"]:
                            src = folder_topology_new / fname
                            if src.exists():
                                shutil.copy(src, folder_topology_existing / fname)
                    netw_size_matrix.index.name = ""
                    netw_size_matrix.to_csv(
                        folder_topology_existing / "size.csv",
                        sep=";",
                        decimal=".",
                        float_format="%.6f",
                    )
                    if years_this_step is not None:
                        rl = check_component_remaining_lifetime(
                            m,
                            network,
                            years_this_step,
                            prev_interval,
                            prev_remaining_lifetimes_netw,
                        )
                        if rl is not None:
                            remaining_lifetime_netw_dict[network] = rl
                else:
                    if base_name in json_netw["existing"]:
                        json_netw["existing"].remove(base_name)
            continue  # Always skip further processing for _existing networks

        # --- New network ---
        new_size_matrix = netw_size_matrix.copy()
        existing_netw_name = network + "_existing"
        existing_size_matrix = create_empty_network_matrix(list(prev_model.node_blocks))

        if (
            existing_netw_name in prev_model.network_block
            and existing_netw_name not in expired_netws
        ):
            for arc in prev_model.network_block[existing_netw_name].set_arcs:
                existing_size_matrix.loc[arc] = (
                    prev_model.network_block[existing_netw_name]
                    .arc_block[arc]
                    .var_size.value
                    or 0
                )

        total_matrix = new_size_matrix + existing_size_matrix
        new_sum = float(new_size_matrix.values.sum())
        existing_sum = float(existing_size_matrix.values.sum())
        active_network = float(total_matrix.values.sum()) > 1e-6

        if active_network:
            if base_name not in json_netw["existing"]:
                json_netw["existing"].append(base_name)
            if not folder_topology_existing.exists():
                folder_topology_existing.mkdir(parents=True, exist_ok=True)
                for fname in ["distance.csv", "connection.csv"]:
                    src = folder_topology_new / fname
                    if src.exists():
                        shutil.copy(src, folder_topology_existing / fname)
                    else:
                        warnings.warn(f"{src} not found, skipping copy.")
            total_matrix.index.name = ""
            total_matrix.to_csv(
                folder_topology_existing / "size.csv",
                sep=";",
                decimal=".",
                float_format="%.6f",
            )

            if years_this_step is not None:
                lifetimes = []
                if new_sum > 1e-6:
                    comp_data = (
                        m[prev_interval].data.network_data[prev_interval].get(network)
                    )
                    if comp_data is not None:
                        full_lt = comp_data.economics.get("lifetime")
                        if full_lt is not None:
                            lifetimes.append(full_lt - years_this_step)
                if existing_sum > 1e-6:
                    ex_rl = check_component_remaining_lifetime(
                        m,
                        existing_netw_name,
                        years_this_step,
                        prev_interval,
                        prev_remaining_lifetimes_netw,
                    )
                    if ex_rl is not None:
                        lifetimes.append(ex_rl)
                if lifetimes:
                    remaining_lifetime_netw_dict[network] = min(lifetimes)
                    if len(lifetimes) > 1:
                        warnings.warn(
                            f"Network '{network}': both new ({new_sum:.3f}) and existing "
                            f"({existing_sum:.3f}) vintages active. Using minimum remaining "
                            f"lifetime ({min(lifetimes)} years). Consider distinct network "
                            "names per investment period for precise lifetime tracking."
                        )
        else:
            if base_name in json_netw["existing"]:
                json_netw["existing"].remove(base_name)
            if folder_topology_existing.exists():
                zero = create_empty_network_matrix(list(prev_model.node_blocks))
                zero.index.name = ""
                zero.to_csv(
                    folder_topology_existing / "size.csv",
                    sep=";",
                    decimal=".",
                    float_format="%.6f",
                )

    with open(json_netw_file_path, "w") as f:
        json.dump(json_netw, f, indent=4)

    if years_this_step is not None:
        netw_lifetime_path = casepath / interval / "Networks_lifetime.json"
        with open(netw_lifetime_path, "w") as f:
            json.dump({"remaining_lifetime": remaining_lifetime_netw_dict}, f, indent=4)


def check_component_remaining_lifetime(
    m, name, years_this_step, prev_interval, prev_remaining_lifetimes, node=None
):
    """
    Computes the remaining lifetime of a technology or network after subtracting the
    years elapsed between two consecutive intervals.

    If the component's remaining lifetime was already tracked in the previous interval's
    JSON (via ``prev_remaining_lifetimes``), that value is used as the starting point.
    Otherwise, the original lifetime from the component's economics data is used
    (i.e. first transition for this component).

    Returns ``None`` if no lifetime is defined for the component (no check performed).
    Returns the new remaining lifetime, which may be zero or negative if expired.

    :param dict m: Model dictionary containing interval-specific ModelHub objects.
    :param str name: Name of the technology or network to check.
    :param int years_this_step: Number of years between the previous and current interval.
    :param str prev_interval: Name of the previous interval.
    :param dict prev_remaining_lifetimes: Remaining lifetimes read from the previous
        interval's JSON under the ``"remaining_lifetime"`` key.
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
            return None

    return prev_remaining - years_this_step
