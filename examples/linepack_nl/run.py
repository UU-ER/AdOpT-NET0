"""
Solves the NL hydrogen case, with and without the linepack.

The case is built for a full year by ``build.py``, so the horizon of a run is chosen
here with ``start_period`` and ``end_period``. The default is the first week of the
year, which is consecutive, so the cyclic linepack balance closes on a real
chronology.

Two runs are compared. Both use the same topology, the same frozen backbone, the same
candidate corridors and the same costs, and differ only in the network type:

- ``fixed_size_pipeline``: the flow of an arc is bounded by its capacity and nothing
  else, the pipeline holds no fluid
- ``fluidynamic_pipeline``: the flow follows the quasi-dynamic pipeline equation and
  the fluid stored in the pipeline can be used as a short term storage

The difference between the two objectives is what the linepack is worth in this
system. The difference between the two designs of the small clusters, i.e. which
corridor is built with which pipeline type and how much local storage goes with it,
is where it comes from.

.. note::
    ``FluidynamicPipeline`` derives from ``FixedSizePipeline``, so the two runs offer
    exactly the same infrastructure: the same corridors, the same pipeline types, the
    same capacity per arc derived from the same geometry, and the same investment
    cost. The only difference is the transport itself. The generic ``fluid`` network
    would not do: its size is a continuous decision, so it could right-size every arc
    and the comparison would measure that instead of the linepack.

.. note::
    The capacity factors of wind and PV are read from ``res_capacity_factors.csv``
    and written into the fitted coefficients, because the climate data of the case is
    synthetic. This has to happen between ``read_data`` and ``construct_model``: at
    ``typicaldays.N = 0`` and ``timestaging = 0`` the used coefficients are the full
    resolution ones by reference (``technology.py:417-421``), so patching the full
    resolution ones is picked up everywhere.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import adopt_net0 as adopt

SEP = ";"
PERIOD = "period1"
CARRIER = "hydrogen"

BACKBONE = "H2Pipeline_backbone"
CANDIDATE_TYPES = ["H2Pipeline_large", "H2Pipeline_medium", "H2Pipeline_small"]
SMALL_NODES = ["Arnhem", "Dordrecht", "Venlo"]

#: The two network types compared, the reference first.
REFERENCE = "fixed_size_pipeline"
LINEPACK = "fluidynamic_pipeline"

#: How many of the most recent runs the refreshed page carries.
VIEWER_RUNS = 12

#: One consecutive week, starting on 1 January.
END_PERIOD = 168

#: Technology name -> column of res_capacity_factors.csv
RES_SERIES = {"WindTurbine_Onshore_4000": "wind", "Photovoltaic": "solar"}


def set_network_type(input_path: Path, network_type: str):
    """
    Switches every network of the case between the two types that are compared.

    :param Path input_path: input data folder
    :param str network_type: ``fixed_size_pipeline`` or ``fluidynamic_pipeline``
    """
    for name in [BACKBONE] + CANDIDATE_TYPES:
        netw_file = input_path / PERIOD / "network_data" / f"{name}.json"
        netw_data = json.loads(netw_file.read_text())
        netw_data["network_type"] = network_type
        netw_file.write_text(json.dumps(netw_data, indent=2), encoding="utf-8")


def override_capacity_factors(pyhub, capacity_factors: pd.DataFrame, n_hours: int):
    """
    Replaces the fitted capacity factors with the ones of the study.

    :param pyhub: model hub, after reading the data and before constructing the model
    :param pd.DataFrame capacity_factors: capacity factors of the full year
    :param int n_hours: number of hours of the horizon
    """
    for period in pyhub.data.technology_data:
        for node in pyhub.data.technology_data[period]:
            for name, tec in pyhub.data.technology_data[period][node].items():
                column = next(
                    (v for k, v in RES_SERIES.items() if name.startswith(k)), None
                )
                if column is None:
                    continue
                series = capacity_factors[column].to_numpy()[:n_hours]
                coeff = tec.processed_coeff.time_dependent_full
                if coeff["capfactor"].shape != series.shape:
                    raise ValueError(
                        f"{period}/{node}/{name}: capfactor has shape "
                        f"{coeff['capfactor'].shape} but the horizon has {n_hours} "
                        "hours"
                    )
                coeff["capfactor"] = series.copy()


def collect_design(b_period) -> dict:
    """
    Size chosen for every technology of the small clusters.

    :param b_period: pyomo block of the investment period
    :return: size per node and technology, in MW or MWh
    """
    design = {}
    for node in SMALL_NODES:
        b_node = b_period.node_blocks[node]
        for tec in b_node.set_technologies:
            size = b_node.tech_blocks_active[tec].var_size.value
            if size is not None and size > 1e-6:
                design[(node, tec)] = size
    return design


def collect_network(b_period) -> dict:
    """
    Arcs that are built, with their size and their linepack.

    :param b_period: pyomo block of the investment period
    :return: size and linepack statistics per network and arc
    """
    built = {}
    for name in b_period.network_block:
        b_netw = b_period.network_block[name]
        has_linepack = hasattr(b_netw, "var_linepack")
        for arc in b_netw.set_arcs_unique:
            size = b_netw.arc_block[arc].var_size.value
            if size is None or size <= 1e-6:
                continue
            linepack = None
            if has_linepack:
                linepack = np.array(
                    [b_netw.var_linepack[(t,) + arc].value for t in b_period.set_t_full]
                )
            built[(name, arc)] = {"size": size, "linepack": linepack}
    return built


def collect_start(model) -> dict:
    """
    Takes the discrete part of a solution, to be handed to the next model as a start.

    Keyed by the pyomo name of each variable, which is the same in both models
    wherever the component itself is. The flow of every arc is kept as well: the
    linepack model has a direction binary that the reference has no counterpart for,
    and the sign of the flow is what decides it.

    :param model: solved pyomo model
    :return: values by variable name, and the flow of each arc block
    """
    start = {}
    for var in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if var.value is not None and var.is_binary():
            start[var.name] = round(var.value)

    flows = {}
    b_period = model.periods[PERIOD]
    for name in b_period.network_block:
        b_netw = b_period.network_block[name]
        for arc in b_netw.arc_block:
            b_arc = b_netw.arc_block[arc]
            flows[b_arc.name] = {
                t: (b_arc.var_flow[t].value or 0.0) for t in b_period.set_t_full
            }

    return {"binaries": start, "flows": flows}


def apply_start(model, start: dict) -> tuple:
    """
    Writes a start into a model, so that gurobi has an incumbent from the first node.

    ADOPT already asks the solver for a warm start (``modelhub.py:926``), so setting
    the values is all that is needed. A partial start is allowed: gurobi solves for
    whatever is left. Variables the two models do not share are simply not found.

    :param model: constructed pyomo model, before solving
    :param dict start: the output of :func:`collect_start`
    :return: how many binaries and how many directions were set
    """
    binaries = 0
    for var in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if var.is_binary() and var.name in start["binaries"]:
            var.set_value(start["binaries"][var.name], skip_validation=True)
            binaries += 1

    # The direction has to come from the *net* flow of a corridor. The reference does
    # not forbid flow in both directions of one at the same timestep, it has no
    # per-timestep direction at all, so taking each arc on its own would set both
    # directions to 1 and break "direction_ij + direction_ji <= installed".
    directions = 0
    b_period = model.periods[PERIOD]
    for name in b_period.network_block:
        b_netw = b_period.network_block[name]
        for arc in b_netw.arc_block:
            b_arc = b_netw.arc_block[arc]
            if not hasattr(b_arc, "var_direction"):
                continue
            reverse = b_netw.arc_block[arc[1], arc[0]].name
            forward_flow = start["flows"].get(b_arc.name)
            reverse_flow = start["flows"].get(reverse)
            if forward_flow is None or reverse_flow is None:
                continue
            for t in b_period.set_t_full:
                net = forward_flow.get(t, 0.0) - reverse_flow.get(t, 0.0)
                b_arc.var_direction[t].set_value(
                    1 if net > 1e-6 else 0, skip_validation=True
                )
                directions += 1

    return binaries, directions


def run(
    input_path: Path,
    network_type: str,
    capacity_factors: pd.DataFrame,
    start: dict = None,
) -> dict:
    """
    Reads, constructs and solves the case for one network type.

    :param Path input_path: input data folder
    :param str network_type: ``fixed_size_pipeline`` or ``fluidynamic_pipeline``
    :param pd.DataFrame capacity_factors: capacity factors of the full year
    :param dict start: a solution of the other network type, used as a warm start
    :return: objective, design of the small clusters, arcs built, and a start
    """
    set_network_type(input_path, network_type)

    pyhub = adopt.ModelHub()
    pyhub.read_data(str(input_path), start_period=0, end_period=END_PERIOD)
    override_capacity_factors(pyhub, capacity_factors, END_PERIOD)

    pyhub.construct_model()
    pyhub.construct_balances()

    model = pyhub.model[pyhub.info_solving_algorithms["aggregation_model"]]
    if start:
        binaries, directions = apply_start(model, start)
        print(
            f"warm start: {binaries} binaries and {directions} directions taken "
            "from the reference run"
        )

    pyhub.solve()

    b_period = model.periods[PERIOD]

    return {
        "cost": model.var_npv.value,
        "design": collect_design(b_period),
        "network": collect_network(b_period),
        "start": collect_start(model),
    }


def report(results: dict):
    """
    Prints the two runs next to each other.

    :param dict results: result of each network type
    """
    print("\n" + "=" * 72)
    print(f"{'network type':26}{'cost':>18}")
    print("-" * 72)
    for network_type, result in results.items():
        print(f"{network_type:26}{result['cost']:18.6g}")
    print("=" * 72)

    reference = results[REFERENCE]["cost"]
    delta = results[LINEPACK]["cost"] - reference
    print(
        f"\nlinepack changes the objective by {delta:.6g} "
        f"({delta / abs(reference):.3%})"
    )

    print("\ndesign of the small clusters")
    keys = sorted(set(results[REFERENCE]["design"]) | set(results[LINEPACK]["design"]))
    print(f"{'node':12}{'technology':28}{'fixed':>12}{'fluidynamic':>14}")
    for node, tec in keys:
        a = results[REFERENCE]["design"].get((node, tec), 0.0)
        b = results[LINEPACK]["design"].get((node, tec), 0.0)
        print(f"{node:12}{tec:28}{a:12.0f}{b:14.0f}")

    print("\narcs built")
    keys = sorted(
        set(results[REFERENCE]["network"]) | set(results[LINEPACK]["network"])
    )
    print(f"{'network':26}{'arc':32}{'fixed':>10}{'fluidynamic':>13}")
    for name, arc in keys:
        a = results[REFERENCE]["network"].get((name, arc), {}).get("size", 0.0)
        b = results[LINEPACK]["network"].get((name, arc), {}).get("size", 0.0)
        label = name.replace("H2Pipeline_", "").replace("_existing", "")
        print(f"{label:26}{arc[0] + ' - ' + arc[1]:32}{a:10.0f}{b:13.0f}")

    print("\nlinepack of the arcs that are built")
    print(f"{'network':26}{'arc':32}{'mean MWh':>10}{'swing MWh':>12}")
    for (name, arc), data in results[LINEPACK]["network"].items():
        series = data["linepack"]
        if series is None:
            continue
        label = name.replace("H2Pipeline_", "").replace("_existing", "")
        print(
            f"{label:26}{arc[0] + ' - ' + arc[1]:32}{series.mean():10.0f}"
            f"{series.max() - series.min():12.0f}"
        )


def main():
    """
    Runs both network types and reports the difference.
    """
    case_dir = Path(__file__).parent
    input_path = case_dir / "input_data"
    capacity_factors = pd.read_csv(
        case_dir / "res_capacity_factors.csv", sep=SEP, index_col=0
    )

    # the reference solves in seconds and its discrete decisions are a feasible
    # point for the linepack model, which on its own struggles to find one at all
    results = {}
    start = None
    for network_type in (REFERENCE, LINEPACK):
        print(f"\n=== {network_type} ===")
        results[network_type] = run(input_path, network_type, capacity_factors, start)
        start = results[network_type]["start"]

    # Leave the case in the state build.py wrote it in
    set_network_type(input_path, LINEPACK)
    report(results)
    refresh_viewer()


def refresh_viewer():
    """
    Rebuilds the result page so that it carries the runs just solved.

    The page embeds its data, and a file opened from disk can neither list a folder
    nor fetch a local file, so it cannot pick up a new run by itself. Rebuilding it
    here is what keeps it current without having to remember.
    """
    import viewer

    case_dir = Path(__file__).parent
    output = case_dir / "results.html"
    runs = []
    for folder in viewer.all_results(case_dir)[:VIEWER_RUNS]:
        try:
            runs.append(viewer.read_results(folder, case_dir))
        except (KeyError, OSError) as error:
            # a run of an older shape of the case is not worth failing over
            print(f"  skipped {folder.name}: {type(error).__name__}")
    if not runs:
        print("\nviewer not refreshed: no result folder could be read")
        return
    viewer.write_html(runs, output)
    print(f"\nviewer refreshed with {len(runs)} runs: {output}")


if __name__ == "__main__":
    main()
