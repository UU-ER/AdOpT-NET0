"""
Solves the NL hydrogen case, with or without the linepack.

The case is built for a full year by ``build.py``, so the horizon of a run is chosen
here with ``--hours``. The default is the first week of the year, which is
consecutive, so the cyclic linepack balance closes on a real chronology.

Two network types can be solved. Both use the same topology, the same frozen backbone,
the same candidate corridors and the same costs, and differ only in the network type:

- ``fixed_size_pipeline``: the flow of an arc is bounded by its capacity and nothing
  else, the pipeline holds no fluid
- ``fluidynamic_pipeline``: the flow follows the quasi-dynamic pipeline equation and
  the fluid stored in the pipeline can be used as a short term storage

The difference between the two objectives is what the linepack is worth in this
system. The difference between the two designs of the small clusters, i.e. which
corridor is built with which pipeline type and how much local storage goes with it,
is where it comes from.

What is solved, which corridors it may build, whether the linepack run is warm started
and every gurobi parameter are command line options::

    python run.py                                  both, warm started, 168 h
    python run.py --case linepack --no-warmstart   the linepack run on its own
    python run.py --case reference --hours 72      the reference on three days
    python run.py --threads 8 --timelim 2 --mipfocus 1 --cuts 2
    python run.py --layout my_layout.json          the corridors that file declares
    python run.py --case linepack --freeze reference     operate the reference design
    python run.py --freeze userData/20260915220834-1     operate an earlier design

A solver option that is not given keeps the value ``build.py`` wrote into
``ConfigModel.json``, so the command line carries only what differs from the case.

``--layout`` and ``--freeze`` rewrite which corridors exist and which ones can be
built, see :mod:`layout`. A layout with no candidate leaves no install binary in the
model, which is the operational problem of a design decided before, and ``--freeze``
writes exactly that out of a solved design. The case is put back as ``build.py`` wrote
it when the runs are over.

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

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import adopt_net0 as adopt

import layout

SEP = ";"
PERIOD = "period1"
CARRIER = "hydrogen"

BACKBONE = "H2Pipeline_backbone"
CANDIDATE_TYPES = ["H2Pipeline_large", "H2Pipeline_medium", "H2Pipeline_small"]
SMALL_NODES = ["Arnhem", "Dordrecht", "Venlo"]

#: The two network types compared, the reference first.
REFERENCE = "fixed_size_pipeline"
LINEPACK = "fluidynamic_pipeline"

#: What each network type is called in the tables.
LABELS = {REFERENCE: "fixed", LINEPACK: "fluidynamic"}

#: ``--case`` value -> the network types it solves, in the order they are solved.
CASES = {
    "reference": [REFERENCE],
    "linepack": [LINEPACK],
    "both": [REFERENCE, LINEPACK],
}

#: How many of the most recent runs the refreshed page carries.
VIEWER_RUNS = 12

#: One consecutive week, starting on 1 January.
DEFAULT_HOURS = 168

#: Technology name -> column of res_capacity_factors.csv
RES_SERIES = {"WindTurbine_Onshore_4000": "wind", "Photovoltaic": "solar"}

#: The solver options of ``ConfigModel.json`` that the command line can override,
#: with the type they are read as and what they do. ``get_gurobi_parameters``
#: (``utilities.py:4-27``) hands exactly this list to the solver, so a parameter
#: outside it cannot be set from here.
SOLVER_OPTIONS = {
    "timelim": (float, "time limit, in hours"),
    "mipgap": (float, "relative MIP gap to stop at"),
    "threads": (int, "number of threads"),
    "mipfocus": (int, "0 balanced, 1 feasibility, 2 optimality, 3 bound"),
    "heuristics": (float, "fraction of the time spent in MIP heuristics"),
    "presolve": (int, "-1 auto, 0 off, 1 conservative, 2 aggressive"),
    "cuts": (int, "-1 auto, 0 off, up to 3 very aggressive"),
    "numericfocus": (int, "0 auto, up to 3 most careful"),
    "method": (int, "algorithm used for the continuous relaxations"),
    "branchdir": (int, "-1 down first, 0 auto, 1 up first"),
    "lpwarmstart": (int, "how the LP warm start information is used"),
    "nodefilestart": (float, "memory in GB before nodes are written to disk"),
    "intfeastol": (float, "integer feasibility tolerance"),
    "feastol": (float, "feasibility tolerance of the constraints"),
    "solver": (str, "solver name, gurobi or glpk"),
}


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


def set_precise_directions(input_path: Path, precise: bool):
    """
    Makes the reference forbid a corridor carrying both ways at the same timestep.

    ``bidirectional_network_precise`` adds the disjunction of ``network.py:883``, which
    is what the linepack model gets for free from its direction binary. The case is
    built with it off, so the reference may send hydrogen both ways down one corridor
    in the same hour: physically nonsense, and cheaper than anything the linepack model
    is allowed to do. With it on, the two runs differ only by the pipeline equation and
    the linepack, which is the difference the case is meant to measure.

    :param Path input_path: input data folder
    :param bool precise: whether one direction at a time is enforced
    """
    for name in [BACKBONE] + CANDIDATE_TYPES:
        netw_file = input_path / PERIOD / "network_data" / f"{name}.json"
        netw_data = json.loads(netw_file.read_text())
        netw_data["Performance"]["bidirectional_network_precise"] = int(precise)
        netw_file.write_text(json.dumps(netw_data, indent=2), encoding="utf-8")


def override_solver_options(pyhub, solver_options: dict) -> dict:
    """
    Writes the solver options of the command line into the configuration of the hub.

    ``solve`` builds the solver out of ``data.model_config`` when it is called
    (``modelhub.py:409-417``), so setting the values after ``read_data`` is enough and
    the case files are left as ``build.py`` wrote them.

    :param pyhub: model hub, after reading the data and before solving
    :param dict solver_options: option name of ``ConfigModel.json`` -> value, where a
        value of ``None`` means the one of the case is kept
    :return: the options that changed, each with the value it had before
    """
    config = pyhub.data.model_config["solveroptions"]
    changed = {}
    for option, value in solver_options.items():
        if value is None:
            continue
        if option not in config:
            raise KeyError(f"{option} is not a solver option of the case")
        if config[option]["value"] == value:
            continue
        changed[option] = (config[option]["value"], value)
        config[option]["value"] = value
    return changed


def apply_layout(input_path: Path, new_layout: dict, backup: Path = None) -> Path:
    """
    Writes a layout into the case, keeping the backup of the layout it started with.

    A run can write two of them, one asked for on the command line and one frozen out
    of the reference solution, and it is the first backup that holds the case as
    ``build.py`` wrote it.

    :param Path input_path: input data folder
    :param dict new_layout: corridors per type, see :mod:`layout`
    :param Path backup: backup of an earlier call, when there was one
    :return: the backup to restore at the end of the run
    """
    written = layout.write(input_path, new_layout)
    if backup is None:
        return written
    layout.discard(written)
    return backup


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
    hours: int = DEFAULT_HOURS,
    start: dict = None,
    solver_options: dict = None,
    precise: bool = False,
) -> dict:
    """
    Reads, constructs and solves the case for one network type.

    :param Path input_path: input data folder
    :param str network_type: ``fixed_size_pipeline`` or ``fluidynamic_pipeline``
    :param pd.DataFrame capacity_factors: capacity factors of the full year
    :param int hours: length of the horizon
    :param dict start: a solution of the other network type, used as a warm start
    :param dict solver_options: solver options overriding the ones of the case
    :param bool precise: whether the reference forbids a corridor carrying both ways at
        the same timestep, see :func:`set_precise_directions`. The linepack model
        enforces it through its own direction binary, so the flag only touches the
        reference
    :return: objective, design of the small clusters, arcs built, and a start
    """
    set_network_type(input_path, network_type)
    set_precise_directions(input_path, precise and network_type == REFERENCE)

    pyhub = adopt.ModelHub()
    pyhub.read_data(str(input_path), start_period=0, end_period=hours)
    override_capacity_factors(pyhub, capacity_factors, hours)

    changed = override_solver_options(pyhub, solver_options or {})
    for option, (before, after) in changed.items():
        print(f"solver option {option}: {before} -> {after}")

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


def report(results: dict, same_layout: bool = True):
    """
    Prints the runs that were solved next to each other.

    :param dict results: result of every network type that was solved
    :param bool same_layout: whether every run saw the same corridors, which
        ``--freeze reference`` makes false
    """
    solved = [network_type for network_type in CASES["both"] if network_type in results]
    columns = "".join(f"{LABELS[network_type]:>14}" for network_type in solved)

    print("\n" + "=" * 72)
    print(f"{'network type':26}{'cost':>18}")
    print("-" * 72)
    for network_type in solved:
        print(f"{network_type:26}{results[network_type]['cost']:18.6g}")
    print("=" * 72)

    if len(solved) == 2:
        reference = results[REFERENCE]["cost"]
        delta = results[LINEPACK]["cost"] - reference
        print(
            f"\nlinepack changes the objective by {delta:.6g} "
            f"({delta / abs(reference):.3%})"
        )
        if not same_layout:
            print(
                "the two runs did not see the same corridors: the reference decided "
                "the design and the linepack run only operated it"
            )

    print("\ndesign of the small clusters")
    keys = sorted(set().union(*(results[t]["design"] for t in solved)))
    print(f"{'node':12}{'technology':28}{columns}")
    for node, tec in keys:
        sizes = "".join(
            f"{results[t]['design'].get((node, tec), 0.0):14.0f}" for t in solved
        )
        print(f"{node:12}{tec:28}{sizes}")

    print("\narcs built")
    keys = sorted(set().union(*(results[t]["network"] for t in solved)))
    print(f"{'network':26}{'arc':32}{columns}")
    for name, arc in keys:
        sizes = "".join(
            f"{results[t]['network'].get((name, arc), {}).get('size', 0.0):14.0f}"
            for t in solved
        )
        label = name.replace("H2Pipeline_", "").replace("_existing", "")
        print(f"{label:26}{arc[0] + ' - ' + arc[1]:32}{sizes}")

    if LINEPACK not in solved:
        return

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


def parse_args(argv=None):
    """
    Reads what to solve, over how long a horizon and with which solver options.

    :param list argv: arguments to read, the command line when left out
    :return: the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Solves the NL hydrogen case, with or without the linepack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        choices=sorted(CASES),
        default="both",
        help="which network type is solved",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS,
        help="length of the horizon, counted from 1 January",
    )
    parser.add_argument(
        "--warmstart",
        dest="warmstart",
        action="store_true",
        default=True,
        help="hand the discrete solution of the reference to the linepack run",
    )
    parser.add_argument(
        "--no-warmstart",
        dest="warmstart",
        action="store_false",
        help="solve every run from scratch",
    )
    parser.add_argument(
        "--precise",
        action="store_true",
        help="make the reference forbid a corridor carrying both ways in the same "
        "hour, which the linepack model forbids anyway. Without it the reference is "
        "cheaper than the physics allows and the comparison flatters the pipeline",
    )
    parser.add_argument(
        "--no-viewer",
        dest="viewer",
        action="store_false",
        help="do not rebuild results.html after the runs",
    )

    group = parser.add_argument_group(
        "layout", "which corridors exist and which ones can be built, see layout.py"
    )
    group.add_argument(
        "--layout",
        metavar="FILE",
        help="layout file written into the case for the runs",
    )
    group.add_argument(
        "--freeze",
        metavar="SOURCE",
        help="make the arcs of a design existing and offer no candidate, i.e. operate "
        "a design instead of deciding it. SOURCE is a result folder, or 'reference' "
        "to use the design of the reference run of this call",
    )
    group.add_argument(
        "--dump-layout",
        metavar="FILE",
        help="write the layout the case carries to FILE and stop",
    )

    group = parser.add_argument_group(
        "solver options", "an option left out keeps the value of ConfigModel.json"
    )
    for option, (option_type, description) in SOLVER_OPTIONS.items():
        group.add_argument(
            f"--{option}", type=option_type, default=None, help=description
        )

    return parser.parse_args(argv)


def main(argv=None):
    """
    Solves what the command line asks for and reports it.

    :param list argv: arguments to read, the command line when left out
    """
    args = parse_args(argv)
    solver_options = {option: getattr(args, option) for option in SOLVER_OPTIONS}

    case_dir = Path(__file__).parent
    input_path = case_dir / "input_data"

    if args.dump_layout:
        layout.dump(layout.case_layout(input_path), args.dump_layout)
        print(f"layout of the case written to {args.dump_layout}")
        return

    capacity_factors = pd.read_csv(
        case_dir / "res_capacity_factors.csv", sep=SEP, index_col=0
    )

    # the reference solves in seconds and its discrete decisions are a feasible point
    # for the linepack model, which on its own struggles to find one at all, so a warm
    # started linepack run needs the reference even when it was not asked for
    sequence = list(CASES[args.case])
    if args.warmstart and LINEPACK in sequence and REFERENCE not in sequence:
        print("warm start asked for: the reference is solved first to provide it")
        sequence.insert(0, REFERENCE)

    # the layout of the case is put back whatever happens, so that a run that fails
    # does not leave the next one with a topology it did not ask for
    backup = None
    results = {}
    try:
        if args.layout:
            chosen = layout.read(args.layout, input_path)
            backup = apply_layout(input_path, chosen, backup)
            print(f"\nlayout of {args.layout}\n{layout.describe(chosen)}")

        if args.freeze and args.freeze != "reference":
            frozen = layout.from_design(layout.design_from_results(Path(args.freeze)))
            backup = apply_layout(input_path, frozen, backup)
            print(f"\ndesign frozen from {args.freeze}\n{layout.describe(frozen)}")

        start = None
        for network_type in sequence:
            print(f"\n=== {network_type} ===")
            results[network_type] = run(
                input_path,
                network_type,
                capacity_factors,
                hours=args.hours,
                start=start if args.warmstart else None,
                solver_options=solver_options,
                precise=args.precise,
            )
            start = results[network_type]["start"]

            if args.freeze == "reference" and network_type == REFERENCE:
                frozen = layout.from_design(results[REFERENCE]["network"])
                backup = apply_layout(input_path, frozen, backup)
                print(
                    f"\ndesign frozen from the reference run\n{layout.describe(frozen)}"
                )
    finally:
        if backup is not None:
            layout.restore(input_path, backup)
        # Leave the case in the state build.py wrote it in
        set_network_type(input_path, LINEPACK)
        set_precise_directions(input_path, False)

    report(results, same_layout=args.freeze != "reference")
    if args.viewer:
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
