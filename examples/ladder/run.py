"""
Solves one rung of the ladder, with or without the linepack.

A rung is built for a full year by ``build.py``, so the horizon of a run is chosen here
with ``--hours``. The default is the first week of the year, which is consecutive, so
the cyclic linepack balance closes on a real chronology.

Two network types can be solved. Both use the same topology, the same backbone, the same
candidate corridors and the same costs, and differ only in the network type:

- ``fixed_size_pipeline``: the flow of an arc is bounded by its capacity and nothing
  else, the pipeline holds no fluid
- ``fluidynamic_pipeline``: the flow follows the quasi-dynamic pipeline equation and the
  fluid stored in the pipeline can be used as a short term storage

The difference between the two objectives is what the linepack is worth in that rung.
The difference between the two designs of the small clusters, i.e. which corridor is
built with which pipeline type and how much local storage goes with it, is where it
comes from.

What is decided and what is given is ``--design``, see :mod:`design`::

    python run.py --rung 1L1S                       both models, the full design problem
    python run.py --rung 1L1S --design network      only the corridors are dimensioned
    python run.py --rung 1L1S --design tecs         only the clusters are dimensioned
    python run.py --rung 1L1S --design none         pure operation, nothing is decided

and where the given part comes from is ``--from``: ``study`` is the design the study
carries, written next to the case by ``build.py``, ``solved`` is the design the
reference run of this same call finds, and anything else is a result folder of an
earlier run.

Everything else is a command line option as well::

    python run.py --rung 2L2S --case linepack --no-warmstart
    python run.py --rung 1L2S --case reference --hours 72
    python run.py --rung 3L3S --threads 8 --timelim 2 --mipfocus 1 --cuts 2
    python run.py --rung 1L1S --layout my_layout.json      the corridors that file has
    python run.py --rung 1L1S --design none --from 1L1S/userData/20260917101411-1

A solver option that is not given keeps the value ``build.py`` wrote into
``ConfigModel.json``, so the command line carries only what differs from the rung.

``--layout``, ``--design`` and ``--from`` rewrite the case before a run and put it back
afterwards, so a run that fails does not leave the next one with a topology or a design
it did not ask for.

.. note::
    ``FluidynamicPipeline`` derives from ``FixedSizePipeline``, so the two runs offer
    exactly the same infrastructure: the same corridors, the same pipeline types, the
    same capacity per arc derived from the same geometry, and the same investment cost.
    The only difference is the transport itself. The generic ``fluid`` network would not
    do: its size is a continuous decision, so it could right-size every arc and the
    comparison would measure that instead of the linepack.

.. note::
    The capacity factors of wind and PV are read from ``res_capacity_factors.csv`` and
    written into the fitted coefficients, because the climate data of the case is
    synthetic. This has to happen between ``read_data`` and ``construct_model``: at
    ``typicaldays.N = 0`` and ``timestaging = 0`` the used coefficients are the full
    resolution ones by reference (``technology.py:417-421``), so patching the full
    resolution ones is picked up everywhere.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import adopt_net0 as adopt

import design
import layout
import rungs

SEP = ";"
PERIOD = "period1"
CARRIER = "hydrogen"

BACKBONE = "H2Pipeline_backbone"
CANDIDATE_TYPES = ["H2Pipeline_large", "H2Pipeline_medium", "H2Pipeline_small"]

#: Pipeline type put on every candidate corridor when the network is given and nothing
#: has been solved to take it from. The medium type covers the peak demand of the two
#: smaller clusters but not of Arnhem, so a rung whose network is given this way is not
#: trivially served and the storage still has something to do.
DEFAULT_GIVEN_TYPE = "H2Pipeline_medium"

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


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """
    Shows the default of every option and leaves the rung table of the epilog alone.
    """


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


def cap_imports(model, cap: float) -> int:
    """
    Puts a ceiling on what every node may import, or forbids importing altogether.

    The direction binary of an arc only permits flow, it never forces it, so the
    solution in which nothing moves and everyone imports stays feasible however the
    rest of the model is set up. On this case that solution is where the linepack model
    lands: 2.86e7 against a reference of 3.77e6 at 96 h, the network idle, every small
    cluster off grid. Taking the import away leaves the model free to route as it likes
    but obliges it to route something, and it then finds the reference optimum exactly.

    :param model: constructed pyomo model
    :param float cap: ceiling per node and timestep, in MW. Zero forbids imports
    :return: how many variables were bounded
    """
    b_period = model.periods[PERIOD]
    bounded = 0
    for node in b_period.node_blocks:
        b_node = b_period.node_blocks[node]
        for t in b_period.set_t_full:
            variable = b_node.var_import_flow[t, CARRIER]
            if variable.ub is None or variable.ub > cap:
                variable.setub(cap)
                bounded += 1
    return bounded


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


def collect_design(b_period, small: list) -> dict:
    """
    Size chosen for every technology of the small clusters.

    :param b_period: pyomo block of the investment period
    :param list small: small clusters of the rung
    :return: size per node and technology, in MW or MWh
    """
    sizes = {}
    for node in small:
        b_node = b_period.node_blocks[node]
        for tec in b_node.set_technologies:
            size = b_node.tech_blocks_active[tec].var_size.value
            if size is not None and size > 1e-6:
                sizes[(node, tec)] = size
    return sizes


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
    small: list,
    hours: int = DEFAULT_HOURS,
    start: dict = None,
    solver_options: dict = None,
    precise: bool = False,
    import_cap: float = None,
) -> dict:
    """
    Reads, constructs and solves the case for one network type.

    :param Path input_path: input data folder
    :param str network_type: ``fixed_size_pipeline`` or ``fluidynamic_pipeline``
    :param pd.DataFrame capacity_factors: capacity factors of the full year
    :param list small: small clusters of the rung, whose design is reported
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
    if import_cap is not None:
        print(
            f"imports capped at {import_cap} MW: {cap_imports(model, import_cap)} bounds"
        )
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
        "design": collect_design(b_period, small),
        "network": collect_network(b_period),
        "start": collect_start(model),
    }


def report(results: dict, same_layout: bool = True):
    """
    Prints the runs that were solved next to each other.

    :param dict results: result of every network type that was solved
    :param bool same_layout: whether every run saw the same corridors, which
        ``--from solved`` makes false
    """
    solved = [network_type for network_type in CASES["both"] if network_type in results]
    columns = "".join(f"{LABELS[network_type]:>14}" for network_type in solved)

    print("\n" + "=" * 72)
    print(f"{'network type':26}{'cost':>18}")
    print("-" * 72)
    for network_type in solved:
        cost = results[network_type]["cost"]
        # a run the solver could not answer carries no objective, which is a result of
        # the rung and not a reason to lose the runs that did solve
        shown = f"{cost:18.6g}" if cost is not None else f"{'no solution':>18}"
        print(f"{network_type:26}{shown}")
    print("=" * 72)

    if len([t for t in solved if results[t]["cost"] is not None]) == 2:
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

    # A corridor given to one run and decided by the other is two network names to
    # ADOPT, the given one carrying the ``_existing`` suffix, so the rows are merged on
    # the pipeline type rather than on the name, or the same corridor appears twice
    print("\narcs built")
    merged = {}
    for network_type in solved:
        for (name, arc), data in results[network_type]["network"].items():
            label = name.replace("H2Pipeline_", "").replace("_existing", "")
            row = merged.setdefault((label, arc), {})
            row[network_type] = max(row.get(network_type, 0.0), data["size"])
    print(f"{'network':26}{'arc':32}{columns}")
    for label, arc in sorted(merged):
        sizes = "".join(f"{merged[(label, arc)].get(t, 0.0):14.0f}" for t in solved)
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


def read_manifest(case_dir: Path) -> dict:
    """
    What ``build.py`` recorded about the rung, i.e. which nodes it carries.

    :param Path case_dir: folder of the rung
    :return: the manifest
    """
    path = case_dir / "rung.json"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing: build the rung first with "
            f"python build.py --rung {case_dir.name}"
        )
    return json.loads(path.read_text())


def given_layout(
    input_path: Path, source: str, given_type: str, manifest: dict
) -> dict:
    """
    The corridors of a network that is given rather than decided.

    ``study`` puts one pipeline of ``given_type`` on every candidate corridor of the
    rung, and keeps the backbone where the rung has one. Nothing is left to build, so
    the run carries no install binary at all. Any other source is a result folder, and
    the network is then the one that run built.

    :param Path input_path: input data folder
    :param str source: ``study``, or a result folder of an earlier run
    :param str given_type: pipeline type put on every corridor, when ``source`` is
        ``study``
    :param dict manifest: what :func:`read_manifest` returned
    :return: corridors per type, under ``existing`` and ``candidates``
    """
    if source != "study":
        return layout.from_design(layout.design_from_results(Path(source)))

    own = layout.own_arcs(input_path)
    existing = {}
    if manifest["backbone"] and own.get(manifest["backbone"]):
        existing[manifest["backbone"]] = own[manifest["backbone"]]
    if own.get(given_type):
        existing[given_type] = own[given_type]
    return {"existing": existing, "candidates": {}}


def given_design(case_dir: Path, source: str, small: list) -> dict:
    """
    The sizes of a cluster that is given rather than dimensioned.

    :param Path case_dir: folder of the rung
    :param str source: ``study``, or a result folder of an earlier run
    :param list small: small clusters of the rung
    :return: ``{node: {technology: size}}``
    """
    if source == "study":
        return design.read(case_dir / "reference_design.json")
    return design.from_results(Path(source), small)


def apply_design(
    input_path: Path, pinned: dict, small: list, backup: Path = None
) -> Path:
    """
    Pins a design into the case, keeping the backup of the one it started with.

    Mirrors :func:`apply_layout`: a run can pin twice, once from the command line and
    once out of the reference solution, and it is the first backup that holds the case
    as ``build.py`` wrote it.

    :param Path input_path: input data folder
    :param dict pinned: ``{node: {technology: size}}``
    :param list small: small clusters of the rung
    :param Path backup: backup of an earlier call, when there was one
    :return: the backup to restore at the end of the run
    """
    written = design.pin(input_path, pinned, small)
    if backup is None:
        return written
    design.discard(written)
    return backup


def parse_args(argv=None):
    """
    Reads what to solve, over how long a horizon and with which solver options.

    :param list argv: arguments to read, the command line when left out
    :return: the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Solves one rung of the ladder, with or without the linepack.",
        formatter_class=HelpFormatter,
        epilog="rungs of the ladder\n" + rungs.describe(),
    )
    parser.add_argument(
        "--rung",
        choices=list(rungs.RUNGS),
        required=True,
        help="which rung is solved, i.e. which node set. Build it first with build.py",
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
        "--import-cap",
        type=float,
        default=None,
        help="ceiling on what a node may import, in MW per hour. Zero forbids it, "
        "which is what stops the model answering with an idle network and imports",
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
    parser.add_argument(
        "--input",
        metavar="FOLDER",
        help="input data folder, so that a run can work on its own copy of the case "
        "instead of the one of the rung (default: the one of the rung)",
    )
    parser.add_argument(
        "--copy-case",
        action="store_true",
        help="copy the case to a folder of this run before touching it, and remove "
        "the copy at the end. Two runs that share one case rewrite each other's "
        "topology: the one that finishes first puts the case back while the other is "
        "still reading it",
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
        "--dump-layout",
        metavar="FILE",
        help="write the layout the case carries to FILE and stop",
    )

    group = parser.add_argument_group(
        "design", "what is dimensioned and what is given, see design.py"
    )
    group.add_argument(
        "--design",
        choices=list(design.MODES),
        default="all",
        help="all: corridors and technologies are decided. network: only the "
        "corridors. tecs: only the technologies. none: neither, i.e. pure operation",
    )
    group.add_argument(
        "--from",
        dest="given_from",
        metavar="SOURCE",
        default="study",
        help="where the given part comes from: 'study' is the design the study "
        "carries, 'solved' is the design the reference run of this call finds, "
        "anything else is a result folder of an earlier run",
    )
    group.add_argument(
        "--given-type",
        choices=CANDIDATE_TYPES,
        default=DEFAULT_GIVEN_TYPE,
        help="pipeline type put on every candidate corridor when the network is given "
        "and --from is 'study', i.e. when nothing has been solved to take it from",
    )
    group.add_argument(
        "--dump-design",
        metavar="FILE",
        help="write the design of a result folder to FILE and stop, see --from",
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

    case_dir = Path(__file__).parent / args.rung
    input_path = Path(args.input) if args.input else case_dir / "input_data"
    manifest = read_manifest(case_dir)
    small = manifest["small"]

    if args.dump_layout:
        layout.dump(layout.case_layout(input_path), args.dump_layout)
        print(f"layout of the case written to {args.dump_layout}")
        return

    if args.dump_design:
        design.dump(given_design(case_dir, args.given_from, small), args.dump_design)
        print(f"design of {args.given_from} written to {args.dump_design}")
        return

    capacity_factors = pd.read_csv(
        case_dir / "res_capacity_factors.csv", sep=SEP, index_col=0
    )

    decide = design.MODES[args.design]

    # "solved" means the given part is taken from the reference run of this call, so it
    # can only be applied once that run is over
    deferred = args.given_from == "solved"
    if deferred and decide["network"] and decide["tecs"]:
        raise SystemExit("--from solved needs a --design that gives something")

    # the reference solves in seconds and its discrete decisions are a feasible point
    # for the linepack model, which on its own struggles to find one at all, so a warm
    # started linepack run needs the reference even when it was not asked for. So does a
    # run that takes its given part from the reference solution
    sequence = list(CASES[args.case])
    if (
        (args.warmstart or deferred)
        and LINEPACK in sequence
        and REFERENCE not in sequence
    ):
        reason = "--from solved" if deferred else "warm start"
        print(f"{reason} asked for: the reference is solved first to provide it")
        sequence.insert(0, REFERENCE)

    # the case is put back whatever happens, so that a run that fails does not leave the
    # next one with a topology or a design it did not ask for
    layout_backup = None
    design_backup = None
    copied = None
    results = {}
    try:
        if args.copy_case:
            copied = case_dir / "userData" / f"case_{os.getpid()}"
            if copied.exists():
                shutil.rmtree(copied)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(input_path, copied)
            input_path = copied
            print(f"\nworking on a copy of the case in {input_path}")

        if args.layout:
            chosen = layout.read(args.layout, input_path)
            layout_backup = apply_layout(input_path, chosen, layout_backup)
            print(f"\nlayout of {args.layout}\n{layout.describe(chosen)}")

        if not decide["network"] and not deferred:
            chosen = given_layout(
                input_path, args.given_from, args.given_type, manifest
            )
            layout_backup = apply_layout(input_path, chosen, layout_backup)
            print(f"\nnetwork given by {args.given_from}\n{layout.describe(chosen)}")

        if not decide["tecs"] and not deferred:
            pinned = given_design(case_dir, args.given_from, small)
            design_backup = apply_design(input_path, pinned, small, design_backup)
            print(
                f"\ntechnologies given by {args.given_from}\n{design.describe(pinned)}"
            )

        start = None
        for network_type in sequence:
            print(f"\n=== {network_type} ===")
            results[network_type] = run(
                input_path,
                network_type,
                capacity_factors,
                small,
                hours=args.hours,
                start=start if args.warmstart else None,
                solver_options=solver_options,
                precise=args.precise,
                import_cap=args.import_cap,
            )
            start = results[network_type]["start"]

            if not deferred or network_type != REFERENCE:
                continue

            if not decide["network"]:
                chosen = layout.from_design(results[REFERENCE]["network"])
                layout_backup = apply_layout(input_path, chosen, layout_backup)
                print(
                    f"\nnetwork given by the reference run\n{layout.describe(chosen)}"
                )
            if not decide["tecs"]:
                pinned = {}
                for (node, tec), size in results[REFERENCE]["design"].items():
                    pinned.setdefault(node, {})[tec] = size
                design_backup = apply_design(input_path, pinned, small, design_backup)
                print(
                    f"\ntechnologies given by the reference run\n"
                    f"{design.describe(pinned)}"
                )
    finally:
        if copied is not None:
            # the copy is the whole isolation, so there is nothing to put back
            shutil.rmtree(copied, ignore_errors=True)
        else:
            if design_backup is not None:
                design.restore(input_path, design_backup)
            if layout_backup is not None:
                layout.restore(input_path, layout_backup)
            # Leave the case in the state build.py wrote it in
            set_network_type(input_path, LINEPACK)
            set_precise_directions(input_path, False)

    report(results, same_layout=not deferred)
    if args.viewer:
        refresh_viewer(case_dir)


def refresh_viewer(case_dir: Path):
    """
    Rebuilds the result page so that it carries the runs just solved.

    The page embeds its data, and a file opened from disk can neither list a folder
    nor fetch a local file, so it cannot pick up a new run by itself. Rebuilding it
    here is what keeps it current without having to remember.

    :param Path case_dir: folder of the rung, which holds its own ``userData``
    """
    import viewer

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
