"""
Solves the linepack case by a bilevel decomposition, master on the binaries.

The idea is the one of Ghilardi et al., *A detailed MILP model and an ad hoc
decomposition algorithm for the operational optimization of gas transport networks*,
Computers & Chemical Engineering 195 (2025) 109006: a master problem that carries the
combinatorics without the expensive physics decides the flow direction of every
pipeline, and a detailed problem that has those directions fixed carries the physics
and returns an operable solution. Two things are done differently here, and both come
from what the case made of the full model (see the notes of 2026-09-15):

- their master is an *approximation*, so it gives no bound. Ours is a **relaxation**:
  the master is the detailed model with the adjacency condition of the interpolation
  dropped (``--master sos2``, the default), or with the pipeline equation and the
  pressure coupling dropped as well (``--master pressure``). Either way the constraints
  only ever come off, so the objective of the master is a valid lower bound of the full
  problem and the detailed problem gives a valid upper bound: the run reports a real
  gap instead of a stopping heuristic. The first relaxation is worth much more — on the
  fixed network at 24 h the second one loses 0.6 % of the bound, because dropping the
  pipeline equation throws away exactly what made the relaxation of the full model as
  strong as the capacity-only reference.
- their master learns from the iterations through integer cuts on the compressor
  stations. Ours learns the physics it is missing: after each master solve, the flows
  it chose are checked against the pressure the pipelines would really need, and every
  path that cannot hold its pressure budget comes back as a linear cut.

**The cut.** The pipeline equation inverted is the pressure a flow costs,

.. math::
    \\Delta p(f) = -p_{ref} + \\sqrt{p_{ref}^2 + f^2 / R}

which is convex in the flow, so every tangent underestimates it. Along a path of arcs
carried in the same direction, the pressure drops add up to the difference between the
two end pressures, which cannot leave the pressure band of the pipeline type:

.. math::
    \\sum_{a \\in P} \\Delta p_a(f_a) = p_{start} - p_{end} \\le p_{max} - p_{min}

Replacing each drop by its tangent keeps the inequality valid, and adding the
direction binary of each arc on the right hand side keeps it valid for the solutions
in which the path is not carried in that direction at all:

.. math::
    \\sum_{a \\in P} \\left[ \\alpha_a f_a + \\beta_a \\right]
    \\le (p_{max} - p_{min}) \\left( 1 + \\sum_{a \\in P} (1 - d_a) \\right)

with :math:`\\alpha_a = \\Delta p_a'(\\hat f_a)` and
:math:`\\beta_a = \\Delta p_a(\\hat f_a) - \\alpha_a \\hat f_a \\le 0`. A cut is
generated only when the master solution violates it, so the master is tightened
exactly where it is wrong.

**Why the master keeps the direction binaries.** A single arc can never break its own
budget: the largest breakpoint of the piecewise approximation is
``pressure_max - pressure_ref``, half of the band. Only a path of two or more arcs can,
and a path is only a path when every arc of it carries flow the same way, which is what
the direction binaries say. That is also the shape the user of this case asked for:
all the binaries in the master, the operation in the subproblem.

**One master solution is a family of subproblems.** A direction is fixed only where the
master carries a flow worth mentioning, as a share of the capacity of the arc — the
remedy Ghilardi et al. report against a subproblem the master makes infeasible, and free
here, since a corridor the master leaves empty is one the subproblem should be allowed
to use. The share is not one number but a knob: a low one hands the subproblem a tight
corridor, a high one leaves it room to reroute, and which is better is not knowable in
advance. They are independent problems, so ``--candidates`` runs several at once and the
round keeps the best upper bound instead of the best of a single guess. The upper bound
is what this case is missing, which is why the cores go here rather than into one solve.

Usage::

    python run_decomposed.py --layout layouts/fixed_network.json --hours 168
    python run_decomposed.py --candidates 0.005,0.01,0.02,0.05 --workers 4 --sub-threads 4
    python run_decomposed.py --input /scratch/case_copy --iterations 10 --timelim 0.25

.. note::
    The subproblem keeps the SOS2 condition of the piecewise approximation. Fixing the
    direction removes the big-M linking of the arc, which is what makes the model hard,
    but the interpolation itself stays exact.

.. note::
    ``--workers`` times ``--sub-threads`` should stay under the cores of the machine,
    and the memory bounds the workers well before the cores do: each 168 h model is a
    few GB. Every subproblem writes into a folder of its own under ``userData`` /
    ``decomposed``, because ADOPT names a result folder after the second it started in
    and appends to a single summary spreadsheet.

.. note::
    ``--input`` points the run at a copy of the case, so that two runs do not rewrite
    each other's topology. The workers of one run never write into the case: the caller
    sets the network type once and they only read.
"""

import argparse
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

import adopt_net0 as adopt

import layout
import run

SEP = run.SEP
PERIOD = run.PERIOD

#: What the master is allowed to drop, by name of the constraint of the arc block.
#: Both are relaxations, so both give a valid lower bound, but they are not worth the
#: same. ``sos2`` drops only the adjacency condition of the interpolation and keeps the
#: pipeline equation itself: the interpolation weights are then a plain convex
#: combination, whose hull is the region between the interpolant and the outer chord,
#: which is most of the physics for none of the combinatorics. ``pressure`` drops the
#: equation and the pressure coupling as well, which is much easier to solve and much
#: weaker — on the fixed network at 24 h it loses 0.6 % of the bound.
MASTER_RELAXATIONS = {
    "sos2": ["const_lambda_sos2"],
    "pressure": [
        "const_delta_pressure_high",
        "const_delta_pressure_low",
        "const_lambda",
        "const_interpolation_delta_pressure",
        "const_interpolation_flow",
        "const_lambda_sos2",
    ],
}

#: Below this share of the capacity of an arc, a flow of the master is not a direction.
#: One master solution gives one subproblem per threshold, and they are independent.
DEFAULT_CANDIDATES = [0.005, 0.01, 0.02, 0.05]

#: Where the subproblems of a round write their results, under the save path of the
#: case. Each one needs a folder of its own: the name ADOPT gives a result folder is
#: the second it started in, and the summary it appends to is a single spreadsheet.
RESULTS_FOLDER = "decomposed"

#: At most this many cuts are added per iteration, the most violated ones first.
MAX_CUTS_PER_ITERATION = 200

#: Longest path the cut generator walks.
MAX_PATH_LENGTH = 4


def build(
    input_path: Path,
    network_type: str,
    capacity_factors: pd.DataFrame,
    hours: int,
    solver_options: dict,
    set_type: bool = True,
    save_path: Path = None,
    write_results: bool = True,
):
    """
    Reads the case and constructs one model, without solving it.

    :param Path input_path: input data folder
    :param str network_type: ``fixed_size_pipeline`` or ``fluidynamic_pipeline``
    :param pd.DataFrame capacity_factors: capacity factors of the full year
    :param int hours: length of the horizon
    :param dict solver_options: solver options overriding the ones of the case
    :param bool set_type: whether the network type is written into the case. A worker
        that shares the case with other workers must not write into it, and does not
        have to: the type is the same for all of them and the caller sets it once
    :param Path save_path: where the results of this solve are written. Concurrent
        solves need one each, since the folder is named after the second it started in
        and the summary is a single spreadsheet
    :param bool write_results: whether the solution is written out. The master is
        solved with the pipeline equation switched off, so its interpolation variables
        come back empty and the result writer of the pipeline has nothing to put in the
        h5 (``fluidynamic_pipeline.py:731``). Its solution is read from the model here
        anyway, so it does not need to be written
    :return: the model hub, constructed and ready to be solved
    """
    if set_type:
        run.set_network_type(input_path, network_type)

    pyhub = adopt.ModelHub()
    pyhub.read_data(str(input_path), start_period=0, end_period=hours)
    run.override_capacity_factors(pyhub, capacity_factors, hours)
    run.override_solver_options(pyhub, solver_options or {})
    pyhub.data.model_config["reporting"]["write_results"]["value"] = int(write_results)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        reporting = pyhub.data.model_config["reporting"]
        reporting["save_path"]["value"] = str(save_path)
        reporting["save_summary_path"]["value"] = str(save_path)

    pyhub.construct_model()
    pyhub.construct_balances()

    return pyhub


def model_of(pyhub):
    """
    The block the solver sees.

    :param pyhub: model hub, after the model is constructed
    :return: the pyomo model
    """
    return pyhub.model[pyhub.info_solving_algorithms["aggregation_model"]]


def relax_pressure(model, relaxation: str = "sos2") -> int:
    """
    Switches off what the master is not made to carry.

    Dropping constraints can only enlarge the feasible set, so whichever of the two
    relaxations of :data:`MASTER_RELAXATIONS` is used, the objective of the master is a
    valid lower bound of the detailed problem.

    :param model: constructed pyomo model
    :param str relaxation: ``sos2`` or ``pressure``, see :data:`MASTER_RELAXATIONS`
    :return: how many constraints were deactivated
    """
    deactivated = 0
    b_period = model.periods[PERIOD]
    for name in b_period.network_block:
        b_netw = b_period.network_block[name]
        for arc in b_netw.arc_block:
            b_arc = b_netw.arc_block[arc]
            for constraint in MASTER_RELAXATIONS[relaxation]:
                component = b_arc.find_component(constraint)
                if component is not None:
                    component.deactivate()
                    deactivated += 1
    return deactivated


def master_bound(pyhub, model) -> tuple:
    """
    The lower bound the master proved, and the solution it ended on.

    The two are the same only when the master is solved to optimality. When it stops on
    its time limit or its gap, the objective of its incumbent is **above** the optimum
    of the master and is not a bound of anything: the valid lower bound of the full
    problem is the best bound of the solver, which is what is read here
    (``save_results.py:192``).

    :param pyhub: model hub, after the master was solved
    :param model: the pyomo model of the master
    :return: the bound and the objective of the incumbent, either of which can be None
    """
    objective = model.var_npv.value
    try:
        bound = float(pyhub.solution.problem(0).lower_bound)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        bound = None
    if bound is not None and not math.isfinite(bound):
        bound = None
    return bound, objective


def arc_geometry(pyhub) -> dict:
    """
    What the cut generator needs to know about every arc.

    ``r_pipeline`` is the coefficient of the quasi-dynamic pipeline equation, fitted
    from the geometry of the arc (``fixed_size_pipeline.py:121``), and the pressures
    are the ones of the pipeline type.

    :param pyhub: model hub, after reading the data
    :return: per ``(network, node_from, node_to)``, its ``r``, reference pressure,
        pressure band and capacity
    """
    geometry = {}
    for name, netw in pyhub.data.network_data[PERIOD].items():
        coeff_ti = netw.processed_coeff.time_independent
        if "r_pipeline" not in coeff_ti:
            continue
        r_pipeline = coeff_ti["r_pipeline"]
        size_max = coeff_ti["size_max_arcs"]
        for node_from in r_pipeline.index:
            for node_to in r_pipeline.columns:
                r = r_pipeline.at[node_from, node_to]
                if pd.isna(r):
                    continue
                geometry[(name, node_from, node_to)] = {
                    "r": float(r),
                    "pressure_ref": float(netw.pressure_ref),
                    "band": float(netw.pressure_max - netw.pressure_min),
                    "capacity": float(size_max.at[node_from, node_to]),
                }
    return geometry


def pressure_drop(geometry: dict, flow: float) -> float:
    """
    Pressure an arc needs to carry a flow, i.e. the pipeline equation inverted.

    :param dict geometry: entry of :func:`arc_geometry`
    :param float flow: flow in the arc, in MW
    :return: pressure difference, in bar
    """
    p_ref = geometry["pressure_ref"]
    return -p_ref + math.sqrt(p_ref**2 + flow**2 / geometry["r"])


def pressure_slope(geometry: dict, flow: float) -> float:
    """
    Derivative of :func:`pressure_drop` in the flow.

    :param dict geometry: entry of :func:`arc_geometry`
    :param float flow: flow in the arc, in MW
    :return: bar per MW
    """
    p_ref = geometry["pressure_ref"]
    r = geometry["r"]
    return (flow / r) / math.sqrt(p_ref**2 + flow**2 / r)


def net_flows(model) -> dict:
    """
    Flow of every arc of every network, per timestep.

    The two directions of a corridor are two arcs, so the flow read here is the one of
    the direction the arc stands for, and it is zero in the other one.

    :param model: solved pyomo model
    :return: ``{(network, node_from, node_to): {t: flow}}``
    """
    flows = {}
    b_period = model.periods[PERIOD]
    for name in b_period.network_block:
        b_netw = b_period.network_block[name]
        for arc in b_netw.arc_block:
            b_arc = b_netw.arc_block[arc]
            flows[(name,) + tuple(arc)] = {
                t: (b_arc.var_flow[t].value or 0.0) for t in b_period.set_t_full
            }
    return flows


def find_cuts(flows: dict, geometry: dict, limit: int) -> list:
    """
    Paths of the master solution that cannot hold their pressure budget.

    The flows of one timestep are walked as a directed graph, one network at a time,
    and every path of at most :data:`MAX_PATH_LENGTH` arcs whose pressure drops add up
    to more than the band of the pipeline type becomes a cut. Only violated paths are
    kept: a cut that the master already satisfies teaches it nothing.

    :param dict flows: what :func:`net_flows` returned for the master solution
    :param dict geometry: what :func:`arc_geometry` returned
    :param int limit: how many cuts to keep, the most violated ones first
    :return: cuts, each a dict with its terms, its budget and the timestep it came from
    """
    timesteps = sorted({t for series in flows.values() for t in series})
    candidates = []

    for t in timesteps:
        # the arcs that carry something at this timestep, per network
        carried = {}
        for arc, series in flows.items():
            flow = series.get(t, 0.0)
            if arc not in geometry:
                continue
            if flow <= geometry[arc]["capacity"] * 1e-4:
                continue
            carried.setdefault(arc[0], {}).setdefault(arc[1], []).append(
                (arc, flow, pressure_drop(geometry[arc], flow))
            )

        for name, outgoing in carried.items():
            for start in list(outgoing):
                stack = [([], start, 0.0)]
                while stack:
                    walked, node, drop = stack.pop()
                    if len(walked) >= MAX_PATH_LENGTH:
                        continue
                    for arc, flow, arc_drop in outgoing.get(node, []):
                        if any(arc == taken[0] for taken in walked):
                            continue
                        path = walked + [(arc, flow)]
                        total = drop + arc_drop
                        budget = geometry[arc]["band"]
                        if len(path) > 1 and total > budget + 1e-6:
                            candidates.append(
                                {
                                    "terms": [
                                        (
                                            a,
                                            pressure_slope(geometry[a], f),
                                            pressure_drop(geometry[a], f)
                                            - pressure_slope(geometry[a], f) * f,
                                        )
                                        for a, f in path
                                    ],
                                    "budget": budget,
                                    "timestep": t,
                                    "violation": total - budget,
                                    "key": (t, tuple(a for a, _ in path)),
                                }
                            )
                        else:
                            stack.append((path, arc[2], total))

    candidates.sort(key=lambda cut: -cut["violation"])
    return candidates[:limit]


def add_cuts(model, cuts: list) -> int:
    """
    Writes the cuts into a freshly constructed master.

    The master is rebuilt at every iteration, so the whole pool is written again. A
    cut is a linear inequality in the flows and the direction binaries of the arcs of
    one path, and is valid for the detailed model as well.

    :param model: constructed pyomo model of the master
    :param list cuts: what :func:`find_cuts` returned over the iterations
    :return: how many cuts were written
    """
    b_period = model.periods[PERIOD]
    if b_period.find_component("const_pressure_cuts") is None:
        b_period.const_pressure_cuts = pyo.ConstraintList()

    written = 0
    for cut in cuts:
        t = cut["timestep"]
        budget = cut["budget"]
        left, right = 0, budget
        for (name, node_from, node_to), slope, intercept in cut["terms"]:
            b_arc = b_period.network_block[name].arc_block[node_from, node_to]
            left += slope * b_arc.var_flow[t] + intercept
            right += budget * (1 - b_arc.var_direction[t])
        b_period.const_pressure_cuts.add(left <= right)
        written += 1
    return written


def fix_directions(model, flows: dict, geometry: dict, threshold: float) -> tuple:
    """
    Fixes the direction of the arcs the master carries a real flow in.

    An arc the master leaves empty keeps its binary free, so that the subproblem can
    still use it. This is the remedy Ghilardi et al. report against a subproblem the
    master makes infeasible.

    :param model: constructed pyomo model of the subproblem
    :param dict flows: what :func:`net_flows` returned for the master solution
    :param dict geometry: what :func:`arc_geometry` returned
    :param float threshold: share of the capacity of an arc below which its flow is
        not taken as a direction
    :return: how many binaries were fixed to one, and how many to zero
    """
    b_period = model.periods[PERIOD]
    fixed_one, fixed_zero = 0, 0
    for name in b_period.network_block:
        b_netw = b_period.network_block[name]
        for arc in b_netw.arc_block:
            b_arc = b_netw.arc_block[arc]
            key = (name,) + tuple(arc)
            reverse = (name, arc[1], arc[0])
            if key not in geometry:
                continue
            capacity = geometry[key]["capacity"]
            for t in b_period.set_t_full:
                forward = flows.get(key, {}).get(t, 0.0)
                backward = flows.get(reverse, {}).get(t, 0.0)
                if forward > capacity * threshold:
                    b_arc.var_direction[t].fix(1)
                    fixed_one += 1
                elif backward > capacity * threshold:
                    b_arc.var_direction[t].fix(0)
                    fixed_zero += 1
    return fixed_one, fixed_zero


def use_seed(seed: int):
    """
    Makes the solver of this process use a random seed of its own.

    ``Seed`` is not one of the parameters ADOPT hands to gurobi
    (``utilities.py:4-27``), so the builder of the solver is wrapped here instead. Two
    subproblems that differ only by their seed search the tree differently, which is
    the cheapest diversification there is when what is missing is a feasible point.

    :param int seed: the seed
    :return: what the builder was, to put back
    """
    import adopt_net0.modelhub as modelhub

    original = modelhub.get_gurobi_parameters

    def seeded(solveroptions):
        solver = original(solveroptions)
        solver.options["Seed"] = seed
        return solver

    modelhub.get_gurobi_parameters = seeded
    return original


def solve_candidate(task: dict) -> dict:
    """
    Solves one subproblem, in its own process.

    One master solution is not one subproblem but a family of them, along three axes.
    The **threshold** decides how much of the master flow pattern is taken as a
    direction: a low one hands the subproblem a tight corridor, a high one leaves it
    room to reroute. **MIPFocus** decides what the solver spends its time on, and 1 is
    the one that looks for feasibility. The **seed** changes nothing about the problem
    and everything about where the search goes. They are independent runs, so a machine
    with cores to spare should try several at once and keep the best upper bound of the
    round instead of the best of a single guess.

    Everything in and out of this function crosses a process boundary, so it is plain
    data: the model is built here and only numbers come back.

    :param dict task: input folder, horizon, threshold, focus, seed, solver options,
        the flows of the master, the geometry of the arcs and where to write results
    :return: what the candidate was, what it reached, and where its results are
    """
    outcome = {
        "threshold": task["threshold"],
        "focus": task.get("focus"),
        "seed": task.get("seed"),
        "objective": None,
        "error": None,
    }
    try:
        options = dict(task["solver_options"])
        if task.get("focus") is not None:
            options["mipfocus"] = task["focus"]
        if task.get("seed") is not None:
            use_seed(task["seed"])

        pyhub = build(
            Path(task["input_path"]),
            run.LINEPACK,
            task["capacity_factors"],
            task["hours"],
            options,
            set_type=False,
            save_path=Path(task["save_path"]),
        )
        model = model_of(pyhub)
        one, zero = fix_directions(
            model, task["flows"], task["geometry"], task["threshold"]
        )
        outcome["fixed_one"], outcome["fixed_zero"] = one, zero

        pyhub.solve()
        outcome["objective"] = model.var_npv.value
    except Exception as error:  # a worker that dies must not take the round with it
        outcome["error"] = f"{type(error).__name__}: {error}"

    folders = sorted(
        Path(task["save_path"]).glob("*/"), key=lambda p: p.stat().st_mtime
    )
    outcome["folder"] = str(folders[-1]) if folders else None
    return outcome


def run_candidates(tasks: list, workers: int) -> list:
    """
    Runs the subproblems of one round, in parallel when there is more than one.

    :param list tasks: what :func:`solve_candidate` takes, one per threshold
    :param int workers: how many to run at once
    :return: the outcome of each, in the order the tasks were given
    """
    if workers <= 1 or len(tasks) == 1:
        return [solve_candidate(task) for task in tasks]

    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        return list(pool.map(solve_candidate, tasks))


def parse_args(argv=None):
    """
    Reads what to decompose and how far to take it.

    :param list argv: arguments to read, the command line when left out
    :return: the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Solves the linepack case by a bilevel decomposition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layout", metavar="FILE", help="layout written into the case")
    parser.add_argument(
        "--input",
        metavar="FOLDER",
        help="input data folder, so that a run can work on its own copy of the case "
        "instead of the one of the example (default: the one of the example)",
    )
    parser.add_argument(
        "--copy-case",
        action="store_true",
        help="copy the case to a folder of this run before touching it, and remove "
        "the copy at the end. Two runs that share one case rewrite each other's "
        "topology: the one that finishes first puts the case back while the other is "
        "still reading it",
    )
    parser.add_argument(
        "--hours", type=int, default=run.DEFAULT_HOURS, help="length of the horizon"
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="how many master-subproblem rounds"
    )
    parser.add_argument(
        "--master",
        choices=sorted(MASTER_RELAXATIONS),
        default="sos2",
        help="what the master drops: the adjacency condition of the interpolation "
        "alone, or the pipeline equation and the pressure coupling as well",
    )
    parser.add_argument(
        "--master-timelim",
        type=float,
        default=None,
        help="time limit of the master in hours, when it should differ from the one "
        "of the subproblems. The master only has to produce a flow pattern and a "
        "bound, so it is usually given less",
    )
    parser.add_argument(
        "--candidates",
        default=",".join(str(threshold) for threshold in DEFAULT_CANDIDATES),
        help="thresholds to hand the subproblems of a round, as a share of the "
        "capacity of an arc. Each one is a subproblem of its own and they run at once",
    )
    parser.add_argument(
        "--focus",
        default="1",
        help="MIPFocus of the subproblems, one value or several. 1 looks for "
        "feasibility, 2 for optimality, 3 for the bound. Every value is combined with "
        "every threshold and every seed",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="random seeds of the subproblems. Two runs that differ only by the seed "
        "search the tree differently, which is the cheapest way to look in more than "
        "one place at once when what is missing is a feasible point",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=len(DEFAULT_CANDIDATES),
        help="how many subproblems run at the same time. More candidates than workers "
        "is allowed, they queue",
    )
    parser.add_argument(
        "--sub-threads",
        type=int,
        default=4,
        help="threads each subproblem gets. Threads times workers should stay under "
        "the cores of the machine, and the memory bounds the workers before the cores "
        "do",
    )
    for option, (option_type, description) in run.SOLVER_OPTIONS.items():
        parser.add_argument(
            f"--{option}", type=option_type, default=None, help=description
        )
    return parser.parse_args(argv)


def main(argv=None):
    """
    Runs the decomposition and reports the bounds it closed.

    :param list argv: arguments to read, the command line when left out
    """
    args = parse_args(argv)
    solver_options = {option: getattr(args, option) for option in run.SOLVER_OPTIONS}

    case_dir = Path(__file__).parent
    input_path = Path(args.input) if args.input else case_dir / "input_data"
    capacity_factors = pd.read_csv(
        case_dir / "res_capacity_factors.csv", sep=SEP, index_col=0
    )
    thresholds = [float(v) for v in args.candidates.split(",") if v.strip()]
    focuses = [int(v) for v in args.focus.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    results_path = case_dir / "userData" / RESULTS_FOLDER

    backup = None
    copied = None
    cuts = []
    best_upper, best_lower = float("inf"), float("-inf")
    try:
        if args.copy_case:
            copied = results_path / f"case_{os.getpid()}"
            if copied.exists():
                shutil.rmtree(copied)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(input_path, copied)
            input_path = copied
            print(f"\nworking on a copy of the case in {input_path}")

        if args.layout:
            chosen = layout.read(args.layout, input_path)
            backup = run.apply_layout(input_path, chosen)
            print(f"\nlayout of {args.layout}\n{layout.describe(chosen)}")

        master_options = dict(solver_options)
        if args.master_timelim is not None:
            master_options["timelim"] = args.master_timelim

        for iteration in range(1, args.iterations + 1):
            print(f"\n=== iteration {iteration}: master ===")
            master = build(
                input_path,
                run.LINEPACK,
                capacity_factors,
                args.hours,
                master_options,
                save_path=results_path / f"it{iteration}_master",
                write_results=False,
            )
            model = model_of(master)
            relaxed = relax_pressure(model, args.master)
            print(f"master relaxation {args.master}: {relaxed} constraints dropped")
            geometry = arc_geometry(master)
            print(f"wrote {add_cuts(model, cuts)} cuts of earlier iterations")

            master.solve()
            lower, master_objective = master_bound(master, model)
            if lower is None:
                print("the master proved no bound: the lower bound is left as it was")
                lower = best_lower
            best_lower = max(best_lower, lower)
            master_flows = net_flows(model)

            # a path that is already cut is cut again at a different point of the
            # curve, which is a tighter tangent, but the same path at the same
            # timestep is only worth one cut per iteration
            known = {cut["key"] for cut in cuts}
            fresh = [
                cut
                for cut in find_cuts(master_flows, geometry, MAX_CUTS_PER_ITERATION)
                if cut["key"] not in known
            ]
            reached = (
                f"{master_objective:.6g}" if master_objective is not None else "none"
            )
            print(
                f"master bound {lower:.6g}, solution {reached}, "
                f"{len(fresh)} paths over their pressure budget"
            )

            print(
                f"\n=== iteration {iteration}: {len(thresholds)} subproblems on "
                f"{min(args.workers, len(thresholds))} workers ==="
            )
            sub_options = dict(solver_options)
            sub_options["threads"] = args.sub_threads
            tasks = [
                {
                    "input_path": str(input_path),
                    "hours": args.hours,
                    "capacity_factors": capacity_factors,
                    "solver_options": sub_options,
                    "flows": master_flows,
                    "geometry": geometry,
                    "threshold": threshold,
                    "focus": focus,
                    "seed": seed,
                    "save_path": str(
                        results_path / f"it{iteration}_t{threshold}_f{focus}_s{seed}"
                    ),
                }
                for threshold, focus, seed in product(thresholds, focuses, seeds)
            ]
            outcomes = run_candidates(tasks, args.workers)

            print(
                f"{'threshold':>10}{'focus':>7}{'seed':>6}{'fixed':>8}"
                f"{'objective':>18}  where"
            )
            for outcome in outcomes:
                fixed = outcome.get("fixed_one", 0) + outcome.get("fixed_zero", 0)
                if outcome["objective"] is None:
                    reached = outcome["error"] or "no solution"
                else:
                    reached = f"{outcome['objective']:.6g}"
                    best_upper = min(best_upper, outcome["objective"])
                folder = Path(outcome["folder"]).name if outcome["folder"] else "-"
                print(
                    f"{outcome['threshold']:>10}{outcome['focus']:>7}"
                    f"{outcome['seed']:>6}{fixed:>8}{reached:>18}  {folder}"
                )

            gap = None
            if math.isfinite(best_upper):
                gap = (best_upper - best_lower) / abs(best_upper)
            print(
                f"\nbounds after {iteration}: lower {best_lower:.6g}, "
                f"upper {best_upper:.6g}"
                + (f", gap {gap:.3%}" if gap is not None else "")
            )

            if not fresh:
                print("the master respects every pressure budget: nothing left to cut")
                break
            cuts += fresh
    finally:
        if copied is not None:
            # the copy is the whole isolation, so there is nothing to put back
            shutil.rmtree(copied, ignore_errors=True)
        else:
            if backup is not None:
                layout.restore(input_path, backup)
            run.set_network_type(input_path, run.LINEPACK)


if __name__ == "__main__":
    main()
