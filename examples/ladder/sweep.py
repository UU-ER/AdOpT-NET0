"""
Runs the ladder across the axes that decide how hard a rung is, and tabulates it.

The question the sweep answers is where the complexity of the linepack model comes
from. Four things can be varied independently, and the point of the sweep is that they
are varied one at a time::

- **the rung**, i.e. how many nodes and how many corridors the case has. What matters
  for the pipeline equation is not the node count but the number of independent cycles
  of the corridor graph, ``corridors - nodes + 1``, because the pressure of a node is
  shared by every arc that touches it: on a tree the flows fix the pressures, and every
  cycle adds one more equation than there are unknowns
- **what is decided**, i.e. ``--design none`` for an operation with the corridors
  already given, against ``--design network`` where which corridor to build is the
  decision. This separates the install binaries from the direction binaries
- **how many pipeline types** are offered on each corridor, one or three. A type is a
  network of its own with its own pressure field, so three types is three times the
  pressure machinery and three times the install binaries per corridor
- **the horizon**, since the direction binaries are per timestep and the linepack
  balance couples the whole horizon cyclically

Every job solves both network models in one call, the reference first, so the linepack
run is warm started from it and the two objectives are comparable. What the sweep
reports per job and per model is the size of the model, the time the first incumbent
took, the final gap and whether optimality was proved.

Each job works on its own copy of the case, because ``run.py`` writes the layout and the
design into the case before solving and puts them back afterwards: two jobs sharing one
rung would rewrite each other. The copy also carries its own result folder, so the
``Summary.xlsx`` of a rung is not appended to by several jobs at once.

Usage::

    python sweep.py                                  the whole sweep, 8 at a time
    python sweep.py --workers 12 --threads 4         more jobs, fewer threads each
    python sweep.py --timelim 0.5 --hours 24,96      half an hour a model, two horizons
    python sweep.py --only 3L3S                      one rung
    python sweep.py --dry-run                        list the jobs and stop

The results land in ``sweep/<stamp>/``: one folder per job with its case copy, its
stdout and its solver logs, plus ``results.csv`` and ``summary.txt`` at the top.

.. note::
    ``--precise`` is on by default, so the reference forbids a corridor carrying both
    ways in the same hour, which the linepack model forbids anyway. Without it the
    reference is cheaper than the physics allows and the cost difference the sweep
    reports is not the worth of the linepack but the worth of an illegal reference.

.. note::
    No import cap is set. A rung with one large cluster cannot serve Rotterdam without
    imports, so capping them at zero makes those rungs infeasible, and a cap that
    applies to some rungs and not others is not one axis but two.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import layout

CASE_DIR = Path(__file__).parent
REPO_ROOT = CASE_DIR.parent.parent

ONE_TYPE = "H2Pipeline_medium"
ALL_TYPES = ["H2Pipeline_large", "H2Pipeline_medium", "H2Pipeline_small"]

#: Rungs in the order they are meant to be climbed, i.e. by how many independent cycles
#: the corridor graph of one pipeline type has: 0, 1, 2, 3, 7.
RUNG_ORDER = ["1L1S", "1L2S", "2L2S", "1L3S", "3L3S"]

#: Horizons solved for every rung, in hours.
DEFAULT_HOURS = [24, 48, 96]

#: Corridor counts of the arc curve, i.e. the same rung with fewer candidates offered.
DEFAULT_ARC_COUNTS = [3, 5, 6, 7, 8, 10]

#: Rung the arc curve is measured on, and the horizon it uses.
ARC_RUNG = "3L3S"
ARC_HOURS = 24

#: What each network type is called in the tables, in the order they are solved.
MODELS = ["reference", "linepack"]


def cycles_of(arcs) -> int:
    """
    Independent cycles of a corridor graph, i.e. ``corridors - nodes + 1``.

    This is the count that decides whether the pressures are determined by the flows.
    Every arc ties the pressures of its two end nodes twice, once through the pipeline
    equation and once through the linepack, against one pressure per node: on a tree
    the system is exactly determined and a flow pattern always has a pressure field,
    and every cycle adds one equation more than there are unknowns.

    :param arcs: corridors, each a pair of node names
    :return: the cycle count, never below zero
    """
    arcs = {tuple(sorted(arc)) for arc in arcs}
    nodes = {node for arc in arcs for node in arc}
    return max(0, len(arcs) - len(nodes) + 1)


def cycles_of_job(job: dict, input_path: Path) -> int:
    """
    Cycles the corridor graph of one pipeline type has, as the job offers it.

    The pressure of a node belongs to the network block, i.e. to the pipeline type, so
    a node reached by two types carries one pressure per type and the graphs do not
    share their cycles. The count reported is therefore the worst of the types, not the
    one of their union, and it follows the corridors the job actually offers rather
    than the ones the rung carries.

    :param dict job: the job, see :func:`build_jobs`
    :param Path input_path: input data folder of the rung
    :return: the cycle count of the type that has the most
    """
    case = layout.case_layout(input_path)
    chosen = candidate_corridors(input_path)
    if job["corridors"] is not None:
        chosen = chosen[: job["corridors"]]

    counts = [cycles_of(chosen)] if job["types"] else [0]
    counts += [cycles_of(arcs) for arcs in case["existing"].values() if arcs]
    return max(counts)


def split_corridors(input_path: Path) -> tuple:
    """
    The candidate corridors of the rung, split into the ones that connect it and
    the ones that close a cycle.

    The order is what the arc curve varies, so it decides what a run with ``k``
    corridors is. Taking them alphabetically would offer a star on whichever node comes
    first and leave the others with no corridor at all, and a small cluster that is cut
    off cannot import, so the job would be infeasible rather than easy. The corridors
    are therefore ordered as Kruskal would take them, shortest first and seeded with
    the backbone: the first few join the rung into one system and every one after that
    adds exactly one cycle.

    :param Path input_path: input data folder of the rung
    :return: the corridors that join the rung, then the ones that add a cycle
    """
    case = layout.case_layout(input_path)
    lengths = layout.known_arcs(input_path)

    arcs = set()
    for arcs_of_type in case["candidates"].values():
        arcs.update(tuple(sorted(arc)) for arc in arcs_of_type)

    def length_of(arc):
        return lengths.get(arc, lengths.get((arc[1], arc[0]), float("inf")))

    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(node_a, node_b) -> bool:
        root_a, root_b = find(node_a), find(node_b)
        if root_a == root_b:
            return False
        parent[root_b] = root_a
        return True

    # the backbone is already there, so the corridors that join what it joins are not
    # what connects the rung and belong with the ones that add a cycle
    for arcs_of_type in case["existing"].values():
        for arc in arcs_of_type:
            union(arc[0], arc[1])

    tree, rest = [], []
    for arc in sorted(arcs, key=length_of):
        (tree if union(arc[0], arc[1]) else rest).append(arc)
    return tree, rest


def candidate_corridors(input_path: Path) -> list:
    """
    Every corridor the rung offers as a candidate, connected ones first.

    :param Path input_path: input data folder of the rung
    :return: corridors, each a sorted pair of node names
    """
    tree, rest = split_corridors(input_path)
    return tree + rest


def connecting_corridors(input_path: Path) -> int:
    """
    How many candidates the rung needs before it is one system.

    A job offered fewer than this leaves a cluster with no corridor at all, and a small
    cluster cannot import, so such a job is infeasible rather than small.

    :param Path input_path: input data folder of the rung
    :return: the number of corridors that join the rung without closing a cycle
    """
    tree, _ = split_corridors(input_path)
    return len(tree)


def job_layout(input_path: Path, types: list, corridors: int) -> dict:
    """
    A layout offering the given types on the first ``corridors`` candidates.

    What the rung carries as existing is kept, so restricting the candidates never
    takes the backbone away.

    :param Path input_path: input data folder of the rung
    :param list types: pipeline types offered as candidates
    :param int corridors: how many corridors to offer, all of them when ``None``
    :return: corridors per type, under ``existing`` and ``candidates``
    """
    case = layout.case_layout(input_path)
    chosen = candidate_corridors(input_path)
    if corridors is not None:
        chosen = chosen[:corridors]
    return {
        "existing": {
            name: [list(arc) for arc in arcs]
            for name, arcs in case["existing"].items()
            if arcs
        },
        "candidates": {name: [list(arc) for arc in chosen] for name in types},
    }


def build_jobs(hours: list, arc_counts: list, only: list) -> list:
    """
    The jobs of the sweep, i.e. one call of ``run.py`` each.

    The main grid crosses every rung with every horizon and with the three ways of
    posing the problem. The arc curve then holds the rung and the horizon still and
    varies only how many corridors are on offer, which is the one axis the main grid
    cannot separate from the node count.

    :param list hours: horizons of the main grid
    :param list arc_counts: corridor counts of the arc curve
    :param list only: rungs to keep, all of them when empty
    :return: the jobs, each a dict of what the call needs
    """
    jobs = []
    for rung in RUNG_ORDER:
        if only and rung not in only:
            continue
        for n_hours in hours:
            jobs.append(
                {
                    "rung": rung,
                    "hours": n_hours,
                    "design": "none",
                    "types": [ONE_TYPE],
                    "corridors": None,
                    "axis": "grid",
                }
            )
            jobs.append(
                {
                    "rung": rung,
                    "hours": n_hours,
                    "design": "network",
                    "types": [ONE_TYPE],
                    "corridors": None,
                    "axis": "grid",
                }
            )
            jobs.append(
                {
                    "rung": rung,
                    "hours": n_hours,
                    "design": "network",
                    "types": list(ALL_TYPES),
                    "corridors": None,
                    "axis": "grid",
                }
            )

    if not only or ARC_RUNG in only:
        arc_case = CASE_DIR / ARC_RUNG / "input_data"
        available = len(candidate_corridors(arc_case))
        connects = connecting_corridors(arc_case)
        for count in arc_counts:
            # fewer than that leaves a cluster with no corridor, and the whole grid
            # already carries the point where every corridor is offered
            if count < connects or count >= available:
                continue
            for mode in ["none", "network"]:
                jobs.append(
                    {
                        "rung": ARC_RUNG,
                        "hours": ARC_HOURS,
                        "design": mode,
                        "types": [ONE_TYPE],
                        "corridors": count,
                        "axis": "arcs",
                    }
                )

    for job in jobs:
        job["id"] = (
            f"{job['rung']}_h{job['hours']}_{job['design']}"
            f"_{len(job['types'])}type"
            + (f"_{job['corridors']}arc" if job["corridors"] is not None else "")
        )
    return jobs


def prepare(job: dict, out_dir: Path) -> dict:
    """
    Gives a job its own copy of the case and its own result folder.

    The save paths of ``ConfigModel.json`` are absolute and point at the ``userData`` of
    the rung, so a copy that keeps them would write its results next to every other
    job's and append to the same summary spreadsheet.

    :param dict job: the job, see :func:`build_jobs`
    :param Path out_dir: folder of the sweep
    :return: the job, with the paths it was given
    """
    job_dir = out_dir / job["id"]
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True)

    source = CASE_DIR / job["rung"] / "input_data"
    case = job_dir / "input_data"
    shutil.copytree(source, case)

    results = job_dir / "results"
    results.mkdir()
    config_file = case / "ConfigModel.json"
    config = json.loads(config_file.read_text())
    config["reporting"]["save_path"]["value"] = str(results)
    config["reporting"]["save_summary_path"]["value"] = str(results)
    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # the case default already offers every type on every corridor, so a layout is only
    # written when the job asks for less than that
    layout_file = None
    if job["corridors"] is not None or job["types"] != ALL_TYPES:
        layout_file = job_dir / "layout.json"
        layout.dump(job_layout(source, job["types"], job["corridors"]), layout_file)

    job["dir"] = job_dir
    job["case"] = case
    job["results"] = results
    job["layout_file"] = layout_file
    return job


def command(
    job: dict, timelim: float, mipgap: float, threads: int, precise: bool
) -> list:
    """
    The ``run.py`` call of a job.

    :param dict job: the job, after :func:`prepare`
    :param float timelim: time limit of each network type, in hours
    :param float mipgap: relative gap every job stops at
    :param int threads: threads the solver is given
    :param bool precise: whether the reference forbids both directions at once
    :return: the argument list
    """
    argv = [
        sys.executable,
        str(CASE_DIR / "run.py"),
        "--rung",
        job["rung"],
        "--input",
        str(job["case"]),
        "--hours",
        str(job["hours"]),
        "--design",
        job["design"],
        "--case",
        "both",
        "--timelim",
        str(timelim),
        "--mipgap",
        str(mipgap),
        "--threads",
        str(threads),
        "--no-viewer",
    ]
    if precise:
        argv.append("--precise")
    if job["layout_file"] is not None:
        argv += ["--layout", str(job["layout_file"])]
    return argv


def parse_log(path: Path) -> dict:
    """
    What one gurobi log says about the model and about the search.

    :param Path path: the solver log
    :return: size of the model, the root relaxation, the first incumbent and the end
    """
    text = path.read_text(errors="replace")
    found = {}

    match = re.search(r"Optimize a model with (\d+) rows, (\d+) columns", text)
    if match:
        found["rows"] = int(match.group(1))
        found["cols"] = int(match.group(2))

    match = re.search(r"Model has (\d+) SOS constraint", text)
    found["sos"] = int(match.group(1)) if match else 0

    match = re.search(
        r"Variable types: \d+ continuous, \d+ integer \((\d+) binary\)", text
    )
    if match:
        found["binaries"] = int(match.group(1))

    match = re.search(
        r"Root relaxation: objective ([-\d.e+]+), \d+ iterations, ([\d.]+) seconds",
        text,
    )
    if match:
        found["root_obj"] = float(match.group(1))
        found["root_s"] = float(match.group(2))

    # the first line that carries an incumbent, i.e. the first heuristic or improved
    # solution. A run that never finds one leaves this empty, which is itself a result
    for line in text.splitlines():
        if re.match(r"^[H*]\s*\d+", line):
            at = re.search(r"([\d.]+)s\s*$", line)
            if at:
                found["first_s"] = float(at.group(1))
                break

    found["mipstart"] = int(
        "User MIP start produced solution" in text or "Loaded user MIP start" in text
    )

    match = re.search(
        r"Explored (\d+) nodes \((\d+) simplex iterations\) in ([\d.]+) seconds", text
    )
    if match:
        found["nodes"] = int(match.group(1))
        found["simplex"] = int(match.group(2))
        found["solve_s"] = float(match.group(3))

    match = re.search(
        r"Best objective ([-\d.e+]+), best bound ([-\d.e+]+), gap ([\d.]+)%", text
    )
    if match:
        found["obj"] = float(match.group(1))
        found["bound"] = float(match.group(2))
        found["gap"] = float(match.group(3)) / 100
    else:
        match = re.search(r"Optimal objective\s+([-\d.e+]+)", text)
        if match:
            found["obj"] = float(match.group(1))
            found["bound"] = found["obj"]
            found["gap"] = 0.0

    if "Time limit reached" in text:
        found["status"] = "timelimit"
    elif "Optimal solution found" in text or "Optimal objective" in text:
        found["status"] = "optimal"
    elif "infeasible" in text.lower():
        found["status"] = "infeasible"
    else:
        found["status"] = "unknown"

    if "obj" not in found:
        found["status"] = "no solution"
    return found


def solve_job(
    job: dict,
    timelim: float,
    mipgap: float,
    threads: int,
    precise: bool,
    keep_cases: bool,
) -> dict:
    """
    Runs one job and reads back what its logs say.

    A job that fails is reported rather than raised: a rung that does not solve is a
    result of the sweep and not a reason to lose the jobs that did.

    :param dict job: the job, after :func:`prepare`
    :param float timelim: time limit of each network type, in hours
    :param int threads: threads the solver is given
    :param bool precise: whether the reference forbids both directions at once
    :param bool keep_cases: whether the copy of the case is kept after the run
    :return: one row per network type solved
    """
    argv = command(job, timelim, mipgap, threads, precise)
    started = time.time()
    with open(job["dir"] / "stdout.txt", "w", encoding="utf-8") as handle:
        handle.write(" ".join(argv) + "\n\n")
        handle.flush()
        finished = subprocess.run(
            argv,
            cwd=str(CASE_DIR),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=with_pythonpath(),
        )
    wall = time.time() - started

    # the copy is several MB of case files per job and has done its work once the run
    # is over, so only what the run produced is kept
    if not keep_cases:
        shutil.rmtree(job["case"], ignore_errors=True)

    # the result folders of ADOPT are named after the second the run started in, so
    # sorting them puts the reference first and the linepack second, as they are solved
    logs = sorted(job["results"].glob("*/solver_log.txt"))
    rows = []
    for index, log in enumerate(logs[: len(MODELS)]):
        row = {
            "id": job["id"],
            "axis": job["axis"],
            "rung": job["rung"],
            "cycles": job["cycles"],
            "hours": job["hours"],
            "design": job["design"],
            "types": len(job["types"]),
            "corridors": job["corridors"] if job["corridors"] is not None else "all",
            "model": MODELS[index],
            "wall_s": round(wall, 1),
            "exit": finished.returncode,
        }
        row.update(parse_log(log))
        rows.append(row)

    if not rows:
        rows = [
            {
                "id": job["id"],
                "axis": job["axis"],
                "rung": job["rung"],
                "cycles": job["cycles"],
                "hours": job["hours"],
                "design": job["design"],
                "types": len(job["types"]),
                "corridors": (
                    job["corridors"] if job["corridors"] is not None else "all"
                ),
                "model": "-",
                "wall_s": round(wall, 1),
                "exit": finished.returncode,
                "status": "no log",
            }
        ]
    return rows


def with_pythonpath() -> dict:
    """
    The environment a job runs in, with the package importable.

    :return: a copy of the environment of the sweep
    """
    import os

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


COLUMNS = [
    "id",
    "axis",
    "rung",
    "cycles",
    "hours",
    "design",
    "types",
    "corridors",
    "model",
    "status",
    "obj",
    "bound",
    "gap",
    "first_s",
    "solve_s",
    "wall_s",
    "nodes",
    "simplex",
    "rows",
    "cols",
    "binaries",
    "sos",
    "root_obj",
    "root_s",
    "mipstart",
    "exit",
]


def write_csv(rows: list, path: Path):
    """
    Writes every row of the sweep, one per job and network type.

    :param list rows: the rows collected
    :param Path path: file to write
    """
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list) -> str:
    """
    The sweep as a table, the hardest jobs last.

    :param list rows: the rows collected
    :return: the lines, ready to print
    """
    lines = []
    header = (
        f"{'rung':7}{'cyc':>4}{'h':>5}{'design':>9}{'typ':>4}{'arcs':>6}"
        f"{'model':>11}{'status':>12}{'objective':>14}{'gap':>9}"
        f"{'first':>8}{'solve':>8}{'nodes':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    order = {name: index for index, name in enumerate(RUNG_ORDER)}
    for row in sorted(
        rows,
        key=lambda r: (
            order.get(r["rung"], 99),
            r["hours"],
            r["design"],
            r["types"],
            str(r["corridors"]),
            r["model"],
        ),
    ):
        gap = row.get("gap")
        obj = row.get("obj")
        first = row.get("first_s")
        lines.append(
            f"{row['rung']:7}{row.get('cycles', 0):>4}{row['hours']:>5}"
            f"{row['design']:>9}{row['types']:>4}{str(row['corridors']):>6}"
            f"{row['model']:>11}{row.get('status', '-'):>12}"
            + (f"{obj:>14.6g}" if obj is not None else f"{'-':>14}")
            + (f"{gap:>9.3%}" if gap is not None else f"{'-':>9}")
            + (f"{first:>8.0f}" if first is not None else f"{'-':>8}")
            + (
                f"{row['solve_s']:>8.0f}"
                if row.get("solve_s") is not None
                else f"{'-':>8}"
            )
            + (f"{row['nodes']:>9}" if row.get("nodes") is not None else f"{'-':>9}")
        )
    return "\n".join(lines)


def parse_args(argv=None):
    """
    Reads how much of the sweep to run and how much machine to give it.

    :param list argv: arguments to read, the command line when left out
    :return: the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Runs the ladder across the axes that decide how hard a rung is.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        metavar="FOLDER",
        default=None,
        help="where the sweep writes, a stamped folder under sweep/ by default",
    )
    parser.add_argument(
        "--hours",
        default=",".join(str(value) for value in DEFAULT_HOURS),
        help="horizons of the main grid",
    )
    parser.add_argument(
        "--arc-counts",
        default=",".join(str(value) for value in DEFAULT_ARC_COUNTS),
        help=f"corridor counts of the arc curve, measured on {ARC_RUNG}",
    )
    parser.add_argument(
        "--only",
        default="",
        help="rungs to run, comma separated. All of them when left out",
    )
    parser.add_argument(
        "--timelim",
        type=float,
        default=0.25,
        help="time limit per network type, in hours",
    )
    parser.add_argument(
        "--mipgap",
        type=float,
        default=0.01,
        help="relative gap every job stops at, so that the times compare",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="how many jobs run at the same time"
    )
    parser.add_argument(
        "--threads", type=int, default=6, help="threads each job gives the solver"
    )
    parser.add_argument(
        "--no-precise",
        dest="precise",
        action="store_false",
        help="let the reference carry a corridor both ways in the same hour, which the "
        "linepack model cannot do. The cost difference is then not comparable",
    )
    parser.add_argument(
        "--keep-cases",
        action="store_true",
        help="keep the copy of the case each job worked on, which is several MB a job "
        "and only worth it when a job has to be reproduced by hand",
    )
    parser.add_argument("--dry-run", action="store_true", help="list the jobs and stop")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Prepares every job, runs them and writes the table.

    :param list argv: arguments to read, the command line when left out
    """
    args = parse_args(argv)
    hours = [int(value) for value in args.hours.split(",") if value.strip()]
    arc_counts = [int(value) for value in args.arc_counts.split(",") if value.strip()]
    only = [value.strip() for value in args.only.split(",") if value.strip()]

    jobs = build_jobs(hours, arc_counts, only)
    for job in jobs:
        job["cycles"] = cycles_of_job(job, CASE_DIR / job["rung"] / "input_data")

    print(f"{len(jobs)} jobs, {args.workers} at a time, {args.threads} threads each")
    print(f"each solves both network models, {args.timelim} h a model at most\n")
    for job in jobs:
        print(f"  {job['id']:34}cycles {job['cycles']}")
    if args.dry_run:
        return

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = Path(args.out) if args.out else CASE_DIR / "sweep" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting to {out_dir}\n")

    for job in jobs:
        prepare(job, out_dir)

    rows = []
    done = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                solve_job,
                job,
                args.timelim,
                args.mipgap,
                args.threads,
                args.precise,
                args.keep_cases,
            )
            for job in jobs
        ]
        for future in futures:
            rows += future.result()
            done += 1
            elapsed = (time.time() - started) / 60
            print(f"  {done}/{len(jobs)} done, {elapsed:.0f} min elapsed", flush=True)
            write_csv(rows, out_dir / "results.csv")

    write_csv(rows, out_dir / "results.csv")
    table = summarize(rows)
    (out_dir / "summary.txt").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\n{out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
