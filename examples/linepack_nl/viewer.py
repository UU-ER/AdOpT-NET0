"""
Turns the results of a run into a standalone HTML map.

Reads ``optimization_results.h5`` and the node locations of the case, and writes a
single self-contained page showing what the optimizer built and how it operates it:
the network on a map, the flow and the linepack of every pipeline over time, and the
technologies of every node.

Every run of the case is embedded in the same page and picked from the menu at the
top, so the page only has to be written again when a new run has been solved. A run
of the reference network writes no linepack, which is how the two cases are told
apart in the menu.

The page has no external dependencies, so it can be opened from disk.

Run with::

    python examples/linepack_nl/viewer.py [result_folder] [-o results.html] [-n 12]

Without a result folder every result folder of the case is read, the most recent
ones first.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SEP = ";"
CASE_DIR = Path(__file__).parent


def _text(dataset) -> str:
    """
    Reads one h5 dataset holding a string.

    :param dataset: h5 dataset
    :return: its content
    """
    value = dataset[()]
    return value.decode() if isinstance(value, bytes) else str(value)


def _scalar(dataset) -> float:
    """
    Reads one h5 dataset holding a number.

    Some of them are written as an array of one element rather than as a scalar, so
    both shapes are accepted.

    :param dataset: h5 dataset
    :return: its value
    """
    return float(np.asarray(dataset).flatten()[0])


def _series(group, key) -> list:
    """
    Reads one h5 dataset holding a time series, rounded for the page.

    :param group: h5 group
    :param str key: name of the dataset
    :return: the series, or an empty list if it is not there
    """
    if key not in group:
        return []
    return [round(float(v), 3) for v in np.asarray(group[key])]


def all_results(case_dir: Path) -> list:
    """
    Every result folder of the case that holds results, newest first.

    :param Path case_dir: folder of the case
    :return: the folders holding optimization_results.h5
    """
    results_path = case_dir / "userData"
    if not results_path.is_dir():
        raise FileNotFoundError(f"No results under {results_path}")
    folders = [
        p
        for p in sorted(results_path.iterdir(), reverse=True)
        if p.is_dir() and (p / "optimization_results.h5").exists()
    ]
    if not folders:
        raise FileNotFoundError(f"No optimization_results.h5 under {results_path}")
    return folders


def read_results(result_folder: Path, case_dir: Path) -> dict:
    """
    Collects everything the page shows from one result folder.

    :param Path result_folder: folder holding optimization_results.h5
    :param Path case_dir: folder of the case, for the node locations
    :return: the data of the page
    """
    locations = pd.read_csv(
        case_dir / "input_data" / "NodeLocations.csv", sep=SEP, index_col=0
    )

    with h5py.File(result_folder / "optimization_results.h5", "r") as f:
        summary = {
            k: float(np.asarray(v).flatten()[0])
            for k, v in f["summary"].items()
            if np.asarray(v).dtype.kind in "fiu"
        }
        periods = [
            p.decode() if isinstance(p, bytes) else str(p)
            for p in f["topology"]["periods"][()]
        ]
        period = periods[0]
        nodes = [
            n.decode() if isinstance(n, bytes) else str(n)
            for n in f["topology"]["nodes"][()]
        ]

        arcs = {}
        design = f["design"]["networks"][period]
        operation = f["operation"]["networks"][period]
        for network in design:
            for key in design[network]:
                arc_design = design[network][key]
                size = _scalar(arc_design["size"])
                node_from = _text(arc_design["fromNode"])
                node_to = _text(arc_design["toNode"])
                arc_operation = (
                    operation[network][key] if key in operation[network] else {}
                )
                arcs.setdefault(
                    (network, tuple(sorted((node_from, node_to)))),
                    {
                        "network": network,
                        "nodes": sorted((node_from, node_to)),
                        "size": size,
                        "capex": _scalar(arc_design["capex"]),
                        "flow": {},
                        "linepack": [],
                    },
                )
                entry = arcs[(network, tuple(sorted((node_from, node_to))))]
                entry["size"] = max(entry["size"], size)
                # the linepack network writes the flow at both ends of an arc, the
                # reference one writes the single flow of the parent class
                flow = _series(arc_operation, "flow_in") or _series(
                    arc_operation, "flow"
                )
                entry["flow"][f"{node_from}->{node_to}"] = flow
                if not entry["linepack"]:
                    entry["linepack"] = _series(arc_operation, "linepack")

        # Compressors are written into the same group as the technologies of a node,
        # but they are not a decision of the same kind, so they are kept apart
        technologies = {}
        compressors = {}
        node_design = f["design"]["nodes"][period]
        for node in node_design:
            entries, compressor_entries = [], []
            for name in node_design[node]:
                group = node_design[node][name]
                if "size" not in group:
                    continue
                size = _scalar(group["size"])
                if size <= 1e-6:
                    continue
                entry = {
                    "name": name,
                    "size": size,
                    "capex": _scalar(group["capex"]) if "capex" in group else 0.0,
                }
                target = compressor_entries if "_Compressor_" in name else entries
                target.append(entry)
            technologies[node] = sorted(entries, key=lambda e: -e["size"])
            compressors[node] = sorted(compressor_entries, key=lambda e: -e["size"])

        balances = {}
        for node in f["operation"]["energy_balance"][period]:
            group = f["operation"]["energy_balance"][period][node]
            if "hydrogen" not in group:
                continue
            car = group["hydrogen"]
            balances[node] = {
                key: _series(car, key)
                for key in (
                    "demand",
                    "technology_outputs",
                    "technology_inputs",
                    "network_inflow",
                    "network_outflow",
                    "import",
                    "export",
                )
                if key in car
            }

    horizon = max(
        (len(a["linepack"]) for a in arcs.values() if a["linepack"]), default=0
    )
    if horizon == 0:
        horizon = max(
            (len(s) for arc in arcs.values() for s in arc["flow"].values()), default=0
        )

    # A run of the reference network writes no linepack, which is how the two cases
    # are told apart on the page
    with_linepack = any(a["linepack"] for a in arcs.values())

    return {
        "run": result_folder.name,
        "kind": "linepack" if with_linepack else "fixed size",
        "period": period,
        "summary": summary,
        "horizon": horizon,
        "nodes": [
            {
                "name": name,
                "lon": float(locations.at[name, "lon"]),
                "lat": float(locations.at[name, "lat"]),
                "technologies": technologies.get(name, []),
                "compressors": compressors.get(name, []),
                "balance": balances.get(name, {}),
            }
            for name in nodes
        ],
        "arcs": sorted(arcs.values(), key=lambda a: -a["size"]),
    }


def write_html(runs: list, output: Path):
    """
    Writes the page, with the data of every run embedded in it.

    :param list runs: the data of each run, newest first
    :param Path output: file to write
    """
    template = (CASE_DIR / "viewer_template.html").read_text(encoding="utf-8")
    data = json.dumps(runs, separators=(",", ":"))
    output.write_text(template.replace("__RESULTS_JSON__", data), encoding="utf-8")


def main():
    """
    Reads the result folders of the case and writes the page next to them.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_folder",
        nargs="?",
        help="a single folder to read; every result folder of the case by default",
    )
    parser.add_argument("-o", "--output", default=str(CASE_DIR / "results.html"))
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=12,
        help="how many of the most recent runs to embed (default 12)",
    )
    args = parser.parse_args()

    if args.result_folder:
        folders = [Path(args.result_folder)]
    else:
        folders = all_results(CASE_DIR)[: max(args.runs, 1)]

    runs = []
    for folder in folders:
        try:
            runs.append(read_results(folder, CASE_DIR))
        except (KeyError, OSError) as error:
            print(f"skipped {folder.name}: {type(error).__name__}: {error}")
    if not runs:
        raise SystemExit("No result folder could be read")

    output = Path(args.output)
    write_html(runs, output)

    print(f"{'run':22}{'kind':12}{'horizon':>9}{'arcs built':>12}")
    for run in runs:
        built = sum(1 for a in run["arcs"] if a["size"] > 1e-6)
        print(
            f"{run['run']:22}{run['kind']:12}{run['horizon']:>7} h"
            f"{built:>8} /{len(run['arcs']):3}"
        )
    print(f"\nwritten   : {output}")


if __name__ == "__main__":
    main()
