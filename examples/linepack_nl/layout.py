"""
Which corridors of the case exist and which ones can be built.

``build.py`` writes one layout: the corridors of the study exist as a backbone, and
every candidate type is offered on every candidate corridor. This module rewrites that
layout before a run, so that the same case can be solved as a design problem, as a
smaller design problem, or as pure operation.

A layout is a json file with two blocks, each mapping a pipeline type to the corridors
it holds. ``"all"`` stands for the corridors that type holds in the case as ``build.py``
wrote it, i.e. the six corridors of the study for the backbone and the candidate
corridors for a candidate type. A type that is left out is not in the model at all::

    {
      "existing":   {"H2Pipeline_backbone": "all",
                     "H2Pipeline_large": [["Rotterdam", "Arnhem"]]},
      "candidates": {"H2Pipeline_medium": "all"}
    }

The two blocks are what ``Networks.json`` calls ``existing`` and ``new``. An existing
arc is built and paid for: ``var_installed`` is fixed to one, the capex is zero and the
size is the capacity of the type (``fixed_size_pipeline.py:184-222``). A candidate arc
carries the install binary that makes the case a design problem.

.. note::
    A layout with no candidate at all leaves no install binary in the model, which is
    the operational problem of a design that was decided before. :func:`from_design`
    writes that layout out of a solution, so that the design of one run can be operated
    by the next one.

.. note::
    A type that appears in both blocks is two networks to ADOPT, the existing one named
    ``<type>_existing``, so a corridor can carry one pipeline of a type and a second one
    can be built next to it.
"""

import json
import shutil
import tempfile
from pathlib import Path

import h5py
import pandas as pd

import build

SEP = ";"
PERIOD = "period1"
TOPOLOGY = "network_topology"

#: What stands for every corridor the case knows for a type.
ALL = "all"

#: Suffix ADOPT gives the block of an existing network.
EXISTING_SUFFIX = "_existing"


def _topology(input_path: Path) -> Path:
    """
    Folder holding the topology matrices of the case.

    :param Path input_path: input data folder
    :return: the ``network_topology`` folder of the period
    """
    return input_path / PERIOD / TOPOLOGY


def _arc(node_a: str, node_b: str) -> tuple:
    """
    Corridor as it is keyed here, i.e. without a direction.

    :param str node_a: one end
    :param str node_b: the other end
    :return: the two nodes, sorted
    """
    return tuple(sorted((node_a, node_b)))


def _read_matrix(folder: Path, name: str) -> pd.DataFrame:
    """
    Reads one topology matrix.

    :param Path folder: folder of the network
    :param str name: file name, without the extension
    :return: the matrix, nodes on both axes
    """
    return pd.read_csv(folder / f"{name}.csv", sep=SEP, index_col=0)


def _arcs_of(folder: Path) -> list:
    """
    Corridors a network folder declares, read from its connection matrix.

    :param Path folder: folder of the network
    :return: the corridors, each as a sorted pair of nodes
    """
    connection = _read_matrix(folder, "connection")
    arcs = set()
    for node_from in connection.index:
        for node_to in connection.columns:
            if connection.at[node_from, node_to] > 0:
                arcs.add(_arc(node_from, node_to))
    return sorted(arcs)


def case_layout(input_path: Path) -> dict:
    """
    The layout the case carries at the moment.

    :param Path input_path: input data folder
    :return: corridors per type, under ``existing`` and ``candidates``
    """
    networks = json.loads((input_path / PERIOD / "Networks.json").read_text())
    return {
        "existing": {
            name: _arcs_of(_topology(input_path) / "existing" / name)
            for name in networks["existing"]
        },
        "candidates": {
            name: _arcs_of(_topology(input_path) / "new" / name)
            for name in networks["new"]
        },
    }


def known_arcs(input_path: Path) -> dict:
    """
    Every corridor the case knows, with its length.

    A corridor is taken from wherever it is declared, so a type can be offered on a
    corridor another type was written for, e.g. a candidate on a backbone corridor.

    :param Path input_path: input data folder
    :return: length in km, per corridor
    """
    lengths = {}
    for folder in sorted(_topology(input_path).glob("*/*")):
        if not (folder / "connection.csv").exists():
            continue
        distance = _read_matrix(folder, "distance")
        for arc in _arcs_of(folder):
            lengths[arc] = float(distance.at[arc[0], arc[1]])
    return lengths


def own_arcs(input_path: Path) -> dict:
    """
    The corridors each type holds in the case, whether it is existing or a candidate.

    This is what ``"all"`` stands for, so that it means the six corridors of the study
    for the backbone and the candidate corridors for a candidate type.

    :param Path input_path: input data folder
    :return: corridors per type
    """
    own = {}
    for arcs in case_layout(input_path).values():
        for name, declared in arcs.items():
            own.setdefault(name, set()).update(declared)
    return {name: sorted(arcs) for name, arcs in own.items()}


def parse(payload: dict, lengths: dict, own: dict) -> dict:
    """
    Reads a layout and checks it against the corridors and types of the case.

    :param dict payload: content of the layout file
    :param dict lengths: length per corridor, from :func:`known_arcs`
    :param dict own: corridors per type, from :func:`own_arcs`, i.e. what ``"all"``
        stands for
    :return: corridors per type, under ``existing`` and ``candidates``
    """
    unknown = set(payload) - {"existing", "candidates"}
    if unknown:
        raise ValueError(f"a layout has no block {sorted(unknown)}")

    layout = {"existing": {}, "candidates": {}}
    for block in layout:
        for name, arcs in payload.get(block, {}).items():
            if name not in build.PIPELINE_TYPES:
                raise ValueError(f"{name} is not a pipeline type of the case")
            if arcs == ALL:
                if name not in own:
                    raise ValueError(
                        f"{name}: the case declares no corridor for it, so "
                        f'"{ALL}" has nothing to stand for'
                    )
                arcs = own[name]
            parsed = []
            for pair in arcs:
                if len(pair) != 2:
                    raise ValueError(f"{name}: {pair} is not a pair of nodes")
                arc = _arc(*pair)
                if arc not in lengths:
                    raise ValueError(f"{name}: {arc} is not a corridor of the case")
                parsed.append(arc)
            if parsed:
                layout[block][name] = sorted(set(parsed))

    for name in set(layout["existing"]) & set(layout["candidates"]):
        both = set(layout["existing"][name]) & set(layout["candidates"][name])
        if both:
            raise ValueError(
                f"{name}: {sorted(both)} cannot be existing and a candidate at once"
            )

    return layout


def read(path: Path, input_path: Path) -> dict:
    """
    Reads a layout file.

    :param Path path: layout file
    :param Path input_path: input data folder
    :return: corridors per type, under ``existing`` and ``candidates``
    """
    return parse(
        json.loads(Path(path).read_text()),
        known_arcs(input_path),
        own_arcs(input_path),
    )


def dump(layout: dict, path: Path):
    """
    Writes a layout out, in the shape :func:`read` takes back.

    :param dict layout: corridors per type
    :param Path path: file to write
    """
    payload = {
        block: {name: [list(arc) for arc in arcs] for name, arcs in types.items()}
        for block, types in layout.items()
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def from_design(design: dict) -> dict:
    """
    Layout in which the arcs of a solved design exist and nothing can be built.

    :param dict design: ``{(network, arc): ...}``, as :func:`run.collect_network`
        returns it or :func:`design_from_results` reads it
    :return: corridors per type, under ``existing`` and ``candidates``
    """
    existing = {}
    for name, arc in design:
        if name.endswith(EXISTING_SUFFIX):
            name = name[: -len(EXISTING_SUFFIX)]
        existing.setdefault(name, set()).add(_arc(*arc))
    return {
        "existing": {name: sorted(arcs) for name, arcs in sorted(existing.items())},
        "candidates": {},
    }


def design_from_results(folder: Path) -> dict:
    """
    Arcs that are built in a result folder of an earlier run.

    :param Path folder: folder holding ``optimization_results.h5``
    :return: ``{(network, arc): size}`` of the arcs that carry a size
    """
    design = {}
    with h5py.File(Path(folder) / "optimization_results.h5", "r") as results:
        networks = results[f"design/networks/{PERIOD}"]
        for name in networks:
            for arc in networks[name]:
                b_arc = networks[name][arc]
                size = float(b_arc["size"][()])
                if size <= 1e-6:
                    continue
                node_from = b_arc["fromNode"][()].decode()
                node_to = b_arc["toNode"][()].decode()
                design[(name, (node_from, node_to))] = size
    return design


def describe(layout: dict) -> str:
    """
    One line per type, saying how many corridors it holds.

    :param dict layout: corridors per type
    :return: the lines, ready to print
    """
    lines = []
    for block in ("existing", "candidates"):
        for name, arcs in layout[block].items():
            lines.append(f"  {block:11}{name:26}{len(arcs):4} corridors")
    if not layout["candidates"]:
        lines.append("  nothing can be built: the run is operation only")
    return "\n".join(lines) or "  the layout is empty"


def write(input_path: Path, layout: dict) -> Path:
    """
    Writes a layout into the case, after saving the one it replaces.

    The size of an existing arc is the capacity of its type, which is what
    ``FixedSizePipeline`` fixes the size to anyway (``fixed_size_pipeline.py:156-190``),
    and it has to be symmetric for a bidirectional network (``network.py:427``).

    :param Path input_path: input data folder
    :param dict layout: corridors per type
    :return: folder holding the layout that was replaced, for :func:`restore`
    """
    backup = Path(tempfile.mkdtemp(prefix="linepack_nl_layout_"))
    shutil.copytree(_topology(input_path), backup / TOPOLOGY)
    shutil.copyfile(input_path / PERIOD / "Networks.json", backup / "Networks.json")

    lengths = known_arcs(input_path)
    nodes = _node_names(input_path)

    for block, where in (("existing", "existing"), ("candidates", "new")):
        for name in sorted(set(build.PIPELINE_TYPES)):
            folder = _topology(input_path) / where / name
            arcs = layout[block].get(name, [])
            if not arcs:
                if folder.exists():
                    shutil.rmtree(folder)
                continue
            connection, distance = _empty(nodes), _empty(nodes)
            size = _empty(nodes)
            for arc in arcs:
                a, b = arc
                if arc not in lengths:
                    raise KeyError(
                        f"{name}: {arc} is not a corridor the case carries a "
                        "distance for"
                    )
                length = lengths[arc]
                capacity = build.arc_capacity(build.PIPELINE_TYPES[name], length)
                connection.loc[a, b] = connection.loc[b, a] = 1.0
                distance.loc[a, b] = distance.loc[b, a] = length
                size.loc[a, b] = size.loc[b, a] = capacity
            folder.mkdir(parents=True, exist_ok=True)
            connection.to_csv(folder / "connection.csv", sep=SEP)
            distance.to_csv(folder / "distance.csv", sep=SEP)
            if block == "existing":
                size.to_csv(folder / "size.csv", sep=SEP)

    networks = {
        "existing": sorted(layout["existing"]),
        "new": sorted(layout["candidates"]),
    }
    (input_path / PERIOD / "Networks.json").write_text(
        json.dumps(networks, indent=2), encoding="utf-8"
    )

    return backup


def restore(input_path: Path, backup: Path):
    """
    Puts back the layout :func:`write` replaced.

    :param Path input_path: input data folder
    :param Path backup: what :func:`write` returned
    """
    shutil.rmtree(_topology(input_path))
    shutil.copytree(backup / TOPOLOGY, _topology(input_path))
    shutil.copyfile(backup / "Networks.json", input_path / PERIOD / "Networks.json")
    shutil.rmtree(backup, ignore_errors=True)


def discard(backup: Path):
    """
    Throws away a backup that is not the one to restore.

    :param Path backup: what :func:`write` returned
    """
    shutil.rmtree(backup, ignore_errors=True)


def _node_names(input_path: Path) -> list:
    """
    Nodes of the case, in the order the topology matrices carry them.

    :param Path input_path: input data folder
    :return: node names
    """
    for folder in sorted(_topology(input_path).glob("*/*")):
        if (folder / "connection.csv").exists():
            return list(_read_matrix(folder, "connection").index)
    raise FileNotFoundError("the case carries no connection matrix")


def _empty(nodes: list) -> pd.DataFrame:
    """
    Empty square matrix over the nodes, as the topology files expect it.

    :param list nodes: nodes of the case
    :return: matrix of zeros
    """
    return pd.DataFrame(0.0, index=nodes, columns=nodes)
