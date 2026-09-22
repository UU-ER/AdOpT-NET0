"""
Which technologies of the small clusters are dimensioned and which ones are given.

:mod:`layout` does this for the corridors, this module does it for the technologies, and
the two together are the four ways a rung can be posed:

===============  ================  =============================================
``--design``     network           technologies
===============  ================  =============================================
``all``          decided           decided        the full investment problem
``network``      decided           given          only the corridors are dimensioned
``tecs``         given             decided        only the clusters are dimensioned
``none``         given             given          pure operation, no investment left
===============  ================  =============================================

A technology is *given* by setting ``size_min`` and ``size_max`` to the same value, so
that ``var_size`` is fixed by its own bounds (``technology.py:592-595``). Nothing else
changes: the technology is still built and still paid for, the investment cost is simply
a constant of the run. That is deliberate — the objective of a given run stays comparable
with the objective of a run that decided the same design, which it would not be if the
investment cost were dropped.

Where the given sizes come from is the ``--from`` of ``run.py``:

- the design of the study, which ``build.py`` writes next to the case as
  ``reference_design.json``. This is the default and needs nothing to have been solved
- a result folder of an earlier run, which is how the design one rung finds is operated
  under a different pipeline model, or on a longer horizon

.. note::
    Only the small clusters are touched. The technologies of the large clusters are
    frozen by ``build.py`` already, and the compressors are sized by ADOPT out of the
    components they feed, not by anything here.

.. note::
    A size read back from a solved run is a solver answer, so it carries the solver
    tolerance with it. It is written as it is: rounding it to something tidy would be a
    different design, and the operation run would no longer be the operation of the
    design that was found.
"""

import json
import shutil
import tempfile
from pathlib import Path

import h5py

PERIOD = "period1"

#: Sizes below this are read as "not built" and pinned to zero rather than to the
#: solver residue they carry.
TOLERANCE = 1e-6

#: The technology build.py gives its own headroom, see :func:`headroom`.
STORAGE_TECHNOLOGY = "Storage_H2"

#: What ``--design`` may ask for: whether the network and the technologies are decided.
MODES = {
    "all": {"network": True, "tecs": True},
    "network": {"network": True, "tecs": False},
    "tecs": {"network": False, "tecs": True},
    "none": {"network": False, "tecs": False},
}


def _tec_file(input_path: Path, node: str, name: str) -> Path:
    """
    Json of one technology at one node.

    :param Path input_path: input data folder
    :param str node: node the technology sits at
    :param str name: name of the technology
    :return: the file, which may not exist
    """
    return input_path / PERIOD / "node_data" / node / "technology_data" / f"{name}.json"


def case_design(input_path: Path, small: list) -> dict:
    """
    The bounds the case carries at the moment, per node and technology.

    :param Path input_path: input data folder
    :param list small: small clusters of the rung
    :return: ``{node: {technology: (size_min, size_max)}}``
    """
    bounds = {}
    for node in small:
        node_path = input_path / PERIOD / "node_data" / node
        declared = json.loads((node_path / "Technologies.json").read_text())
        bounds[node] = {}
        for name in declared["new"]:
            data = json.loads(_tec_file(input_path, node, name).read_text())
            bounds[node][name] = (
                float(data.get("size_min", 0.0)),
                float(data.get("size_max", 0.0)),
            )
    return bounds


def read(path: Path) -> dict:
    """
    Reads a design file, i.e. a size per node and technology.

    :param Path path: the file, as ``build.py`` writes ``reference_design.json``
    :return: ``{node: {technology: size}}``
    """
    payload = json.loads(Path(path).read_text())
    return {
        node: {name: float(size) for name, size in sizes.items()}
        for node, sizes in payload.items()
    }


def from_results(folder: Path, small: list) -> dict:
    """
    Sizes a solved run gave the technologies of the small clusters.

    A technology the run did not build comes back as zero rather than being left out, so
    that pinning the design really forbids it instead of leaving it free.

    :param Path folder: folder holding ``optimization_results.h5``
    :param list small: small clusters of the rung
    :return: ``{node: {technology: size}}``
    """
    design = {}
    with h5py.File(Path(folder) / "optimization_results.h5", "r") as results:
        nodes = results[f"design/nodes/{PERIOD}"]
        for node in small:
            if node not in nodes:
                continue
            design[node] = {}
            for name in nodes[node]:
                group = nodes[node][name]
                if "size" not in group or "_Compressor_" in name:
                    continue
                size = float(group["size"][()])
                design[node][name] = size if size > TOLERANCE else 0.0
    return design


def dump(design: dict, path: Path):
    """
    Writes a design out, in the shape :func:`read` takes back.

    :param dict design: ``{node: {technology: size}}``
    :param Path path: file to write
    """
    Path(path).write_text(json.dumps(design, indent=2), encoding="utf-8")


def describe(design: dict) -> str:
    """
    One line per technology that is pinned.

    :param dict design: ``{node: {technology: size}}``
    :return: the lines, ready to print
    """
    lines = []
    for node, sizes in sorted(design.items()):
        for name, size in sorted(sizes.items()):
            note = "  (not built)" if size <= TOLERANCE else ""
            lines.append(f"  {node:12}{name:28}{size:14,.1f}{note}")
    return "\n".join(lines) or "  the design is empty"


def pin(input_path: Path, design: dict, small: list) -> Path:
    """
    Fixes the size of every technology of the small clusters, and backs up what it
    replaces.

    A technology the design does not mention is pinned to zero: a design that is given
    has to say what the cluster is, not only what part of it is worth mentioning.

    :param Path input_path: input data folder
    :param dict design: ``{node: {technology: size}}``
    :param list small: small clusters of the rung
    :return: folder holding the technology jsons that were replaced, for :func:`restore`
    """
    backup = Path(tempfile.mkdtemp(prefix="ladder_design_"))

    for node in small:
        node_path = input_path / PERIOD / "node_data" / node
        declared = json.loads((node_path / "Technologies.json").read_text())
        sizes = design.get(node, {})
        for name in declared["new"]:
            tec_file = _tec_file(input_path, node, name)
            target = backup / node
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tec_file, target / f"{name}.json")

            size = float(sizes.get(name, 0.0))
            data = json.loads(tec_file.read_text())
            data["size_min"] = size
            data["size_max"] = size
            tec_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return backup


def headroom(
    input_path: Path,
    reference: dict,
    small: list,
    factor: float = None,
    storage_factor: float = None,
) -> Path:
    """
    Overrides the upper bound of the small clusters' technologies at run time.

    ``build.py`` bakes ``SIZE_HEADROOM``/``STORAGE_HEADROOM`` into ``size_max`` when the
    rung is built (``build.py:156-162``). This rewrites ``size_max`` to ``factor`` times
    the design of the study instead, without rebuilding the rung. A technology whose
    factor is left as ``None`` keeps the bound ``build.py`` wrote.

    :param Path input_path: input data folder
    :param dict reference: design of the study, as :func:`read` returns it
    :param list small: small clusters of the rung
    :param float factor: multiple of the study design for every technology but storage
    :param float storage_factor: multiple of the study design for storage
    :return: folder holding the technology jsons that were replaced, for :func:`restore`
    """
    backup = Path(tempfile.mkdtemp(prefix="ladder_headroom_"))

    for node in small:
        node_path = input_path / PERIOD / "node_data" / node
        declared = json.loads((node_path / "Technologies.json").read_text())
        sizes = reference.get(node, {})
        for name in declared["new"]:
            chosen = storage_factor if name == STORAGE_TECHNOLOGY else factor
            if chosen is None:
                continue
            tec_file = _tec_file(input_path, node, name)
            target = backup / node
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tec_file, target / f"{name}.json")

            data = json.loads(tec_file.read_text())
            data["size_max"] = float(sizes.get(name, 0.0)) * chosen
            tec_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return backup


def restore(input_path: Path, backup: Path):
    """
    Puts back the technology jsons :func:`pin` replaced.

    :param Path input_path: input data folder
    :param Path backup: what :func:`pin` returned
    """
    backup = Path(backup)
    for node_folder in backup.iterdir():
        if not node_folder.is_dir():
            continue
        for tec_file in node_folder.glob("*.json"):
            shutil.copyfile(
                tec_file, _tec_file(input_path, node_folder.name, tec_file.stem)
            )
    shutil.rmtree(backup, ignore_errors=True)


def discard(backup: Path):
    """
    Throws away a backup that is not the one to restore.

    :param Path backup: what :func:`pin` returned
    """
    shutil.rmtree(backup, ignore_errors=True)
