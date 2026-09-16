"""
Freezes the technologies of the small clusters, and gives them back their decisions.

``build.py`` leaves the small clusters as the investment part of the case: their wind,
PV, electrolyzer and storage are sized by the optimizer. A run that is meant to measure
how hard the *operation* of a pipeline network is does not want them, because the sizes
are decided together with the flows and the objective then carries an investment the
network model can shift around. This script pins them: the storage is there already at
a size given on the command line, nothing else is offered, and ``new`` is emptied.

What is left to decide in such a case is the operation alone, and on the linepack side
that is the direction binary and the SOS2 set of every arc and timestep::

    python pin_small_clusters.py pin 400
    python run.py --layout layouts/fixed_network.json --case both --hours 168
    python pin_small_clusters.py restore

The files it touches are the ``Technologies.json`` of the small clusters, saved next to
this script while they are pinned so that ``restore`` puts the case back.

.. note::
    The technologies of the large clusters are pinned by ``build_case`` already, so
    pinning the small ones leaves no size decision anywhere in the model.
"""

import json
import shutil
import sys
from pathlib import Path

CASE = Path(__file__).parent / "input_data"
BACKUP = Path(__file__).parent / ".technologies_backup"
PERIOD = "period1"
SMALL_NODES = ["Arnhem", "Dordrecht", "Venlo"]
STORAGE = "Storage_H2"


def _file(node: str) -> Path:
    """
    Technology list of one node.

    :param str node: node name
    :return: its ``Technologies.json``
    """
    return CASE / PERIOD / "node_data" / node / "Technologies.json"


def pin(size: float):
    """
    Leaves the small clusters a storage that is already there and nothing to decide.

    :param float size: size of the storage of each small cluster, in MWh
    """
    BACKUP.mkdir(parents=True, exist_ok=True)
    for node in SMALL_NODES:
        path = _file(node)
        shutil.copyfile(path, BACKUP / f"{node}.json")
        data = json.loads(path.read_text())
        data["existing"] = {STORAGE: float(size)}
        data["new"] = []
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"{node}: existing {STORAGE} {size} MWh, nothing to decide")


def restore():
    """
    Gives the small clusters their investment decisions back.
    """
    for node in SMALL_NODES:
        saved = BACKUP / f"{node}.json"
        if saved.exists():
            shutil.copyfile(saved, _file(node))
            print(f"{node}: Technologies.json restored")
    shutil.rmtree(BACKUP, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1] == "pin":
        pin(float(sys.argv[2]))
    else:
        restore()
