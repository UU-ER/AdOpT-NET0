"""
The rungs of the ladder, i.e. which nodes of the NL study each case carries.

The full case of ``examples/linepack_nl`` is six large clusters and three small ones,
which is large enough that a run of it is an afternoon. The rungs here are the same
system seen through a smaller window: the same demand, the same capacity factors, the
same costs and the same pipeline types, restricted to a handful of nodes. A result of a
rung therefore carries over to the full case, and a rung that does not solve says
something about the full case rather than about a toy.

The ladder grows in two directions at once, and the two are worth separating when a run
misbehaves:

- **small clusters** decide how many *candidate corridors* there are, which is where the
  install binaries are, i.e. the size of the design problem
- **large clusters** decide how much *backbone* there is, which is where the ring and
  therefore the routing choice is, i.e. the size of the operational problem

``1L1S`` has one candidate corridor and no backbone at all, so it is a pure
"connect or produce locally" question with no routing in it. ``3L3S`` has a backbone
path, every small cluster within reach of every large one, and is the first rung where
the linepack of one corridor can be released while another one is filled.

.. note::
    A rung with one large cluster has no backbone corridor, because the backbone of the
    study only ever joins large clusters. :mod:`build` then writes no existing network
    at all and the case carries candidates only, which is legitimate but worth knowing
    when a layout file mentions ``H2Pipeline_backbone``.

.. note::
    The large clusters of a rung are chosen so that the backbone corridors between them
    are corridors the study really has: ``Rotterdam - Zeeland`` and
    ``Zeeland - Chemelot``. Picking two large clusters the ring does not join directly
    would leave them unconnected, and the rung would be two separate systems.
"""

#: Node sets of each rung: ``large`` carries the backbone and is frozen at the design
#: of the study, ``small`` is where the technologies and the connections are decided.
#: Ordered from the smallest rung to the largest.
RUNGS = {
    "1L1S": {
        "large": ["Rotterdam"],
        "small": ["Dordrecht"],
        "about": "one large cluster and the nearest small one, 20 km apart",
    },
    "1L2S": {
        "large": ["Rotterdam"],
        "small": ["Dordrecht", "Venlo"],
        "about": "one large cluster, a small one next to it and one 132 km away",
    },
    "1L3S": {
        "large": ["Rotterdam"],
        "small": ["Dordrecht", "Venlo", "Arnhem"],
        "about": "one large cluster and every small cluster of the study",
    },
    "2L2S": {
        "large": ["Rotterdam", "Zeeland"],
        "small": ["Dordrecht", "Venlo"],
        "about": "two large clusters joined by one backbone corridor, two small ones",
    },
    "3L3S": {
        "large": ["Rotterdam", "Zeeland", "Chemelot"],
        "small": ["Dordrecht", "Venlo", "Arnhem"],
        "about": "a backbone path of three large clusters, every small cluster",
    },
}


def nodes(rung: str) -> tuple:
    """
    Large and small clusters of a rung.

    :param str rung: name of the rung
    :return: the large clusters and the small ones
    """
    if rung not in RUNGS:
        raise KeyError(f"{rung} is not a rung, choose from {sorted(RUNGS)}")
    return list(RUNGS[rung]["large"]), list(RUNGS[rung]["small"])


def describe() -> str:
    """
    One line per rung, in the order they are meant to be climbed.

    :return: the lines, ready to print
    """
    lines = []
    for name, rung in RUNGS.items():
        large = ", ".join(rung["large"])
        small = ", ".join(rung["small"])
        lines.append(f"  {name:8}{large:38}{small:34}{rung['about']}")
    return "\n".join(lines)
