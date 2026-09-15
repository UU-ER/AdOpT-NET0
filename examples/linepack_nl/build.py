"""
Builds the NL hydrogen case study for the quasi-dynamic pipeline.

The topology is the stylized Dutch hydrogen system. The six large clusters carry the
backbone, which is given, and the three small clusters are not connected to anything
to start with::

    North_Sea --- Rotterdam --- Zeeland --- Chemelot          Arnhem
        |                                      |              Dordrecht
    North_Netherlands ----------------- Zuidwending           Venlo

The backbone is a closed ring, so the direction of each of its pipelines is a real
decision and the linepack of one corridor can be released while another one is being
filled. Everything at the large clusters is existing and frozen at the design of the
study, so the backbone is a boundary condition rather than a decision.

At the small clusters wind, PV, the electrolyzer and the storage are all new, and so
are the connections. Each candidate corridor is offered in three pipeline types, and
the optimizer decides which of them, if any, to build. The types are calibrated on the
peak demand of the small clusters, so that the choice is a real trade-off:

- ``large`` covers the peak demand of any small cluster on any candidate corridor,
  so a cluster connected with it needs neither storage nor local production
- ``medium`` covers the two smaller clusters but not Arnhem, so it has to be
  complemented with storage
- ``small`` covers none of them on its own and is only a supplement

Nothing forbids building two types on the same corridor. Two parallel pipelines are
physical, and the fixed part of the investment cost discourages them, so the case is
left free and what the optimizer does with it is part of the result.

Compressors are switched on. Without them moving hydrogen from a low pressure
pipeline into the backbone would be free, every type would look equally good at the
node and the choice between them would be decided by the investment cost alone.

No energy balance may be violated. A small cluster therefore has exactly three ways
to answer its demand, and the case is the trade-off between them: build wind and PV
and an electrolyzer and enough storage to ride out the lulls, connect itself to a
large cluster with a pipeline, or some mixture. Electricity is not transported and
cannot be imported, so the electricity of a local electrolyzer has to come from the
wind and PV of that same cluster. Hydrogen can be imported, but only at the large
clusters, at the backstop price of the study.

The case is built for a full year at hourly resolution and without typical days, so
the horizon of a run is chosen afterwards with the ``start_period`` and
``end_period`` arguments of ``ModelHub.read_data``.

The time series, the technology costs and the design of the large clusters come from
the study in ``Linepack_modelling``, which is imported for the data only.

.. note::
    The size of a fluidynamic arc is derived from the geometry and not read from the
    json file, so a pinned capacity is not preserved. ``size.csv`` of the backbone is
    rewritten with the derived capacity: for an existing arc it is only the lower
    bound of ``var_size`` (``network.py:578-583``) and the upper bound is the derived
    one, so any smaller value would be silently overruled.

.. note::
    ``build_case`` of the study deletes the whole output folder when it is called
    with ``overwrite``, so it is called without it here and only ``input_data`` is
    removed beforehand. The output folder is the one holding this script.

Run with the environment of the study, which has both packages importable::

    PYTHONPATH=<adopt>;<linepack_modelling> python examples/linepack_nl/build.py
"""

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

SEP = ";"
PERIOD = "period1"
CARRIER = "hydrogen"
STUDY_PATH = Path(r"C:\Users\Masse007\Documents\Code\Linepack_modelling")

#: Large clusters. They carry the backbone and their design is frozen.
LARGE_NODES = [
    "North_Sea",
    "Rotterdam",
    "Zeeland",
    "Chemelot",
    "Zuidwending",
    "North_Netherlands",
]

#: Small clusters. Their technologies and their connections are decided.
SMALL_NODES = ["Arnhem", "Dordrecht", "Venlo"]

#: A corridor is a candidate if it touches a small cluster and is not longer than
#: this. Every large-small and small-small pair of the case is inside 250 km, so that
#: value offers all of them and leaves the choice entirely to the optimizer. Lowering
#: it is the cheapest way to shrink the model, at the price of deciding part of the
#: topology by hand.
MAX_CORRIDOR_KM = 250.0

#: Hourly hydrogen a large cluster may import, as a multiple of its own peak demand.
#: The study caps import at the volume its scenario assumes, which leaves Rotterdam
#: unable to serve its own demand in 63 hours out of 72: it produces 2493 MW on
#: average against a demand of 3429 and may import only 490. The study prices that
#: shortfall as unserved demand, which is what it measures; with the balance forced
#: to close it is infeasible instead.
#:
#: Twice the peak demand leaves room for the largest shortfall of the case, 2475 MW
#: at Rotterdam, without making import unbounded. That matters beyond tidiness: the
#: size of an *existing* compressor is fixed rather than optimised, at
#: ``min(bound of the source, bound of the sink)`` (``utilities.py``), so the import
#: limit sets the size of every Import -> Storage compressor. Raising it to 100 GW
#: moved that minimum from the import to the storage and inflated five of those
#: compressors from 120 MW to 3059 MW, 14.7 GW of compression nobody chose, at a
#: cost that ``var_npv`` carries but the summary does not report.
IMPORT_LIMIT_FACTOR = 2.0

#: Fluid properties of hydrogen, shared by every pipeline type.
FLUID_PROPERTIES = {
    "roughness": 4.5e-05,
    "temperature": 288.15,
    "compressibility_factor": 1.1,
    "molar_mass": 2.016,
    "energy_density": 33.33,
    "nr_breakpoints": 5,
}

#: Pipeline types. The backbone is the one of the study, the three candidate types
#: are calibrated on the 212 MW peak demand of Arnhem, the largest of the small
#: clusters, at a reference length of 100 km: 1.5 times it, half of it and 0.15 of
#: it. The reference pressure is the midpoint of the range, so that the piecewise
#: linear relation is equally accurate above and below it.
PIPELINE_TYPES = {
    "H2Pipeline_backbone": {
        "diameter": 1.2,
        "pressure_min": 30,
        "pressure_max": 70,
        "pressure_ref": 50,
    },
    "H2Pipeline_large": {
        "diameter": 0.35,
        "pressure_min": 15,
        "pressure_max": 30,
        "pressure_ref": 22.5,
    },
    "H2Pipeline_medium": {
        "diameter": 0.22,
        "pressure_min": 15,
        "pressure_max": 30,
        "pressure_ref": 22.5,
    },
    "H2Pipeline_small": {
        "diameter": 0.18,
        "pressure_min": 5,
        "pressure_max": 15,
        "pressure_ref": 10,
    },
}

BACKBONE = "H2Pipeline_backbone"
CANDIDATE_TYPES = ["H2Pipeline_large", "H2Pipeline_medium", "H2Pipeline_small"]

#: Pressure of each component that exchanges hydrogen at a node, in bar. Every pair
#: of them that needs compression gets a compressor, so these values are what makes
#: one pipeline type more expensive to feed than another.
TECHNOLOGY_PRESSURES = {
    "Electrolyzer": {"outlet": 30},
    "Storage_H2": {"inlet": 300, "outlet": 300},
    "Storage_H2_Cavern": {"inlet": 180, "outlet": 180},
}
EXCHANGE_PRESSURES = {
    "Demand": 25.0,
    "Export": 40.0,
    "Import": 40.0,
    "Generic production": 0.0,
}

#: Technologies of a small cluster, and the attribute of the study holding the size
#: its design has. The size is not pinned, it only sets the upper bound.
SMALL_TECHNOLOGIES = {
    "WindTurbine_Onshore_4000": "p_wind",
    "Photovoltaic": "p_solar",
    "Electrolyzer": "p_elec",
    "Storage_H2": "storage_mwh",
}

#: How much larger than the design of the study a small cluster may become.
SIZE_HEADROOM = 3.0

#: Storage gets a far larger headroom than the rest. A cluster that answers the
#: demand locally has to ride out the wind lulls of a whole week, so the size of the
#: study is not a meaningful ceiling for it: leaving it low would decide the
#: trade-off against the local option before the optimizer sees it.
STORAGE_HEADROOM = 20.0
STORAGE_TECHNOLOGY = "Storage_H2"

#: Price of an energy balance that does not close. -1 forbids it outright
#: (template_creation.py:436). ``build_case`` of the study prices it at the hydrogen
#: backstop instead, which in ADOPT applies to *every* carrier at every node
#: (construct_balances.py:428-431), so the electricity of a small cluster could be
#: bought out of nothing. Here every balance has to close on real components.
VIOLATION_PRICE = -1

#: Relative MIP gap. The difference the linepack makes is of the order of a percent,
#: so a looser gap would hide it in solver slack.
MIPGAP = 0.01

#: Solver options for a model of this shape. It is one SOS2 and one direction binary
#: per arc block per timestep, so the tree is large and a first feasible solution is
#: hard to come by: a 168 h run spent 36 minutes in the root cut loop without ever
#: finding one. These are aimed at that, not at closing the gap faster.
#:
#: - ``mipfocus = 1`` asks for feasible solutions rather than for the bound
#: - ``heuristics`` above the 0.05 default spends more of the tree on finding them
#: - ``lpwarmstart = -1`` is the gurobi default; ADOPT forces it off, and the root
#:   LP here is re-solved once per cut round, which is what warm starting is for
#: - ``threads = 4`` rather than every core: the profiling study of the group found
#:   the root cut loop *grows* with the thread count (64 s at 4 threads against
#:   176 s at 16 and 48) while peak memory scales with it
SOLVER_OPTIONS = {
    "mipfocus": 1,
    "heuristics": 0.2,
    "lpwarmstart": -1,
    "threads": 4,
}


def load_spec(n_hours: int = 8760):
    """
    Loads the system of the study.

    :param int n_hours: number of consecutive hours to read, starting on 1 January
    :return: the system description
    """
    sys.path.insert(0, str(STUDY_PATH))
    from adopt_case.common import load_system

    spec = load_system(starts=(0,), n_hours=n_hours)
    missing = [n for n in LARGE_NODES + SMALL_NODES if n not in spec.node_specs]
    if missing:
        raise KeyError(f"Nodes {missing} are not in the study")
    return spec


def backbone_spec(spec):
    """
    Restricts the corridors of the study to the backbone ring.

    The distribution corridors of the study are dropped, they are replaced by the
    candidate corridors written afterwards.

    :param spec: the system description
    :return: the system description with only the backbone corridors
    """
    edges = [e for e in spec.edges if e["pipeline_class"] == "backbone"]
    return replace(spec, edges=edges)


def candidate_corridors(distances: pd.DataFrame) -> dict:
    """
    Corridors offered to the optimizer, i.e. the ones touching a small cluster.

    :param pd.DataFrame distances: distance between every pair of nodes, in km
    :return: length of each candidate corridor, in km
    """
    corridors = {}
    for small in SMALL_NODES:
        for other in LARGE_NODES + SMALL_NODES:
            if other == small or (other, small) in corridors:
                continue
            length = float(distances.at[small, other])
            if 0 < length <= MAX_CORRIDOR_KM:
                corridors[(small, other)] = length
    return corridors


def arc_capacity(properties: dict, length_km: float) -> float:
    """
    Capacity of an arc, i.e. the flow at the largest pressure difference.

    Mirrors ``FluidynamicPipeline.fit_network_performance``: the coefficient of the
    quasi-dynamic pipeline equation is derived from the geometry, and the capacity is
    the flow at the last breakpoint, with the lower pressure at the reference one.

    :param dict properties: geometry and fluid properties of the pipeline type
    :param float length_km: length of the arc, in km
    :return: capacity of the arc, in MW
    """
    p = {**FLUID_PROPERTIES, **properties}
    r_specific = 8314 / p["molar_mass"]
    friction_factor = (2 * np.log10(3.7 * p["diameter"] / p["roughness"])) ** -2
    r_per_km = (
        p["energy_density"] ** 2
        * 3.6**2
        * np.pi**2
        / 16
        * p["diameter"] ** 5
        * 1e10
        / (
            friction_factor
            * 1000
            * p["compressibility_factor"]
            * r_specific
            * p["temperature"]
        )
    )
    delta_pressure = p["pressure_max"] - p["pressure_ref"]
    return float(
        np.sqrt(
            r_per_km
            / length_km
            * delta_pressure
            * (2 * p["pressure_ref"] + delta_pressure)
        )
    )


def _write_json(path: Path, payload: dict):
    """
    Writes one json file, creating the folder if needed.

    :param Path path: file to write
    :param dict payload: content of the file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _matrix(nodes: list) -> pd.DataFrame:
    """
    Empty square matrix over the nodes, as the topology files expect it.

    :param list nodes: nodes of the case
    :return: matrix of zeros
    """
    return pd.DataFrame(0.0, index=nodes, columns=nodes)


def network_json(name: str, gammas: dict) -> dict:
    """
    Network json of one pipeline type.

    The scalar pressure is set to the reference pressure of the type: it is what the
    compressors see, while the pressure of the pipeline itself stays free between
    ``pressure_min`` and ``pressure_max``.

    :param str name: name of the pipeline type
    :param dict gammas: investment cost coefficients of its pressure class
    :return: content of the network json
    """
    properties = {**FLUID_PROPERTIES, **PIPELINE_TYPES[name]}
    return {
        "network_type": "fluidynamic_pipeline",
        "size_is_int": 0,
        "capex_defined_per_arc": 0,
        "size_max_defined_per_arc": 0,
        "decommission": "impossible",
        "size_min": 0,
        "size_max": 1e6,
        "Economics": {
            **gammas,
            "opex_variable": 0.0,
            "opex_fixed": 0.04,
            "discount_rate": 0.05,
            "lifetime": 25,
            "decommission_cost": 0.0,
        },
        "Performance": {
            "carrier": CARRIER,
            "bidirectional_network": 1,
            "bidirectional_network_precise": 0,
            "rated_capacity": 1,
            "loss": 0.0,
            "min_transport": 0.0,
            "loss2emissions": 0.0,
            "emissionfactor": 0.0,
            "energyconsumption": {},
            "pressure": {
                CARRIER: {
                    "inlet": properties["pressure_ref"],
                    "outlet": properties["pressure_ref"],
                }
            },
            **properties,
        },
        "Units": {"size": "MW", "transport_carrier": {CARRIER: "MW"}},
    }


def write_networks(input_path: Path, spec, corridors: dict) -> dict:
    """
    Writes the backbone and the candidate pipeline types.

    The backbone keeps the corridors of the study and stays existing. Every candidate
    type is offered on every candidate corridor, so the optimizer picks the type per
    corridor, or none of them.

    :param Path input_path: input data folder
    :param spec: the system description
    :param dict corridors: length of each candidate corridor, in km
    :return: capacity of each arc of each type, in MW
    """
    sys.path.insert(0, str(STUDY_PATH))
    from adopt_case.common import class_gammas

    period_path = input_path / PERIOD
    nodes = spec.node_names
    topology_path = period_path / "network_topology"

    _write_json(
        period_path / "Networks.json",
        {"existing": [BACKBONE], "new": list(CANDIDATE_TYPES)},
    )

    # The backbone is priced as the backbone of the study, the candidate types as the
    # class their pressure range belongs to
    gammas_of = {
        BACKBONE: class_gammas("backbone"),
        "H2Pipeline_large": class_gammas("distribution"),
        "H2Pipeline_medium": class_gammas("distribution"),
        "H2Pipeline_small": class_gammas("local"),
    }

    capacities = {}

    # Backbone, existing
    _write_json(
        period_path / "network_data" / f"{BACKBONE}.json",
        network_json(BACKBONE, gammas_of[BACKBONE]),
    )
    connection, distance, size = _matrix(nodes), _matrix(nodes), _matrix(nodes)
    for edge in spec.edges:
        a, b, length = edge["node_a"], edge["node_b"], edge["length_km"]
        capacity = arc_capacity(PIPELINE_TYPES[BACKBONE], length)
        connection.loc[a, b] = connection.loc[b, a] = 1.0
        distance.loc[a, b] = distance.loc[b, a] = length
        # must be symmetric for a bidirectional existing network (network.py:427)
        size.loc[a, b] = size.loc[b, a] = capacity
        capacities[(BACKBONE, a, b)] = capacity
    target = topology_path / "existing" / BACKBONE
    target.mkdir(parents=True, exist_ok=True)
    connection.to_csv(target / "connection.csv", sep=SEP)
    distance.to_csv(target / "distance.csv", sep=SEP)
    size.to_csv(target / "size.csv", sep=SEP)

    # Candidate types, new
    for name in CANDIDATE_TYPES:
        _write_json(
            period_path / "network_data" / f"{name}.json",
            network_json(name, gammas_of[name]),
        )
        connection, distance = _matrix(nodes), _matrix(nodes)
        for (a, b), length in corridors.items():
            connection.loc[a, b] = connection.loc[b, a] = 1.0
            distance.loc[a, b] = distance.loc[b, a] = length
            capacities[(name, a, b)] = arc_capacity(PIPELINE_TYPES[name], length)
        target = topology_path / "new" / name
        target.mkdir(parents=True, exist_ok=True)
        connection.to_csv(target / "connection.csv", sep=SEP)
        distance.to_csv(target / "distance.csv", sep=SEP)

    # The distribution network of the study is no longer part of the case
    stale = period_path / "network_data" / "H2Pipeline_distribution.json"
    if stale.exists():
        stale.unlink()

    return capacities


def free_small_clusters(input_path: Path, spec):
    """
    Turns the technologies of the small clusters into investment decisions.

    ``build_case`` pins every technology and zeroes its investment cost, which is what
    the study needs and what the large clusters keep here. At the small clusters the
    sizes are decided instead, so the costs of the study are written back and the
    design it carries becomes the upper bound rather than the answer.

    :param Path input_path: input data folder
    :param spec: the system description
    """
    sys.path.insert(0, str(STUDY_PATH))
    from adopt_case.build_case import ELECTROLYZER_ETA, storage_unit_capex_eur_per_mwh
    from Case_studies_prod_dem.components import costs

    period_path = input_path / PERIOD
    price_storage = storage_unit_capex_eur_per_mwh()

    for node in SMALL_NODES:
        node_spec = spec.node_specs[node]
        node_path = period_path / "node_data" / node
        available = [
            name
            for name, attribute in SMALL_TECHNOLOGIES.items()
            if node_spec.get(attribute, 0) > 0
        ]
        _write_json(node_path / "Technologies.json", {"existing": {}, "new": available})

        for name in available:
            tec_file = node_path / "technology_data" / f"{name}.json"
            data = json.loads(tec_file.read_text())
            economics = data["Economics"]
            economics["capex_model"] = 1
            economics["fix_capex"] = 0.0
            design = float(node_spec[SMALL_TECHNOLOGIES[name]])

            if name == "WindTurbine_Onshore_4000":
                economics["unit_capex"] = costs.wind_capex(1.0)
                economics["opex_fixed"] = costs.OPEX_FRAC["wind"]
            elif name == "Photovoltaic":
                economics["unit_capex"] = costs.solar_capex(1.0)
                economics["opex_fixed"] = costs.OPEX_FRAC["solar"]
            elif name == "Electrolyzer":
                # the specific investment cost of the study depends on the size, a
                # MILP needs a constant, so it is evaluated at the design size
                economics["unit_capex"] = costs.electrolyzer_capex(design) / design
                economics["opex_fixed"] = costs.OPEX_FRAC["electrolyzer"]
                data["Performance"]["performance"] = {
                    "in": [0, 1],
                    "out": [0, ELECTROLYZER_ETA],
                }
                data["Performance"]["min_part_load"] = 0
            elif name == "Storage_H2":
                economics["unit_capex"] = price_storage
                economics["opex_fixed"] = costs.OPEX_FRAC["storage"]

            data["size_min"] = 0
            headroom = STORAGE_HEADROOM if name == STORAGE_TECHNOLOGY else SIZE_HEADROOM
            data["size_max"] = design * headroom
            data["decommission"] = "impossible"
            _write_json(tec_file, data)


def open_the_imports(input_path: Path, spec):
    """
    Lets a large cluster import enough hydrogen to close its own balance.

    See :data:`IMPORT_LIMIT_FACTOR`. The price is the backstop of the study, so
    import stays the most expensive way to serve a demand and is used only where
    nothing else reaches. Small clusters keep no import at all: whether they are
    served locally or through a pipeline is the decision the case is about.

    :param Path input_path: input data folder
    :param spec: the system description
    :return: the limit given to each node, in MW
    """
    sys.path.insert(0, str(STUDY_PATH))
    from adopt_case.build_case import import_price_eur_per_mwh

    price = import_price_eur_per_mwh()
    period_path = input_path / PERIOD
    limits = {}
    for node in LARGE_NODES:
        peak = (
            float(spec.demand_mw[node].max()) if node in spec.demand_mw.columns else 0.0
        )
        limits[node] = peak * IMPORT_LIMIT_FACTOR
        carrier_file = (
            period_path / "node_data" / node / "carrier_data" / f"{CARRIER}.csv"
        )
        series = pd.read_csv(carrier_file, sep=SEP, index_col=0)
        series["Import limit"] = limits[node]
        series["Import price"] = price
        series.to_csv(carrier_file, sep=SEP)

    return limits


def tighten_pinned_sizes(input_path: Path):
    """
    Brings the declared size of a frozen technology down to the size it is frozen at.

    The technology database ships headroom for a whole country: ``size_max`` is
    1 500 000 for the onshore turbine and 6e9 for the salt cavern, and
    ``_tune_technology_data`` of the study keeps it even where the design is pinned.
    The bound of ``var_output`` follows from it, so the model carries variables
    declared five orders of magnitude above anything reachable, which is where the
    ``[4e-01, 2e+08]`` bounds range of the solver log comes from.

    It is not only cosmetic. ``determine_flow_existing_compressors`` sizes an existing
    compressor from ``max(var_output.ub)`` of the component it feeds, so the same
    invented headroom also sizes the compressors.

    :param Path input_path: input data folder
    :return: how many technologies were tightened, and by what factor at worst
    """
    period_path = input_path / PERIOD
    tightened, worst = 0, 1.0
    for node in LARGE_NODES:
        node_path = period_path / "node_data" / node
        existing = json.loads((node_path / "Technologies.json").read_text())["existing"]
        for name, size in existing.items():
            tec_file = node_path / "technology_data" / f"{name}.json"
            if not tec_file.exists():
                continue
            data = json.loads(tec_file.read_text())
            declared = float(data.get("size_max", size))
            if declared <= size:
                continue
            data["size_max"] = float(size)
            _write_json(tec_file, data)
            tightened += 1
            worst = max(worst, declared / size if size else declared)

    return tightened, worst


def close_the_balances(input_path: Path):
    """
    Forbids an energy balance that does not close, and sets the solver options.

    ``build_case`` of the study prices a violation at the hydrogen backstop, so that
    unserved hydrogen demand is expensive rather than infeasible. That is what the
    study needs, but the price applies to every carrier at every node, and a small
    cluster whose technologies are a decision can then buy the electricity of its
    electrolyzer out of nothing. Every balance has to close on real components here,
    so a small cluster either produces its hydrogen locally or is connected.

    Hydrogen can still be imported at the large clusters, at the price the study
    gives it, which is the backstop the system really has.

    :param Path input_path: input data folder
    """
    config_file = input_path / "ConfigModel.json"
    config = json.loads(config_file.read_text())
    config["energybalance"]["violation"]["value"] = VIOLATION_PRICE
    config["solveroptions"]["mipgap"]["value"] = MIPGAP
    for option, value in SOLVER_OPTIONS.items():
        config["solveroptions"][option]["value"] = value
    _write_json(config_file, config)


def enable_compressors(input_path: Path, spec):
    """
    Switches the pressure of the model on and gives every component its pressure.

    A compressor is created for each pair of components exchanging hydrogen at a node
    whose pressures require it (``handle_input_data.py:850-935``), so these values are
    what makes one pipeline type more expensive to feed than another.

    The repository ships no compressor template for hydrogen, so ``copy_compressor_data``
    finds nothing. The json used here is a data file of this example instead.

    :param Path input_path: input data folder
    :param spec: the system description
    """
    config_file = input_path / "ConfigModel.json"
    config = json.loads(config_file.read_text())
    config["performance"]["pressure"]["pressure_on"]["value"] = 1
    config["performance"]["pressure"]["pressure_carriers"]["value"] = [CARRIER]
    _write_json(config_file, config)

    period_path = input_path / PERIOD
    for node in spec.node_names:
        node_path = period_path / "node_data" / node
        _write_json(
            node_path / "carrier_data" / "PressureExchangeData.json",
            {
                CARRIER: {
                    name: {"value": value, "unit": "bar"}
                    for name, value in EXCHANGE_PRESSURES.items()
                }
            },
        )
        for tec_file in (node_path / "technology_data").glob("*.json"):
            pressure = TECHNOLOGY_PRESSURES.get(tec_file.stem)
            if pressure is None:
                continue
            data = json.loads(tec_file.read_text())
            data["Performance"]["pressure"] = {CARRIER: dict(pressure)}
            _write_json(tec_file, data)

    compressor_dir = period_path / "compressor_data"
    compressor_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        Path(__file__).parent / "compressor_hydrogen.json",
        compressor_dir / f"{CARRIER}.json",
    )


def node_distances(spec) -> pd.DataFrame:
    """
    Distance between every pair of nodes, as the study measures them.

    :param spec: the system description
    :return: distance matrix, in km
    """
    path = STUDY_PATH / spec.study.data_dir / spec.study.distance_file
    raw = pd.read_csv(path, index_col=0)
    return raw.loc[spec.node_names, spec.node_names]


def main():
    """
    Builds the case study in ``examples/linepack_nl/input_data``.
    """
    sys.path.insert(0, str(STUDY_PATH))
    from adopt_case.build_case import build_case

    out_dir = Path(__file__).parent
    spec = load_spec()
    backbone = backbone_spec(spec)
    corridors = candidate_corridors(node_distances(spec))

    # ``overwrite`` would delete out_dir, which is the folder holding this script
    if (out_dir / "input_data").exists():
        shutil.rmtree(out_dir / "input_data")

    # The large clusters are built pinned, the small ones are freed afterwards
    input_path = build_case(backbone, out_dir, pin_design=True, overwrite=False)
    capacities = write_networks(input_path, backbone, corridors)
    free_small_clusters(input_path, spec)
    import_limits = open_the_imports(input_path, spec)
    tightened, worst = tighten_pinned_sizes(input_path)
    enable_compressors(input_path, spec)
    close_the_balances(input_path)

    # The climate data written by the study is synthetic, so the capacity factors
    # that ADOPT would fit from it are not the ones of the study. They are written
    # out here and read back by run.py, which keeps the run itself independent of
    # the study.
    pd.DataFrame({"wind": spec.cf_wind, "solar": spec.cf_solar}).to_csv(
        out_dir / "res_capacity_factors.csv", sep=SEP, index_label="timestep"
    )

    print(f"\nCase built in {input_path}")

    peaks = spec.demand_mw[SMALL_NODES].max()
    print("\npeak demand of the small clusters [MW]")
    print("  " + "   ".join(f"{n} {peaks[n]:.0f}" for n in SMALL_NODES))

    print("\ncandidate corridors, capacity [MW] per type")
    header = f"{'corridor':32}{'km':>7}"
    header += "".join(f"{n.replace('H2Pipeline_', ''):>10}" for n in CANDIDATE_TYPES)
    print(header)
    for (a, b), length in sorted(corridors.items(), key=lambda kv: kv[1]):
        row = f"{a + ' - ' + b:32}{length:7.1f}"
        row += "".join(f"{capacities[(n, a, b)]:10.0f}" for n in CANDIDATE_TYPES)
        print(row)

    print(
        f"\n{tightened} frozen technologies had their declared size brought down to "
        f"the size they are frozen at, by up to a factor {worst:,.0f}"
    )

    print("\nimport limit per large cluster [MW]")
    for node, limit in import_limits.items():
        print(f"  {node:22}{limit:12,.0f}")

    print("\nbackbone, capacity [MW]")
    for (name, a, b), capacity in capacities.items():
        if name == BACKBONE:
            print(f"  {a + ' - ' + b:38}{capacity:10.0f}")


if __name__ == "__main__":
    main()
