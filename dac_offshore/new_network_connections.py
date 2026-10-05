from pathlib import Path
from math import radians, sin, cos, asin, sqrt
import pandas as pd

base_dir = Path(__file__).parent / "clean_data"
topo_dir = base_dir / "networks_topology"
s_nom_max_new = 5
s_nom_max_new_electricity = 10


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def distance_km(node_a: str, node_b: str, coords: dict[str, tuple[float, float]]) -> float:
    """Great-circle distance between two nodes, looked up by name in coords."""
    return haversine_km(*coords[node_a], *coords[node_b])


def nearest_nodes(
    node: str,
    candidates: set[str],
    coords: dict[str, tuple[float, float]],
    k: int,
) -> list[str]:
    """Return the k nodes from candidates closest to node (node itself excluded)."""
    others = [c for c in candidates if c != node]
    others.sort(key=lambda c: (distance_km(node, c, coords), c))
    return others[:k]


def existing_electricity_lines() -> set[frozenset[str]]:
    """Node pairs that already have an AC or DC line with capacity (s_nom > 0)."""
    lines = set()
    for name in ["electricityAC.csv", "electricityDC.csv"]:
        df = pd.read_csv(topo_dir / name, sep=";")
        df = df[df["s_nom_max"] > 0]
        lines |= {frozenset((a, b)) for a, b in zip(df["node0"], df["node1"])}
    return lines


def propose_co2_pipelines(n_closest_onshore, n_closest_offshore):
    meta = pd.read_excel(base_dir / "nodes" / "nodes.xlsx").dropna(subset=["x", "y"])
    meta["Node"] = meta["Node"].astype(str).str.strip()

    # node name -> (lon, lat)
    coords = {row.Node: (row.x, row.y) for row in meta.itertuples()}
    all_nodes = set(coords)

    storage = pd.read_csv(base_dir / "co2_storage_limits" / "CO2_storage_limits_2040.csv")
    storage_nodes = set(storage["Node"].astype(str).str.strip()) & all_nodes
    onshore_nodes = set(meta.loc[meta["Type"] == "onshore", "Node"])
    offshore_nodes = set(meta.loc[meta["Type"].str.contains("offshore"), "Node"])

    # Each pipeline is an unordered pair of node names, e.g. {"A", "B"} == {"B", "A"}
    pipelines = set()

    # Rule 1: each storage node -> its nearest onshore node
    for storage_node in sorted(storage_nodes):
        for target in nearest_nodes(storage_node, onshore_nodes, coords, k=1):
            pipelines.add(frozenset((storage_node, target)))

    # Rule 2: each onshore node -> its k-nearest onshore nodes
    for node in sorted(onshore_nodes):
        for target in nearest_nodes(node, onshore_nodes, coords, k=n_closest_onshore):
            pipelines.add(frozenset((node, target)))

    # Rule 3: each offshore node -> its nearest n nodes
    for node in sorted(offshore_nodes):
        for target in nearest_nodes(node, all_nodes, coords, k=n_closest_offshore):
            pipelines.add(frozenset((node, target)))

    rows = []
    for n0, n1 in sorted(tuple(sorted(pipeline)) for pipeline in pipelines):
        rows.append({
            "LinePyHub": f"{n0}-{n1}",
            "length": distance_km(n0, n1, coords),
            "s_nom": 0,
            "s_nom_max": s_nom_max_new,
            "node0": n0,
            "node1": n1,
            "Type": "onshore" if n0 in onshore_nodes and n1 in onshore_nodes else "offshore",
        })

    proposed = pd.DataFrame(rows)
    proposed.to_csv(topo_dir / "CO2_Pipeline.csv", sep=";", index=False)
    print(proposed.round(1).to_string(index=False))


def propose_electricity_lines(n_closest):
    meta = pd.read_excel(base_dir / "nodes" / "nodes.xlsx").dropna(subset=["x", "y"])
    meta["Node"] = meta["Node"].astype(str).str.strip()

    # node name -> (lon, lat)
    coords = {row.Node: (row.x, row.y) for row in meta.itertuples()}
    all_nodes = set(coords)

    storage = pd.read_csv(base_dir / "co2_storage_limits" / "CO2_storage_limits_2040.csv")
    storage_nodes = set(storage["Node"].astype(str).str.strip()) & all_nodes
    onshore_nodes = set(meta.loc[meta["Type"] == "onshore", "Node"])
    existing_lines = existing_electricity_lines()

    # Each line is an unordered pair of node names, e.g. {"A", "B"} == {"B", "A"}
    lines = set()

    # Each storage node -> its nearest onshore node + its n_closest nearest grid nodes
    for storage_node in sorted(storage_nodes):
        nearest_onshore = nearest_nodes(storage_node, onshore_nodes, coords, k=1)
        nearest_other = nearest_nodes(storage_node, all_nodes, coords, k=n_closest)
        for target in nearest_onshore + nearest_other:
            lines.add(frozenset((storage_node, target)))

    # Only propose lines that do not exist yet
    lines -= existing_lines

    rows = []
    for n0, n1 in sorted(tuple(sorted(line)) for line in lines):
        rows.append({
            "LinePyHub": f"{n0}-{n1}",
            "length": distance_km(n0, n1, coords),
            "s_nom": 0,
            "s_nom_max": s_nom_max_new_electricity,
            "node0": n0,
            "node1": n1,
            "LineCountry": f"{n0[0:2]}-{n1[0:2]}",
            "Type": "offshore",
        })

    proposed = pd.DataFrame(rows)
    proposed.to_csv(topo_dir / "proposed_electricityDC.csv", sep=";", index=False)
    print(proposed.round(1).to_string(index=False))


if __name__ == "__main__":
    n_closest_onshore = 3
    n_closest_offshore = 3
    propose_co2_pipelines(n_closest_onshore, n_closest_offshore)

    n_closest = 2
    propose_electricity_lines(n_closest)
