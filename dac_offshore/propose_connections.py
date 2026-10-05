from pathlib import Path
from math import radians, sin, cos, asin, sqrt
import pandas as pd

base_dir = Path(__file__).parent / "clean_data"
topo_dir = base_dir / "networks_topology"
n_closest = 5
s_nom_max_new = 10


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def closest(node, candidates, pos, n):
    ranked = sorted((haversine_km(*pos[node], *pos[c]), c) for c in candidates if c != node)
    return [c for _, c in ranked[:n]]


def existing_pairs():
    pairs = set()
    for name in ["electricityAC.csv", "electricityDC.csv"]:
        df = pd.read_csv(topo_dir / name, sep=";")
        df = df[df["s_nom"] > 0]
        pairs |= {frozenset((a, b)) for a, b in zip(df["node0"], df["node1"])}
    return pairs


def propose():
    meta = pd.read_excel(base_dir / "nodes" / "nodes.xlsx").dropna(subset=["x", "y"])
    meta["Node"] = meta["Node"].astype(str).str.strip()
    pos = {row.Node: (row.x, row.y) for row in meta.itertuples()}

    storage = pd.read_csv(base_dir / "co2_storage_limits" / "CO2_storage_limits_2040.csv")
    storage_nodes = sorted(set(storage["Node"].astype(str).str.strip()) & set(pos))
    onshore = set(meta.loc[meta["Type"] == "onshore", "Node"])
    electricity = set(meta.loc[meta["Type"].isin(["onshore", "offshore_existing"]), "Node"])
    built = existing_pairs()

    rows = {}
    for node in storage_nodes:
        targets = closest(node, onshore, pos, 1) + closest(node, electricity, pos, n_closest)
        for target in targets:
            pair = frozenset((node, target))
            if pair in built or pair in rows:
                continue
            n0, n1 = sorted(pair)
            rows[pair] = {
                "LinePyHub": f"{n0}-{n1}",
                "length": haversine_km(*pos[n0], *pos[n1]),
                "s_nom": 0,
                "s_nom_max": s_nom_max_new,
                "node0": n0,
                "node1": n1,
                "Type": "offshore",
            }

    proposed = pd.DataFrame(rows.values())
    proposed.to_csv(topo_dir / "proposed_electricityDC.csv", sep=";", index=False)
    print(proposed.round(1).to_string(index=False))


if __name__ == "__main__":
    propose()
