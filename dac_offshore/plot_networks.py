from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import geopandas as gpd
import geodatasets

here = Path(__file__).parent
base_dir = here / "clean_data"
topo_dir = base_dir / "networks_topology"

colors = {"storage/dac": "#d62728", "new offshore wind": "#ff7f0e", "other": "#2ca02c"}


def load_nodes():
    meta = pd.read_excel(base_dir / "nodes" / "nodes.xlsx").dropna(subset=["x", "y"])
    meta["Node"] = meta["Node"].astype(str).str.strip()
    storage = pd.read_csv(base_dir / "co2_storage_limits" / "CO2_storage_limits_2040.csv")
    storage_nodes = set(storage["Node"].astype(str).str.strip())

    pos, kind = {}, {}
    for _, row in meta.iterrows():
        node = row["Node"]
        pos[node] = (row["x"], row["y"])
        if node in storage_nodes:
            kind[node] = "storage/dac"
        elif str(row["New offshore wind?"]).strip().lower() == "yes":
            kind[node] = "new offshore wind"
        else:
            kind[node] = "other"
    return pos, kind


def load_edges(names, pos):
    edges = []
    for name in names:
        df = pd.read_csv(topo_dir / name, sep=";")
        edges += [(a, b) for a, b in zip(df["node0"], df["node1"]) if a in pos and b in pos]
    return edges


def plot(pos, kind, proposed, background, title, filename):
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xmin, xmax, ymin, ymax = min(xs) - 1.5, max(xs) + 1.5, min(ys) - 1.5, max(ys) + 1.5

    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    fig, ax = plt.subplots(figsize=(12, 12))
    world.cx[xmin - 3.5:xmax + 3.5, ymin - 3.5:ymax + 3.5].plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc")

    for a, b in background:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#333333", lw=1, alpha=0.5)
    for a, b in proposed:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="tab:red", lw=2, ls="--", alpha=0.9)

    for node, (x, y) in pos.items():
        ax.scatter(x, y, s=60, color=colors[kind[node]], edgecolors="white", zorder=3)
        ax.text(x, y + 0.15, node, fontsize=7, ha="center", zorder=4)

    handles = [mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=k)
               for k, c in colors.items()]
    handles.append(mlines.Line2D([0], [0], color="tab:red", lw=2, ls="--", label="proposed"))
    if background:
        handles.append(mlines.Line2D([0], [0], color="#333333", lw=1.5, label="existing"))

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")
    ax.legend(handles=handles, loc="upper left")
    fig.savefig(here / filename, bbox_inches="tight", facecolor="white")
    plt.show()


if __name__ == "__main__":
    pos, kind = load_nodes()
    plot(pos, kind,
         load_edges(["proposed_electricityDC.csv"], pos),
         load_edges(["electricityAC.csv", "electricityDC.csv"], pos),
         "Proposed electricity connections", "proposed_connections.svg")
    plot(pos, kind,
         load_edges(["CO2_Pipeline.csv"], pos),
         [],
         "Proposed CO2 pipelines", "proposed_co2_pipelines.svg")
