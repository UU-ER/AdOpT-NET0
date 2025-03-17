# from adopt_net0.model_configuration import ModelConfiguration
import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np


# Specify the path to your input data
path = Path("./testCCS")

# Create template files (comment these lines if already defined)
adopt.create_optimization_templates(path)

# Load json template
with open(path / "Topology.json", "r") as json_file:
    topology = json.load(json_file)
# Nodes
topology["nodes"] = ["industrial_cluster"]
# Carriers:
topology["carriers"] = [
    "electricity",
    "CO2captured",
    "heat",
    "hydrogen",
    "gas",
    "clinker",
    "hydrogen",
]
# Investment periods:
topology["investment_periods"] = ["period1"]
# Save json template
with open(path / "Topology.json", "w") as json_file:
    json.dump(topology, json_file, indent=4)

end_period = 10 * 180


# Load json template
with open(path / "ConfigModel.json", "r") as json_file:
    configuration = json.load(json_file)
# Change objective
configuration["optimization"]["objective"]["value"] = "costs"
# Set MILP gap
configuration["solveroptions"]["mipgap"]["value"] = 0.02
# Save json template
with open(path / "ConfigModel.json", "w") as json_file:
    json.dump(configuration, json_file, indent=4)

adopt.create_input_data_folder_template(path)

node_location = pd.read_csv(path / "NodeLocations.csv", sep=";", index_col=0, header=0)
node_location.at["industrial_cluster", "lon"] = 10
node_location.at["industrial_cluster", "lat"] = 10
node_location.at["industrial_cluster", "alt"] = 10
node_location = node_location.reset_index()
node_location.to_csv(path / "NodeLocations.csv", sep=";", index=False)

with open(
    path / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "r"
) as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["CementHybridCCS"]

with open(
    path / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "w"
) as json_file:
    json.dump(technologies, json_file, indent=4)

# Copy over technology files
adopt.copy_technology_data(path)


# Set import limits/cost
adopt.fill_carrier_data(
    path,
    value_or_data=5000,
    columns=["Import limit"],
    carriers=["electricity"],
    nodes=["industrial_cluster"],
)
adopt.fill_carrier_data(
    path,
    value_or_data=20000,
    columns=["Import limit"],
    carriers=["heat"],
    nodes=["industrial_cluster"],
)
adopt.fill_carrier_data(
    path,
    value_or_data=80,
    columns=["Import price"],
    carriers=["electricity"],
    nodes=["industrial_cluster"],
)
adopt.fill_carrier_data(
    path,
    value_or_data=30,
    columns=["Import price"],
    carriers=["heat"],
    nodes=["industrial_cluster"],
)
adopt.fill_carrier_data(
    path,
    value_or_data=1 / 3 * 2 * 0,
    columns=["Demand"],
    carriers=["cement"],
    nodes=["industrial_cluster"],
)
adopt.fill_carrier_data(
    path,
    value_or_data=1 / 3 * 2 * 0,
    columns=["Demand"],
    carriers=["electricity"],
    nodes=["industrial_cluster"],
)


carbon_price = np.linspace(70, 170, 8760)
carbon_cost_path = (
    path / "period1" / "node_data" / "industrial_cluster" / "CarbonCost.csv"
)
carbon_cost_template = pd.read_csv(carbon_cost_path, sep=";", index_col=0, header=0)
carbon_cost_template["price"] = carbon_price
carbon_cost_template = carbon_cost_template.reset_index()
carbon_cost_template.to_csv(carbon_cost_path, sep=";", index=False)

# Construct and solve the model
m = adopt.ModelHub()
m.read_data(path, start_period=0, end_period=end_period)
m.construct_model()
m.construct_balances()
m.solve()
