# from adopt_net0.model_configuration import ModelConfiguration
import json
from pathlib import Path
import adopt_net0.data_preprocessing as dp
from adopt_net0.modelhub import ModelHub
from adopt_net0.result_management.read_results import add_values_to_summary
import pandas as pd

# Specify the path to your input data
path = Path("C:/Users/6574114/PycharmProjects/adopt_net0/test_ccpp")

steam_demand = 0
el_demand = 150

# Create template files (comment these lines if already defined)
# dp.create_optimization_templates(path)
# dp.create_montecarlo_template_csv(path)

# Create folder structure (comment these lines if already defined)
dp.create_input_data_folder_template(path)

# Write demand/prices to datafile
input_data = pd.read_excel(path / "Data.xlsx", header=[0])

# HP Steam Demand
d = 0
if steam_demand:
    d = input_data["HP"]

dp.fill_carrier_data(path, d, columns=["Demand"], carriers=["steam_hp"])

# MP Steam Demand
if steam_demand:
    d = input_data["MP"]
dp.fill_carrier_data(path, d, columns=["Demand"], carriers=["steam_mp"])

# Electricity Demand
if el_demand:
    dp.fill_carrier_data(path, el_demand, columns=["Demand"], carriers=["electricity"])

# Electricity Price & Export
dp.fill_carrier_data(
    path, input_data["P2021"], columns=["Export price"], carriers=["electricity"]
)
dp.fill_carrier_data(path, 500, columns=["Export limit"], carriers=["electricity"])

# Gas import
dp.fill_carrier_data(path, 50, columns=["Import price"], carriers=["gas"])
dp.fill_carrier_data(path, 1000, columns=["Import limit"], carriers=["gas"])
# H2 import
dp.fill_carrier_data(path, 40, columns=["Import price"], carriers=["hydrogen"])
dp.fill_carrier_data(path, 1000, columns=["Import limit"], carriers=["hydrogen"])

# CO2 price
co2_price = pd.read_csv(
    path / "period1/node_data/node1/CarbonCost.csv", sep=";", index_col=0
)
co2_price["price"] = 100
co2_price.to_csv(path / "period1/node_data/node1/CarbonCost.csv", sep=";")

# Technologies
with open(path / "period1/node_data/node1/Technologies.json", "r") as f:
    tecs = json.load(f)
tecs["existing"] = {"CombinedCycle_fixed_size": 1}
with open(path / "period1/node_data/node1/Technologies.json", "w") as f:
    json.dump(tecs, f)

# Copy technology and network data into folder (comment these lines if already defined)
dp.copy_technology_data(path)
# dp.copy_network_data(path, "path to network data")
#
# # Read climate data and fill carried data (comment these lines if already defined)
dp.load_climate_data_from_api(path)
# dp.fill_carrier_data(path, value=0)
#
# # Construct and solve the model
pyhub = ModelHub()
pyhub.read_data(path, start_period=0, end_period=2)
pyhub.quick_solve()

#
# # Add values of (part of) the parameters and variables to the summary file
# add_values_to_summary(Path("path to summary file"))
