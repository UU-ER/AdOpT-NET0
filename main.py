from pathlib import Path

from milp_algorithms.preprocessing.utilities import *

import adopt_net0.data_preprocessing as dp
from adopt_net0.modelhub import ModelHub
from adopt_net0.result_management.read_results import add_values_to_summary
import adopt_net0 as adopt

project_path = Path("./milp_algorithms")
data_path = Path(project_path / "data")

# Set topology/config
dp.create_optimization_templates(data_path)

define_topology(data_path)
define_model_config(data_path)

dp.create_input_data_folder_template(data_path)

define_node_locations(data_path)
define_technologies(data_path)
define_networks(data_path)

# Get technology data
write_technology_jsons(project_path / "raw_data" / "technology_data")

# Copy technology and network data
dp.copy_technology_data(data_path, project_path / "raw_data" / "technology_data")
dp.copy_network_data(data_path, project_path / "raw_data" / "network_data")

# Read climate data and demand
dp.load_climate_data_from_api(data_path)
demand_data = adopt.load_network_city_data()
el_demand =  demand_data.iloc[:, 1]

adopt.fill_carrier_data(data_path, value_or_data=el_demand, columns=['Demand'], carriers=['electricity'],
                        nodes=["node1"])
adopt.fill_carrier_data(data_path, value_or_data=10000, columns=['Import limit'], carriers=['gas'],
                        nodes=["node1"])
adopt.fill_carrier_data(data_path, value_or_data=80, columns=['Import price'], carriers=['gas'],
                        nodes=["node1"])
adopt.fill_carrier_data(data_path, value_or_data=10000, columns=['Import limit'], carriers=['uranium'],
                        nodes=["node1"])
adopt.fill_carrier_data(data_path, value_or_data=14, columns=['Import price'], carriers=['uranium'],
                        nodes=["node1"])


# Construct and solve the model
pyhub = ModelHub()
pyhub.read_data(data_path, start_period=0, end_period=80)
pyhub.quick_solve()

