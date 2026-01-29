# from adopt_net0.model_configuration import ModelConfiguration
from pathlib import Path
import adopt_net0.core.data_preprocessing as dp
from adopt_net0.core.modelhub import ModelHub
from adopt_net0.extensions import load_climate_data_from_api

# Specify the path to your input data
path = "specify path to input data"

# Create template files (comment these lines if already defined)
dp.create_optimization_templates(path)

# Create folder structure (comment these lines if already defined)
dp.create_input_data_folder_template(path)

# Copy technology and network data into folder (comment these lines if already defined)
dp.copy_technology_data(path, "path to tec data")
dp.copy_network_data(path, "path to network data")

# Read climate data and fill carried data (comment these lines if already defined)
load_climate_data_from_api(path)
dp.fill_carrier_data(path, value=0)
# dp.fill_carrier_pressure_data(path, value=0)

# Construct and solve the model
pyhub = ModelHub()
pyhub.read_data(path)
pyhub.quick_solve()

