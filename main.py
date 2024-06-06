# from src.model_configuration import ModelConfiguration
from pathlib import Path
import src.data_preprocessing as dp
from src.energyhub import EnergyHub
from src.result_management.read_results import add_values_to_summary

# Specify the path to your input data
path = "specify path to input data"

# Create template files (comment these lines if already defined)
dp.create_optimization_templates(path)
dp.create_montecarlo_template_csv(path)

# Create folder structure (comment these lines if already defined)
dp.create_input_data_folder_template(path)

# Copy technology and network data into folder (comment these lines if already defined)
dp.copy_technology_data(path, "path to tec data")
dp.copy_network_data(path, "path to network data")

# Read climate data and fill carried data (comment these lines if already defined)
dp.load_climate_data_from_api(path)
dp.fill_carrier_data(path, value=0)

# Construct and solve the model
pyhub = EnergyHub()
pyhub.read_data(path)
pyhub.quick_solve()

# Add values of (part of) the parameters and variables to the summary file
add_values_to_summary(Path("path to summary file"))
