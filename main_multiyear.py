from pathlib import Path
import adopt_net0.data_preprocessing as dp
from adopt_net0.modelhub import ModelHub
from adopt_net0.result_management.read_results import add_values_to_summary
from adopt_net0.utilities import installed_capacities_existing

# Specify the path to your input data
path = "specify path to input data"
casestudy_path = "specify path to case study"

# Create template files (comment these lines if already defined)
dp.create_optimization_templates(path)

# Create folder structure (comment these lines if already defined)
dp.create_input_data_folder_template(path)

# Copy technology and network data into folder (comment these lines if already defined)
dp.copy_technology_data(path, "path to tec data")
dp.copy_network_data(path, "path to network data")
dp.copy_compressor_data(path, "path to compressor data")

# Read climate data and fill carried data (comment these lines if already defined)
dp.load_climate_data_from_api(path)
dp.fill_carrier_data(path, value=0)
dp.fill_carrier_pressure_data(path, value=0)

# Build the model with investment intervals
adopthub = {}
intervals = ["Interval_1", "Interval_2", "Interval_n"]

# Construct and solve the model
for i, interval in enumerate(intervals):
    interval_path = casestudy_path + "/Case_" + interval

    if i != 0:
        prev_interval = intervals[i - 1]
        installed_capacities_existing(adopthub, interval, prev_interval, interval_path)

    adopthub[interval] = ModelHub()
    adopthub[interval].read_data(interval_path)

    # Add interval name as case name
    adopthub[interval].data.model_config["reporting"]["case_name"]["value"] = interval

    adopthub[interval].quick_solve()

# Add values of (part of) the parameters and variables to the summary file
add_values_to_summary(Path("path to summary file"))
