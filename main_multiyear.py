from pathlib import Path
from warnings import warn
import adopt_net0.data_preprocessing as dp
from adopt_net0.modelhub import ModelHub
from adopt_net0.result_management.read_results import (
    add_values_to_summary,
    add_vintage_annualization_to_summary,
)
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
dp.fill_carrier_data(path, value_or_data=0)
dp.fill_carrier_pressure_data(path, pressure_value_bar=0)

# Build the model with investment intervals
adopthub = {}
intervals = ["Interval_1", "Interval_2", "Interval_n"]
intervals_between_years = [10, 10]

# Check correctness of interval and intervals between years:
# If intervals between years is not defined, no life timecheck, but warning
if intervals_between_years is None:
    warn(
        "intervals_between_years is not defined. No lifetime check will be performed on components."
    )

# If intervals between years is defined, it must be have the right length
else:
    expected_length = len(intervals) - 1
    if (
        not isinstance(intervals_between_years, list)
        or len(intervals_between_years) != expected_length
    ):
        raise ValueError(
            f"intervals_between_years must be a list of length {expected_length} "
            f"(number of intervals - 1), got {intervals_between_years}"
        )

# Construct and solve the model
for i, interval in enumerate(intervals):
    interval_path = Path(casestudy_path) / f"Case_{interval}"

    if i != 0:
        prev_interval = intervals[i - 1]
        installed_capacities_existing(
            adopthub, interval, prev_interval, interval_path, intervals_between_years, i
        )
        del adopthub[prev_interval]  # Free memory — previous interval no longer needed

    adopthub[interval] = ModelHub()
    adopthub[interval].read_data(interval_path)

    # Add interval name as case name
    adopthub[interval].data.model_config["reporting"]["case_name"]["value"] = interval

    adopthub[interval].quick_solve()

# Add values of (part of) the parameters and variables to the summary file
add_values_to_summary(Path("path to summary file"))

# Add annualized capex of carried-over vintages to the summary file
add_vintage_annualization_to_summary(
    Path("path to summary file"), casestudy_path, intervals
)
