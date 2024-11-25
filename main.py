import adopt_net0 as adopt
from pathlib import Path

path = Path("data")

adopt.create_optimization_templates(path)
adopt.create_input_data_folder_template(path)
adopt.load_climate_data_from_api(path)
