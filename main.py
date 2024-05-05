# from src.model_configuration import ModelConfiguration
import src.data_preprocessing as dp
from src.energyhub import EnergyHub
import numpy as np

# Todo: save logging to a different place
# Todo: make sure create template functions dont overwrite stuff
# Todo: make it possible to add technology blocks retrospectively
# Todo: logging

# Todo: Cost data
#
path = "caseStudies/dac/nl"

pyhub = EnergyHub()
pyhub.read_data(path, start_period=0, end_period=10)
pyhub.quick_solve()
# pyhub.construct_balances()
# pyhub.solve()


# dm.create_optimization_templates(path)
# dm.create_input_data_folder_template(path)
# data = dm.DataHandle(path)

# print(data.model_config)

# energyhub = EnergyHub(data)
# energyhub.quick_solve()
