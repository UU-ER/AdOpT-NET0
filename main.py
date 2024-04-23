from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import numpy as np

# Todo: save logging to a different place
# Todo: make sure create template functions dont overwrite stuff
# Todo: make it possible to add technology blocks retrospectively
# Todo: logging

path = "C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/ExamplaryCaseStudy"
# dm.create_optimization_templates(path)
# dm.create_input_data_folder_template(path)


pyhub = EnergyHub()
pyhub.read_data(path, start_period=0, end_period=3)
pyhub.construct_model()


# dm.create_optimization_templates(path)
# dm.create_input_data_folder_template(path)
# data = dm.DataHandle(path)

# print(data.model_config)

# energyhub = EnergyHub(data)
# energyhub.quick_solve()
