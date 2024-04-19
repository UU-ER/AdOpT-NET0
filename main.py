from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import numpy as np

# TODO: Can we replace default with value in model_config json?

path = "C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/TEST1"

# dm.create_optimization_templates(path)
# dm.create_input_data_folder_template(path)
data = dm.DataHandle(path)

print(data.model_config)

# energyhub = EnergyHub(data)
# energyhub.quick_solve()


pyhub = PyHub()
pyhub.read_data(path)
pyhub.solve()
