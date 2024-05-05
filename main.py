# from src.model_configuration import ModelConfiguration
import src.data_preprocessing as dp
from src.energyhub import EnergyHub
import numpy as np

# Todo: save logging to a different place
# Todo: make sure create template functions dont overwrite stuff
# Todo: make it possible to add technology blocks retrospectively
# Todo: logging

end_period = 20
#
path = "caseStudies/dac/nl_low_cop"

pyhub = EnergyHub()
pyhub.read_data(path, start_period=0, end_period=end_period)
pyhub.quick_solve()
pyhub.model["full"].pprint()
#
# path = "caseStudies/dac/nl_high_cop"
#
# pyhub = EnergyHub()
# pyhub.read_data(path, start_period=0, end_period=end_period)
# pyhub.quick_solve()
# # pyhub.construct_balances()
# # pyhub.solve()


# dm.create_optimization_templates(path)
# dm.create_input_data_folder_template(path)
# data = dm.DataHandle(path)

# print(data.model_config)

# energyhub = EnergyHub(data)
# energyhub.quick_solve()
