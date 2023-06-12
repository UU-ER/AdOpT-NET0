from cases.NorthSea.preprocessing_simple2 import create_data
from src.energyhub import *
from src.model_configuration import ModelConfiguration
from src.data_management import load_object
from pympler import asizeof


# Save Data File to file
data_save_path = r'.\user_data\CaseStudyNorthSea_input'
#
# Read in climate data and demand data
data = create_data()

# READ TECHNOLOGY AND NETWORK DATA
data.read_technology_data()
data.read_network_data()
data.pprint()

# data.save(data_save_path)

# Read data
configuration = ModelConfiguration()
configuration.optimization.objective = 'pareto'

energyhub = EnergyHub(data, configuration)

# Construct equations
results = energyhub.quick_solve()
results.write_excel(r'.\user_data\CaseStudyNorthSea')
