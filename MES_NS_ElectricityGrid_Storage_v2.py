from cases.NorthSea_helpers.read_input_data import *
import numpy as np

from src.model_configuration import ModelConfiguration
from src.energyhub import EnergyHub
from cases.NorthSea_helpers.utilities import *

# General Settings
settings = Settings()
settings.save_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20230703/MES_NS_ElectricityGrid_Storage'

settings.new_technologies_stage = 'ElectricityStorage'
settings.networks.existing_electricity = 1
settings.networks.new_electricityAC = 1
settings.networks.new_electricityDC = 1
settings.networks.new_hydrogen = 0


# Configuration
configuration = define_configuration()

# Set Data
nodes = read_nodes(settings)
topology = define_topology(settings, nodes)
topology = define_installed_capacities(settings, nodes, topology)
topology = define_networks(settings, topology)
topology = define_new_technologies(settings, nodes, topology)

data = define_data_handle(topology, nodes)
data = define_generic_production(settings, nodes, data)
data = define_hydro_inflow(settings, nodes, data)
data = define_demand(settings, nodes, data)
data = define_imports(settings, nodes, data)
data = define_exports(nodes, data)

# Read data
tec_data_path = settings.data_path + '/Technology_Data/'
write_to_technology_data(tec_data_path, settings)
data.read_technology_data(path =tec_data_path)
data.read_network_data()
data = define_charging_efficiencies(settings, nodes, data)

# Solve
energyhub = EnergyHub(data, configuration)
results = energyhub.quick_solve()
results.write_excel(settings.save_path)