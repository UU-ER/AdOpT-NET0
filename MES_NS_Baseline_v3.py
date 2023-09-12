import cases.MES_NorthSea.utilities as dm
import numpy as np
from src.model_configuration import ModelConfiguration
from src.energyhub import EnergyHub

# General Settings
settings = dm.Settings()
settings.save_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20230912/MES_results_test'

settings.new_technologies_stage = None
settings.networks.existing_electricity = 1
settings.networks.new_electricityAC = 0
settings.networks.new_electricityDC = 0
settings.networks.new_hydrogen = 0
#
# Configuration
configuration = dm.define_configuration()

# Set Data
nodes = dm.read_nodes(settings)
topology = dm.define_topology(settings, nodes)
topology = dm.define_installed_capacities(settings, nodes, topology)
topology = dm.define_networks(settings, topology)
# topology = define_new_technologies(settings, nodes, topology)

data = dm.define_data_handle(topology, nodes)
data = dm.define_generic_production(settings, nodes, data)
data = dm.define_hydro_inflow(settings, nodes, data)
data = dm.define_demand(settings, nodes, data)
data = dm.define_imports_exports(settings, nodes, data)

# Read data
tec_data_path = './cases/MES_NorthSea/Technology_Data'
# write_to_technology_data(tec_data_path, settings)
data.read_technology_data(path=tec_data_path)
data.read_network_data()
data = dm.define_charging_efficiencies(settings, nodes, data)

# Solve
energyhub = EnergyHub(data, configuration)
results = energyhub.quick_solve()
results.write_excel(settings.save_path)
