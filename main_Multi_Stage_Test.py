from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub, load_energyhub_instance
import numpy as np
import src.plotting as pl
import pandas as pd

# TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='12-31 23:00', resolution=1)
# topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-01 23:00', resolution=1)
topology.define_carriers(['electricity'])
topology.define_nodes(['node1', 'node2'])
topology.define_existing_technologies('node1', {'Photovoltaic': 1})
topology.define_new_technologies('node1', ['Storage_Battery'])
topology.define_new_technologies('node2', ['WindTurbine_Onshore_4000'])

# Network
distance = dm.create_empty_network_matrix(topology.nodes)
distance.at['node1', 'node2'] = 100
distance.at['node2', 'node1'] = 100

connection = dm.create_empty_network_matrix(topology.nodes)
connection.at['node1', 'node2'] = 1
connection.at['node2', 'node1'] = 1
topology.define_new_network('electricitySimple', distance=distance, connections=connection)

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# CLIMATE DATA
lat = 52
lon = 5.16
data.read_climate_data_from_file('node1', r'.\data\climate_data_onshore.txt')
data.read_climate_data_from_file('node2', r'.\data\climate_data_onshore.txt')

# DEMAND
electricity_demand = pd.read_excel('./cases/Webinar/Demand.xlsx')
data.read_demand_data('node1', 'electricity', list(electricity_demand['household_demand'])*1000)

# READ TECHNOLOGY AND NETWORK DATA

data.read_technology_data()
data.read_network_data()

# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()

# # Read data
energyhub = EnergyHub(data, configuration)
energyhub.quick_solve()

energyhub.add_technology_to_node('node2', ['Photovoltaic'])
results = energyhub.solve()
# results.write_excel('./user_data/Test_MultiStage.xlsx')