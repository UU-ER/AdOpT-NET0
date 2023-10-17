# TODO: Implement option for complete linearization
# TODO: Implement length of time step
# TODO: Implement all technologies
from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import numpy as np
from pathlib import Path

# Save Data File to file
data_save_path = Path('./user_data/data_handle_test')

# TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-03 23:00', resolution=1)
topology.define_carriers(['electricity'])
topology.define_nodes(['onshore'])
# topology.define_new_technologies('onshore', ['Storage_Battery'])
topology.define_new_technologies('onshore', ['Storage_OceanBattery', 'WindTurbine_Offshore_6000'])


# topology.define_existing_technologies('onshore', {'Storage_Battery': 100})
#
#
# distance = dm.create_empty_network_matrix(topology.nodes)
# distance.at['onshore', 'offshore'] = 100
# distance.at['offshore', 'onshore'] = 100
#
# connection = dm.create_empty_network_matrix(topology.nodes)
# connection.at['onshore', 'offshore'] = 1
# connection.at['offshore', 'onshore'] = 1
# topology.define_new_network('electricitySimple', distance=distance, connections=connection)

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# CLIMATE DATA
from_file = 1
if from_file == 1:
    data.read_climate_data_from_file('onshore', './data/climate_data_onshore.txt')
    # data.read_climate_data_from_file('offshore', './data/climate_data_offshore.txt')
# else:
#     lat = 52
#     lon = 5.16
#     data.read_climate_data_from_api('onshore', lon, lat,save_path='./data/climate_data_onshore.txt')
#     lat = 52.2
#     lon = 4.4
#     data.read_climate_data_from_api('offshore', lon, lat,save_path='./data/climate_data_offshore.txt')

# DEMAND
electricity_demand = np.ones(len(topology.timesteps)) * 1
data.read_demand_data('onshore', 'electricity', electricity_demand)

# production_prof = np.ones(len(topology.timesteps)) * 11

# data.read_production_profile('onshore', 'electricity', production_prof, 1)

# carbontax = np.ones(len(topology.timesteps)) * 11
# carbonsubsidy = np.ones(len(topology.timesteps)) * 11
#
# data.read_carbon_price_data(carbontax, 'tax')
# data.read_carbon_price_data(carbonsubsidy, 'subsidy')
# heat_demand = np.ones(len(topology.timesteps)) * 10
# data.read_demand_data('onshore', 'heat', heat_demand)
# co2 = np.ones(len(topology.timesteps)) * 10000/8760
# data.read_demand_data('onshore', 'CO2', co2)

# IMPORT
# el_import = np.ones(len(topology.timesteps)) * 10000
# data.read_import_limit_data('onshore', 'electricity', el_import)
#
# el_price = np.ones(len(topology.timesteps)) * 1000000
# data.read_import_price_data('onshore', 'electricity', el_price)

# READ TECHNOLOGY AND NETWORK DATA

data.read_technology_data()
data.read_network_data()


# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()

# # Read data
energyhub = EnergyHub(data, configuration)
results = energyhub.quick_solve()

