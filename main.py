# TODO: Implement option for complete linearization
# TODO: Implement length of time step
# TODO: Implement all technologies
from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import numpy as np

# Save Data File to file
data_save_path = r'.\user_data\data_handle_test'
#
# # TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-31 23:00', resolution=1)
topology.define_carriers(['electricity', 'gas', 'hydrogen'])
topology.define_nodes(['onshore', 'offshore'])
topology.define_new_technologies('onshore', ['Storage_Battery'])
topology.define_new_technologies('offshore', ['Photovoltaic', 'WindTurbine_Onshore_1500'])
# topology.define_carriers(['electricity'])

# configuration.optimization.timestaging = 1


distance = dm.create_empty_network_matrix(topology.nodes)
distance.at['onshore', 'offshore'] = 100
distance.at['offshore', 'onshore'] = 100

connection = dm.create_empty_network_matrix(topology.nodes)
connection.at['onshore', 'offshore'] = 1
connection.at['offshore', 'onshore'] = 1
topology.define_new_network('electricitySimple', distance=distance, connections=connection)

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# CLIMATE DATA
from_file = 1
if from_file == 1:
    data.read_climate_data_from_file('onshore', r'.\data\climate_data_onshore.txt')
    data.read_climate_data_from_file('offshore', r'.\data\climate_data_offshore.txt')
else:
    lat = 52
    lon = 5.16
    data.read_climate_data_from_api('onshore', lon, lat,save_path='.\data\climate_data_onshore.txt')
    lat = 52.2
    lon = 4.4
    data.read_climate_data_from_api('offshore', lon, lat,save_path='.\data\climate_data_offshore.txt')

# DEMAND
electricity_demand = np.ones(len(topology.timesteps)) * 1
data.read_demand_data('onshore', 'electricity', electricity_demand)

# IMPORT
# gas_import = np.ones(len(topology.timesteps)) * 100
# data.read_import_limit_data('offshore', 'gas', gas_import)
#
# gas_price = np.ones(len(topology.timesteps)) * 1000
# data.read_import_price_data('onshore', 'gas', gas_price)

# READ TECHNOLOGY AND NETWORK DATA

data.read_technology_data()
data.read_network_data()


# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()
configuration.optimization.typicaldays = 0

# # Read data
energyhub = EnergyHub(data, configuration)
energyhub.quick_solve_model()
results = energyhub.write_results()
results.write_excel(r'.\userData\test_full')

configuration = ModelConfiguration()
configuration.optimization.timestaging = 4
# # Read data
energyhub = EnergyHub(data, configuration)
energyhub.quick_solve_model()
results = energyhub.write_results()
results.write_excel(r'.\userData\test_reduced')