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
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-03 01:00', resolution=1)
topology.define_carriers(['electricity'])
topology.define_nodes(['onshore'])
# topology.define_new_technologies('onshore', ['Storage_Battery'])
topology.define_new_technologies('onshore', ['Storage_OceanBattery_specific'])

# topology.define_existing_technologies('onshore', {'Storage_Battery': 100})
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


# PRODUCTION
electricity_production = np.ones(len(topology.timesteps)) * 1000
data.read_production_profile('onshore', 'electricity', electricity_production, 1)

# DEMAND
electricity_demand_low = 800
electricity_demand_high = 1000
electricity_demand = np.zeros(len(topology.timesteps))
for i in range(len(topology.timesteps)):
    if i % 2 == 0:
        electricity_demand[i] = electricity_demand_low
    else:
        electricity_demand[i] = electricity_demand_high
data.read_demand_data('onshore', 'electricity', electricity_demand)

# IMPORT
# el_import = np.ones(len(topology.timesteps)) * 100
# data.read_import_limit_data('onshore', 'electricity', el_import)
#
# el_import_price_low = 600
# el_import_price_high = 900
# el_import_price = np.zeros(len(topology.timesteps))
# for i in range(len(topology.timesteps)):
#     if i % 2 == 0:
#         el_import_price[i] = el_import_price_low
#     else:
#         el_import_price[i] = el_import_price_high
# # el_import_price = np.ones(len(topology.timesteps)) * 1000
# data.read_import_price_data('onshore', 'electricity', el_import_price)

# EXPORT
el_export = np.ones(len(topology.timesteps)) * 100
data.read_export_limit_data('onshore', 'electricity', el_export)

el_export_price = np.ones(len(topology.timesteps)) * 100
data.read_export_price_data('onshore', 'electricity', el_export_price)
# el_export_price_low = 500
# el_export_price_high = 1000



# carbontax = np.ones(len(topology.timesteps)) * 11
# carbonsubsidy = np.ones(len(topology.timesteps)) * 11
#
# data.read_carbon_price_data(carbontax, 'tax')
# data.read_carbon_price_data(carbonsubsidy, 'subsidy')
# heat_demand = np.ones(len(topology.timesteps)) * 10
# data.read_demand_data('onshore', 'heat', heat_demand)
# co2 = np.ones(len(topology.timesteps)) * 10000/8760
# data.read_demand_data('onshore', 'CO2', co2)


# READ TECHNOLOGY AND NETWORK DATA

data.read_technology_data()
data.read_network_data()


# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()

# # Read data
energyhub = EnergyHub(data, configuration)
results = energyhub.quick_solve()
#
# for tec in data.technology_data['offshore']:
#     size = data.technology_data['offshore'][tec].model_block.report_results()
#     print(size)
#
#