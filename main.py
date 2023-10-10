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
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='12-31 23:00', resolution=1)
topology.define_carriers(['electricity', 'gas', 'hydrogen', 'heat'])
# topology.define_nodes(['onshore'])
topology.define_nodes(['onshore', 'offshore'])
topology.define_new_technologies('onshore', ['Photovoltaic', 'Storage_Battery', 'WindTurbine_Onshore_4000',
                                             'GasTurbine_simple'])
topology.define_new_technologies('offshore', ['WindTurbine_Offshore_6000'])

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
from_file = 0
if from_file == 1:
    data.read_climate_data_from_file('onshore', './data/climate_data_onshore.txt')
    data.read_climate_data_from_file('offshore', './data/climate_data_offshore.txt')
else:
    lat = 52
    lon = 5.16
    data.read_climate_data_from_api('onshore', lon, lat, save_path='./data/climate_data_onshore.txt')
    lat = 52.2
    lon = 4.4
    data.read_climate_data_from_api('offshore', lon, lat, save_path='./data/climate_data_offshore.txt')


# DEMAND
electricity_demand = np.ones(len(topology.timesteps)) * 1000
data.read_demand_data('onshore', 'electricity', electricity_demand)

import_lim = np.ones(len(topology.timesteps)) * 100
data.read_import_limit_data('onshore', 'electricity', import_lim)
gas_import = np.ones(len(topology.timesteps)) * 2000
data.read_import_limit_data('onshore', 'gas', gas_import)

import_lim = np.ones(len(topology.timesteps)) * 10000
data.read_export_limit_data('onshore', 'heat', import_lim)

data.read_import_price_data('onshore', 'electricity', np.ones(len(topology.timesteps)) * 60)
data.read_import_emissionfactor_data('onshore', 'electricity', np.ones(len(data.topology.timesteps)) * 0.1)
# production_prof = np.ones(len(topology.timesteps)) * 1000
# data.read_production_profile('onshore', 'electricity', production_prof, 1)
#
# carbontax = np.ones(len(topology.timesteps)) * 11
# carbonsubsidy = np.ones(len(topology.timesteps)) * 11
#
# data.read_carbon_price_data(carbontax, 'tax')
# data.read_carbon_price_data(carbonsubsidy, 'subsidy')
# heat_demand = np.ones(len(topology.timesteps)) * 10
# data.read_demand_data('onshore', 'heat', heat_demand)
# co2 = np.ones(len(topology.timesteps)) * 10000/8760
# data.read_demand_data('onshore', 'CO2', co2)



gas_price = np.ones(len(topology.timesteps)) * 70
data.read_import_price_data('onshore', 'gas', gas_price)

# READ TECHNOLOGY AND NETWORK DATA

data.read_technology_data()
data.read_network_data()


# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()
configuration.scaling = 0

# # Read data
energyhub = EnergyHub(data, configuration)
energyhub.quick_solve()

# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()
configuration.scaling = 1

# # Read data
energyhub = EnergyHub(data, configuration)
energyhub.quick_solve()


# results.write_excel('./userData/', 'test')