# TODO: Implement option for complete linearization
# TODO: Implement length of time step
from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import numpy as np

# TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-01 10:00', resolution=1)
topology.define_carriers(['electricity', 'gas', 'hydrogen'])
topology.define_nodes({'onshore': {'lon': 10, 'lat': 15}, 'offshore': {'lon': 20, 'lat': 15}})
topology.define_existing_technologies('onshore', {'PowerPlant_Gas': 2000})
topology.define_new_technologies('offshore', ['Electrolyser_PEM'])
topology.define_new_technologies('onshore', ['FuelCell'])

distance = dm.create_empty_network_matrix(topology.nodes)
distance.at['onshore', 'offshore'] = 100
distance.at['offshore', 'onshore'] = 100

connection = dm.create_empty_network_matrix(topology.nodes)
connection.at['onshore', 'offshore'] = 1
connection.at['offshore', 'onshore'] = 1
topology.define_new_network('hydrogenPipelineOffshore', distance=distance, connections=connection)

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

# Generic production
data.read_production_profile('offshore', 'electricity', np.ones(len(topology.timesteps)) * 1500, 1)

# Import
gas_import = np.ones(len(topology.timesteps)) * 2000
data.read_import_limit_data('onshore', 'gas', gas_import)

# export_lim = np.ones(len(topology.timesteps)) * 10000
# data.read_export_limit_data('onshore', 'hydrogen', export_lim)

data.read_export_emissionfactor_data('onshore', 'hydrogen', np.ones(len(data.topology.timesteps)) * -0.18)
data.read_import_emissionfactor_data('onshore', 'gas', np.ones(len(data.topology.timesteps)) * 0.18)

carbontax = np.ones(len(topology.timesteps)) * 80

data.read_carbon_price_data(carbontax, 'tax')

gas_price = np.ones(len(topology.timesteps)) * 40
data.read_import_price_data('onshore', 'gas', gas_price)
data.read_export_price_data('onshore', 'hydrogen', gas_price* carbontax * 0.18)

# READ TECHNOLOGY AND NETWORK DATA
data.read_technology_data(load_path = './mes_north_sea/clean_data/technology_data')
data.read_network_data(load_path = './mes_north_sea/clean_data/network_data')

# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()
configuration.reporting.save_path = './userData/'
configuration.scaling = 1
configuration.scaling_factors.energy_vars = 1e-2
configuration.scaling_factors.cost_vars = 1

# # Read data
energyhub = EnergyHub(data, configuration)
energyhub.quick_solve()
