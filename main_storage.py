import pandas as pd
from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import numpy as np
from pathlib import Path

# INPUT
factors = {}
factors['demand'] = 0.01
factors['offshore'] = 0.5
factors['res_to_demand'] = 1


gas_price = 43.92 # ERAA
co2_price = 110 # ERAA

time_series = pd.read_csv(Path('./cases/storage/clean_data/time_series.csv'))

# TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-01 23:00', resolution=1)
topology.define_carriers(['electricity', 'gas', 'hydrogen'])
topology.define_nodes(['offshore', 'onshore'])
topology.define_existing_technologies('onshore', {'PowerPlant_Gas': max(time_series['demand'] * factors['demand']) * 1.5})

# topology.define_new_technologies('offshore', ['Storage_OceanBattery_general'])
# topology.define_new_technologies('onshore', ['Storage_Battery'])


factors['onshore'] = 1 - factors['offshore']

annual_demand = sum(time_series['demand']) * factors['demand']
onshore_wind_to_onshore_RES_ratio = 100661 / (100661 + 194522)
onshore_pv_to_onshore_RES_ratio = 1 - onshore_wind_to_onshore_RES_ratio
production_fraction_wind_onshore = factors['onshore'] * onshore_wind_to_onshore_RES_ratio
production_fraction_pv_onshore = factors['onshore'] * onshore_pv_to_onshore_RES_ratio
capacity_wind_offshore = factors['res_to_demand'] * annual_demand * factors['offshore'] / sum(time_series['wind_offshore'])
capacity_wind_onshore = factors['res_to_demand'] * annual_demand * production_fraction_wind_onshore / sum(time_series['wind_onshore'])
capacity_pv_onshore = factors['res_to_demand'] * annual_demand * production_fraction_pv_onshore / sum(time_series['PV'])

distance = dm.create_empty_network_matrix(topology.nodes)
distance.at['onshore', 'offshore'] = 100
distance.at['offshore', 'onshore'] = 100

size = dm.create_empty_network_matrix(topology.nodes)
size.at['onshore', 'offshore'] = capacity_wind_offshore
size.at['offshore', 'onshore'] = capacity_wind_offshore
topology.define_existing_network('electricityDC', distance=distance, size=size)

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# CLIMATE DATA
from_file = 1
if from_file == 1:
    data.read_climate_data_from_file('onshore', './data/climate_data_onshore.txt')
    data.read_climate_data_from_file('offshore', './data/climate_data_offshore.txt')
# else:
#     lat = 52
#     lon = 5.16
#     data.read_climate_data_from_api('offshore', lon, lat,save_path='./data/climate_data_offshore.txt')
#     lat = 52.2
#     lon = 4.4
#     data.read_climate_data_from_api('offshore', lon, lat,save_path='./data/climate_data_offshore.txt')

# DEMAND
data.read_demand_data('onshore', 'electricity', (time_series['demand'] * factors['demand']).to_list())

# PRODUCTION
data.read_production_profile('offshore', 'electricity', (time_series['wind_offshore'] * capacity_wind_offshore).to_list(), 1)
data.read_production_profile('onshore', 'electricity', (time_series['PV'] * capacity_pv_onshore).to_list(), 1)
data.read_production_profile('onshore', 'electricity', (time_series['wind_onshore'] * capacity_wind_onshore).to_list(), 1)

# GAS IMPORT
data.read_import_limit_data('onshore', 'gas', np.ones(len(topology.timesteps)) * max(time_series['demand'] * factors['demand']) * 2)
data.read_import_price_data('onshore', 'gas', np.ones(len(topology.timesteps)) * gas_price)

# CO2 price
data.read_carbon_price_data(np.ones(len(topology.timesteps)) * co2_price, 'tax')

# READ TECHNOLOGY AND NETWORK DATA
data.read_technology_data('./cases/storage/technology_data/')
data.read_network_data('./cases/storage/network_data/')

# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()

# # Read data
energyhub = EnergyHub(data, configuration)
results = energyhub.quick_solve()
#
# for tec in data.technology_data['offshore']:
#     size = data.technology_data['offshore'][tec].model_block.report_results()
#     print(size)
