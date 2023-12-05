import pandas as pd
from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import numpy as np
from pathlib import Path

# INPUT
factors = {}
factors['demand'] = 0.01
factors['pv'] = 2000
factors['wind_offshore'] = 1000

gas_price = 40
co2_price = 60

time_series = pd.read_csv(Path('./cases/storage/clean_data/time_series.csv'))

# TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='12-31 23:00', resolution=1)
topology.define_carriers(['electricity', 'gas', 'hydrogen'])
topology.define_nodes(['offshore', 'onshore'])
topology.define_existing_technologies('onshore', {'GasTurbine_simple': max(time_series['demand'] * factors['demand']) * 1.5})

# topology.define_new_technologies('offshore', ['Storage_OceanBattery_general'])
topology.define_new_technologies('onshore', ['Storage_Battery'])

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
annual_demand = sum(time_series['demand']) * factors['demand']

# PRODUCTION
res_to_demand_ratio = 0.5
production_fraction_wind = 0.5
production_fraction_pv = 1 - production_fraction_wind
capacity_wind = res_to_demand_ratio * annual_demand * production_fraction_wind / sum(time_series['wind'])
capacity_pv = res_to_demand_ratio * annual_demand * production_fraction_pv / sum(time_series['PV'])
data.read_production_profile('offshore', 'electricity', (time_series['wind'] * capacity_wind).to_list(), 1)
data.read_production_profile('onshore', 'electricity', (time_series['PV'] * capacity_pv).to_list(), 1)


# GAS IMPORT
data.read_import_limit_data('onshore', 'gas', np.ones(len(topology.timesteps)) * max(time_series['demand'] * factors['demand']) * 2)
data.read_import_price_data('onshore', 'gas', np.ones(len(topology.timesteps)) * gas_price)

# CO2 price
data.read_carbon_price_data( np.ones(len(topology.timesteps)) * co2_price, 'tax')

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
