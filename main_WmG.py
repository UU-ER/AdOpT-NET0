from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub, load_energyhub_instance
import numpy as np
import src.plotting as pl
import pandas as pd

topology = dm.SystemTopology()

# Time Horizon, energy carriers, nodes
topology.define_time_horizon(year=2009,start_date='01-01 00:00', end_date='12-31 23:00', resolution=1)
topology.define_carriers(['electricity', 'gas'])
topology.define_nodes(['onshore', 'offshore'])

# Our offshore farm
topology.define_new_technologies('offshore', ['WindTurbine_Offshore_11000'])
topology.define_new_technologies('onshore', ['Photovoltaic'])
topology.define_existing_technologies('onshore', {'PowerPlant_Gas': 2000})

# Our electricity connection to shore
connection = dm.create_empty_network_matrix(topology.nodes)
connection.at['onshore', 'offshore'] = 1
connection.at['offshore', 'onshore'] = 1
distance_matrix = dm.create_empty_network_matrix(topology.nodes)
distance_matrix.at['onshore', 'offshore'] = 80
distance_matrix.at['offshore', 'onshore'] = 80

topology.define_new_network('electricityDC_int', connections=connection, distance=distance_matrix)

carbon_price = 100
gas_import_limit = 5000
gas_price = 50

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# Carbon price
data.read_carbon_price_data(np.ones(len(topology.timesteps)) * carbon_price, 'tax')

# Allow for gas import for the gas power plant
data.read_import_limit_data('onshore', 'gas', np.ones(len(topology.timesteps)) * gas_import_limit)

# Specify gas price
data.read_import_price_data('onshore', 'gas', np.ones(len(topology.timesteps)) * gas_price)

# Climate data
data.read_climate_data_from_api('onshore', lon=5.16, lat=52, year=2009, save_path='.\data\climate_data_onshore.txt')
data.read_climate_data_from_api('offshore', lon=4.4, lat=52.2, year=2009, save_path='.\data\climate_data_offshore.txt')

# Electricity demand
electricity_demand = pd.read_excel('./cases/wind_meets_gas/Demand.xlsx')
data.read_demand_data('onshore', 'electricity', list(electricity_demand['demand']))

# Read technology and network data
data.read_technology_data('./cases/wind_meets_gas/Technology_Data/')
data.read_network_data('./cases/wind_meets_gas/Network_Data/')

# Load configuration
configuration = ModelConfiguration()

# Solve the model
energyhub = EnergyHub(data, configuration)
results = energyhub.quick_solve()

print(results.summary)
print(results.technologies)
print(results.networks)





