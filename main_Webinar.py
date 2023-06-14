from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub, load_energyhub_instance
import numpy as np
import src.plotting as pl
import pandas as pd

# TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='12-31 23:00', resolution=1)
topology.define_carriers(['electricity'])
topology.define_nodes(['my_house'])
topology.define_existing_technologies('my_house', {'Photovoltaic': 0.01})
topology.define_new_technologies('my_house', ['Storage_Battery'])

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# CLIMATE DATA
lat = 52
lon = 5.16
data.read_climate_data_from_api('my_house', lon, lat)

# DEMAND
electricity_demand = pd.read_excel('./cases/Webinar/HouseholdDemand.xlsx')
data.read_demand_data('my_house', 'electricity', list(electricity_demand['household_demand']))

# IMPORT
electricity_import = np.ones(len(topology.timesteps)) * 1
data.read_import_limit_data('my_house', 'electricity', electricity_import)

electricity_price = np.ones(len(topology.timesteps)) * 300
data.read_import_price_data('my_house', 'electricity', electricity_price)

# EXPORT
data.read_export_limit_data('my_house', 'electricity', electricity_import)
electricity_price = np.ones(len(topology.timesteps)) * 50
data.read_export_price_data('my_house', 'electricity', electricity_price)

# READ TECHNOLOGY AND NETWORK DATA

data.read_technology_data(path='./cases/Webinar/')
data.read_network_data()

# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()

# # Read data
energyhub = EnergyHub(data, configuration)
results = energyhub.quick_solve()

results.write_excel('./user_data/Webinar_Battery.xlsx')