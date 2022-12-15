# Thanks to: ruoyu0088 (piecewise linear modeling)
# TODO: Include hplib
# TODO: Implement option for complete linearization
# TODO: Implement time index for set_t
# TODO: Implement length of time step
# TODO: Implement design days (retain extremes)
# TODO: Implement Lukas Algorithm
import src.data_management as dm
from pyomo.environ import units as u
import pandas as pd
import numpy as np
from src.energyhub import energyhub
from pyomo.environ import *

# Save Data File to file
data_save_path = r'.\user_data\data_handle_test'
#
# # TOPOLOGY
modeled_year = 2001

topology = {}
topology['timesteps'] = pd.date_range(start=str(modeled_year)+'-01-01 00:00', end=str(modeled_year)+'-12-31 23:00', freq='1h')

topology['timestep_length_h'] = 1
topology['carriers'] = ['electricity']
topology['nodes'] = ['onshore', 'offshore']
topology['technologies'] = {}
topology['technologies']['onshore'] = ['battery', 'PV']
topology['technologies']['offshore'] = []

topology['networks'] = {}
topology['networks']['electricitySimple'] = {}
network_data = dm.create_empty_network_data(topology['nodes'])
network_data['distance'].at['onshore', 'offshore'] = 100
network_data['distance'].at['offshore', 'onshore'] = 100
network_data['connection'].at['onshore', 'offshore'] = 1
network_data['connection'].at['offshore', 'onshore'] = 1
topology['networks']['electricitySimple'] = network_data

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# CLIMATE DATA
from_file = 0
if from_file == 1:
    data.read_climate_data_from_file('onshore', r'.\user_data\climate_data_onshore.txt')
    data.read_climate_data_from_file('offshore', r'.\user_data\climate_data_offshore.txt')
else:
    lat = 52
    lon = 5.16
    data.read_climate_data_from_api('onshore', lon, lat,save_path='.\data\climate_data_onshore.txt')
    lat = 52.2
    lon = 4.4
    data.read_climate_data_from_api('offshore', lon, lat,save_path='.\data\climate_data_offshore.txt')

# DEMAND
electricity_demand = np.ones(len(topology['timesteps'])) * 10
data.read_demand_data('onshore', 'electricity', electricity_demand)

# PRINT DATA
# data.pprint()

# READ TECHNOLOGY AND NETWORK DATA
data.read_technology_data()
data.read_network_data()


# # SAVING/LOADING DATA FILE
# data.save(data_save_path)

# # Read data
energyhub = energyhub(data)

# Construct equations
energyhub.construct_model()
energyhub.construct_balances()

# Solve model
energyhub.solve_model()

# Add technology to model and solve again
energyhub.add_technology_to_node('onshore', ['WT_OS_11000'])
energyhub.construct_balances()
energyhub.solve_model()

# Display whole model
# energyhub.model.pprint()

# Save model
# print('Saving Model...')
# start = time.time()
# energyhub.save_model('./data/ehub_instances', 'test_non_transformed')
# print('Saving Model completed in ' + str(time.time()-start) + ' s')