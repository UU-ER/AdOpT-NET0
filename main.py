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
import time
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
topology['carriers'] = ['electricity', 'heat', 'gas']
topology['nodes'] = ['onshore', 'offshore']
topology['technologies'] = {}
topology['technologies']['onshore'] = ['PV', 'Furnace_NG', 'battery']
topology['technologies']['offshore'] = ['WT_OS_11000']

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
heat_demand = np.ones(len(topology['timesteps'])) * 60
electricity_demand = np.ones(len(topology['timesteps'])) * 10

data.read_demand_data('onshore', 'heat', heat_demand)
data.read_demand_data('onshore', 'electricity', electricity_demand)

# PRICE DATA
gas_price = np.ones(len(topology['timesteps'])) * 100
data.read_import_price_data('onshore', 'gas', gas_price)

# IMPORT/EXPORT LIMITS
gas_import = np.ones(len(topology['timesteps'])) * 1000
data.read_import_limit_data('onshore', 'gas', gas_price)

# PRINT DATA
data.pprint()


# READ TECHNOLOGY AND NETWORK DATA
data.read_technology_data()
data.read_network_data()



# # SAVING/LOADING DATA FILE
# data.save(data_save_path)

# Load Data File from file
# data = load_data_handle(data_save_path)
# data.pprint()


# # Read data
print('Reading in data...')
start = time.time()
energyhub = energyhub(data)
print('Reading in data completed in ' + str(time.time()-start) + ' s')
#
# energyhub.print_topology()
#
# # Construct equations
print('Constructing Model...')
start = time.time()
energyhub.construct_model()
print('Constructing Model completed in ' + str(time.time()-start) + ' s')
#
# # energyhub.model.pprint()
# # # Save model
# # print('Saving Model...')
# # start = time.time()
# # energyhub.save_model('./data/ehub_instances', 'test_non_transformed')
# # print('Saving Model completed in ' + str(time.time()-start) + ' s')
# #
# Big-M transformation
print('Performing Big-M transformation...')
start = time.time()
xfrm = TransformationFactory('gdp.bigm')
xfrm.apply_to(energyhub.model)
print('Performing Big-M transformation completed in ' + str(time.time()-start) + ' s')

# #
# # # Save model 2
# # print('Saving Model...')
# # start = time.time()
# # energyhub.save_model('./data/ehub_instances', 'test_transformed')
# # print('Saving Model completed in ' + str(time.time()-start) + ' s')
# # #
# # energyhub.model.node_blocks['onshore'].pprint()
#
#
print('Solving Model...')
start = time.time()
solver = SolverFactory('gurobi')
solution = solver.solve(energyhub.model, tee=True)
solution.write()
# print('Solving Model completed in ' + str(time.time()-start) + ' s')
# # #
# energyhub.write_results('results')

#
# # #
# # # solve = SolverFactory('gurobi_persistent')
# # # solve.set_instance(energyhub.model)
# # # solution = solve.solve(tee=True)
# # # solution.write()
energyhub.model.display()
# # node_data = energyhub.model.node_blocks['onshore']
# # tec_data = node_data.tech_blocks['PV'].var_size.pprint()
