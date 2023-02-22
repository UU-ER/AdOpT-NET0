# TODO: Include hplib
# TODO: Implement option for complete linearization
# TODO: Implement length of time step
# TODO: Implement Lukas Algorithm
# TODO: Implement all technologies
# TODO: Complete ERA5 weather import
import src.data_management as dm
from pyomo.environ import units as u
import pandas as pd
import numpy as np
from src.energyhub import EnergyHub
from pyomo.environ import *

# Save Data File to file
data_save_path = r'.\user_data\data_handle_test'
#
# # TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-04 23:00', resolution=1)
topology.define_carriers(['electricity', 'heat'])
topology.define_nodes(['onshore', 'offshore'])
topology.define_new_technologies('onshore', ['battery', 'PV', 'Furnace_NG'])

distance = dm.create_empty_network_matrix(topology.nodes)
distance.at['onshore', 'offshore'] = 100
distance.at['offshore', 'onshore'] = 100

connection = dm.create_empty_network_matrix(topology.nodes)
connection.at['onshore', 'offshore'] = 1
connection.at['offshore', 'onshore'] = 1
topology.define_new_network('electricitySimple', distance=distance, connections=connection)
#
# # Initialize instance of DataHandle
# data = dm.DataHandle(topology)
#
# # CLIMATE DATA
# from_file = 0
# if from_file == 1:
#     data.read_climate_data_from_file('onshore', r'.\data\climate_data_onshore.txt')
#     data.read_climate_data_from_file('offshore', r'.\data\climate_data_offshore.txt')
# else:
#     lat = 52
#     lon = 5.16
#     data.read_climate_data_from_api('onshore', lon, lat,save_path='.\data\climate_data_onshore.txt')
#     lat = 52.2
#     lon = 4.4
#     data.read_climate_data_from_api('offshore', lon, lat,save_path='.\data\climate_data_offshore.txt')
#
# # DEMAND
# electricity_demand = np.ones(len(topology['timesteps'])) * 10
# data.read_demand_data('onshore', 'electricity', electricity_demand)
# heat_demand = np.ones(len(topology['timesteps'])) * 10
# data.read_demand_data('onshore', 'heat', heat_demand)
#
# # IMPORT
# gas_import = np.ones(len(topology['timesteps'])) * 50
# data.read_import_limit_data('onshore', 'gas', gas_import)
#
# # PRINT DATA
# data.pprint()
#
# # READ TECHNOLOGY AND NETWORK DATA
# data.read_technology_data()
# data.read_network_data()
#
#
# # # SAVING/LOADING DATA FILE
# # data.save(data_save_path)
#
# # # Read data
# energyhub = EnergyHub(data)
#
# # Construct equations
# energyhub.construct_model()
# energyhub.construct_balances()
#
# # Solve model
# energyhub.solve_model()
# results = energyhub.write_results()
# results.write_excel(r'.\userData\results')
#
# # # Add technology to model and solve again
# # energyhub.add_technology_to_node('onshore', ['WT_OS_11000'])
# # energyhub.construct_balances()
# # energyhub.solve_model()
# #
# # # Write results
# # results = energyhub.write_results()
#
# print('done')
# energyhub.model.display()

# Save model
# print('Saving Model...')
# start = time.time()
# energyhub.save_model('./data/ehub_instances', 'test_non_transformed')
# print('Saving Model completed in ' + str(time.time()-start) + ' s')