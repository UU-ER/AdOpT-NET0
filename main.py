# TODO: Implement option for complete linearization
# TODO: Implement length of time step
# TODO: Implement all technologies
from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import numpy as np
from pathlib import Path

import pytest
import numpy as np
from pyomo.environ import *
import pandas as pd

from src.model_configuration import ModelConfiguration
from src.components.utilities import annualize
from src.data_management import *
from src.energyhub import EnergyHub as ehub


topology = SystemTopology()
topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='12-03 23:00', resolution=1)
topology.define_carriers(['electricity', 'gas', 'hydrogen'])
topology.define_nodes(['test_node1'])
topology.define_new_technologies('test_node1', ['Storage_Battery', 'Photovoltaic', 'TestWindTurbine_Onshore_1500', 'GasTurbine_simple'])

# Initialize instance of DataHandle
data = DataHandle(topology)

# NETWORKS
# distance = create_empty_network_matrix(topology.nodes)
# distance.at['test_node1', 'test_node2'] = 1
# distance.at['test_node2', 'test_node1'] = 1
# connection = create_empty_network_matrix(topology.nodes)
# connection.at['test_node1', 'test_node2'] = 1
# connection.at['test_node2', 'test_node1'] = 1
# topology.define_new_network('electricityTest', distance=distance, connections=connection)

# CLIMATE DATA
climate_data_path = './src/test/climate_data_test.p'
data.read_climate_data_from_file('test_node1', climate_data_path)
data.read_climate_data_from_file('test_node2', climate_data_path)

# DEMAND
electricity_demand = np.ones(len(topology.timesteps)) * 10
data.read_demand_data('test_node1', 'electricity', electricity_demand)

# IMPORT
gas_import = np.ones(len(topology.timesteps)) * 200
data.read_import_limit_data('test_node1', 'gas', gas_import)

gas_price = np.ones(len(topology.timesteps)) * 500
data.read_import_price_data('test_node1', 'gas', gas_price)

# IMPORT
el_import = np.ones(len(topology.timesteps)) * 200
data.read_import_limit_data('test_node1', 'electricity', el_import)

el_price = np.ones(len(topology.timesteps)) * 1000
data.read_import_price_data('test_node1', 'electricity', el_price)


# READ TECHNOLOGY AND NETWORK DATA
data.read_technology_data()
data.read_network_data()

# Full resolution
configuration = ModelConfiguration()
energyhub1 = ehub(data, configuration)
energyhub1.model_information.testing = 1
energyhub1.quick_solve()
cost1 = energyhub1.model.var_total_cost.value
assert energyhub1.solution.solver.termination_condition == 'optimal'

# Typical days Method 2 (standard)
configuration = ModelConfiguration()
configuration.optimization.typicaldays.N = 40
energyhub2 = ehub(data, configuration)
energyhub2.model_information.testing = 1
energyhub2.quick_solve()
cost2 = energyhub2.model.var_total_cost.value
assert energyhub2.solution.solver.termination_condition == 'optimal'
assert abs(cost1 - cost2) / cost1 <= 0.2
print(abs(cost1 - cost2) / cost1)
print(cost1)
print(cost2)



#
# # Save Data File to file
# data_save_path = Path('./user_data/data_handle_test')
#
# # TOPOLOGY
# topology = dm.SystemTopology()
# topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-03 23:00', resolution=1)
# topology.define_carriers(['electricity', 'gas', 'CO2', 'heat'])
# topology.define_nodes(['onshore', 'offshore'])
# # topology.define_new_technologies('onshore', ['Storage_Battery'])
# topology.define_new_technologies('onshore', ['Storage_Battery', 'WindTurbine_Offshore_6000'])
#
# # topology.define_existing_technologies('onshore', {'Storage_Battery': 100})
#
#
# distance = dm.create_empty_network_matrix(topology.nodes)
# distance.at['onshore', 'offshore'] = 100
# distance.at['offshore', 'onshore'] = 100
#
# connection = dm.create_empty_network_matrix(topology.nodes)
# connection.at['onshore', 'offshore'] = 1
# connection.at['offshore', 'onshore'] = 1
# topology.define_new_network('electricitySimple', distance=distance, connections=connection)
#
# # Initialize instance of DataHandle
# data = dm.DataHandle(topology)
#
# # CLIMATE DATA
# from_file = 1
# if from_file == 1:
#     data.read_climate_data_from_file('onshore', './data/climate_data_onshore.txt')
#     data.read_climate_data_from_file('offshore', './data/climate_data_offshore.txt')
# else:
#     lat = 52
#     lon = 5.16
#     data.read_climate_data_from_api('onshore', lon, lat,save_path='./data/climate_data_onshore.txt')
#     lat = 52.2
#     lon = 4.4
#     data.read_climate_data_from_api('offshore', lon, lat,save_path='./data/climate_data_offshore.txt')
#
# # DEMAND
# electricity_demand = np.ones(len(topology.timesteps)) * 1
# data.read_demand_data('onshore', 'electricity', electricity_demand)
#
# # production_prof = np.ones(len(topology.timesteps)) * 11
#
# # data.read_production_profile('onshore', 'electricity', production_prof, 1)
#
# carbontax = np.ones(len(topology.timesteps)) * 11
# carbonsubsidy = np.ones(len(topology.timesteps)) * 11
#
# data.read_carbon_price_data(carbontax, 'tax')
# data.read_carbon_price_data(carbonsubsidy, 'subsidy')
# # heat_demand = np.ones(len(topology.timesteps)) * 10
# # data.read_demand_data('onshore', 'heat', heat_demand)
# # co2 = np.ones(len(topology.timesteps)) * 10000/8760
# # data.read_demand_data('onshore', 'CO2', co2)
#
# # IMPORT
# el_import = np.ones(len(topology.timesteps)) * 10000
# data.read_import_limit_data('onshore', 'electricity', el_import)
#
# el_price = np.ones(len(topology.timesteps)) * 1000000
# data.read_import_price_data('onshore', 'electricity', el_price)
#
# # READ TECHNOLOGY AND NETWORK DATA
#
# data.read_technology_data()
# data.read_network_data()
#
#
# # SAVING/LOADING DATA FILE
# configuration = ModelConfiguration()
#
# # # Read data
# energyhub = EnergyHub(data, configuration)
# results = energyhub.quick_solve()
# #
# # for tec in data.technology_data['offshore']:
# #     size = data.technology_data['offshore'][tec].model_block.report_results()
# #     print(size)
# #
# #
# results.write_excel('test')