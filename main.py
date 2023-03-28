# TODO: Include hplib
# TODO: Implement option for complete linearization
# TODO: Implement length of time step
# TODO: Implement all technologies
# TODO: Complete ERA5 weather import
from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import numpy as np

data = dm.load_object(r'./test/test_data/k_means.p')
nr_days_cluster = 40
clustered_data = dm.ClusteredDataHandle(data, nr_days_cluster)


# Save Data File to file
data_save_path = r'.\user_data\data_handle_test'
#
# # TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-01 01:00', resolution=1)
topology.define_carriers(['electricity'])
topology.define_nodes(['onshore'])

configuration = ModelConfiguration()


# distance = dm.create_empty_network_matrix(topology.nodes)
# distance.at['onshore', 'offshore'] = 100
# distance.at['offshore', 'onshore'] = 100
#
# connection = dm.create_empty_network_matrix(topology.nodes)
# connection.at['onshore', 'offshore'] = 1
# connection.at['offshore', 'onshore'] = 1
# topology.define_new_network('electricitySimple', distance=distance, connections=connection)

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# CLIMATE DATA
from_file = 0
if from_file == 1:
    data.read_climate_data_from_file('onshore', r'.\data\climate_data_onshore.txt')
    # data.read_climate_data_from_file('offshore', r'.\data\climate_data_offshore.txt')
else:
    lat = 52
    lon = 5.16
    data.read_climate_data_from_api('onshore', lon, lat,save_path='.\data\climate_data_onshore.txt')
    # lat = 52.2
    # lon = 4.4
    # data.read_climate_data_from_api('offshore', lon, lat,save_path='.\data\climate_data_offshore.txt')

# DEMAND
electricity_demand = np.ones(len(topology.timesteps)) * 10
data.read_demand_data('onshore', 'electricity', electricity_demand)

production_prof = np.ones(len(topology.timesteps)) * 11

data.read_production_profile('onshore', 'electricity', production_prof, 1)
# heat_demand = np.ones(len(topology.timesteps)) * 10
# data.read_demand_data('onshore', 'heat', heat_demand)
# co2 = np.ones(len(topology.timesteps)) * 10000/8760
# data.read_demand_data('onshore', 'CO2', co2)

# IMPORT
# import_car = np.ones(len(topology.timesteps)) * 500
# data.read_import_limit_data('onshore', 'heat', import_car)
# data.read_import_limit_data('onshore', 'gas', import_car)
# data.read_import_limit_data('onshore', 'hydrogen', import_car)
# data.read_import_limit_data('onshore', 'CO2', import_car)

# data.read_export_limit_data('onshore', 'electricity', import_car)
# data.read_export_limit_data('onshore', 'heat', import_car)

# Price
# h2_price = np.ones(len(topology.timesteps)) * 0.02*1000
# data.read_import_price_data('onshore', 'hydrogen', h2_price)
# gas_price = np.ones(len(topology.timesteps)) * 0.05*1000
# data.read_import_price_data('onshore', 'gas', gas_price)
# PRINT DATA
# data.pprint()

# READ TECHNOLOGY AND NETWORK DATA
data.read_technology_data()
data.read_network_data()

# data_clustered = dm.ClusteredDataHandle(data,2)

# # SAVING/LOADING DATA FILE
# data.save(data_save_path)

# # Read data
energyhub = EnergyHub(data, configuration)
energyhub.quick_solve_model()
energyhub.model.pprint()
# energyhub.model.node_blocks['onshore'].tech_blocks_active['DAC_adsorption'].pprint()
results = energyhub.write_results()
results.write_excel(r'.\userData\GT')


# results = energyhub.write_results()
# results.write_excel(r'.\userData\results_two_stage')

# energyhub1 = EnergyHub(data)
# energyhub1.quick_solve_model()

# results = energyhub1.write_results()
# results.write_excel(r'.\userData\results_benchmark')
# Construct equations
# energyhub.construct_model()
# energyhub.construct_balances()

# Solve model
# energyhub.solve_model()
# results = energyhub.write_results()
# results.write_excel(r'.\userData\results')

# # Add technology to model and solve again
# energyhub.add_technology_to_node('onshore', ['WindTurbine_Offshore_11000'])
# energyhub.construct_balances()
# energyhub.solve_model()
#
# # Write results
# results = energyhub.write_results()

# print('done')
# energyhub.model.display()
#
# # Save model
# print('Saving Model...')
# start = time.time()
# energyhub.save_model('./data/ehub_instances', 'test_non_transformed')
# print('Saving Model completed in ' + str(time.time()-start) + ' s')