# TODO: Implement option for complete linearization
# TODO: Implement length of time step
# TODO: Implement all technologies
from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub, load_energyhub_instance
import numpy as np
import src.plotting as pl

# Save Data File to file
data_save_path = r'.\user_data\data_handle_test'

# TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-01 23:00', resolution=1)
topology.define_carriers(['electricity','hydrogen'])
topology.define_nodes(['onshore'])
topology.define_new_technologies('onshore', ['Electrolyser_PEM'])

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# CLIMATE DATA
from_file = 1
if from_file == 1:
    data.read_climate_data_from_file('onshore', r'.\data\climate_data_onshore.txt')

# DEMAND
electricity_demand = np.ones(len(topology.timesteps)) * 1
data.read_demand_data('onshore', 'electricity', electricity_demand)

# GENERIC PRODUCTION
gen_production = np.ones(len(topology.timesteps)) * 1.3
data.read_production_profile('onshore', 'electricity', gen_production, curtailment=1)

# EXPORTS
data.read_export_price_data('onshore', 'hydrogen', np.ones(len(data.topology.timesteps)) * (-180))
data.read_export_limit_data('onshore', 'hydrogen', np.ones(len(data.topology.timesteps)) * 10000)
data.read_export_emissionfactor_data('onshore', 'hydrogen', np.ones(len(data.topology.timesteps)) * (-0.183))

# READ TECHNOLOGY AND NETWORK DATA
tec_data_path = r'./cases/NorthSea_v3' + '/Technology_Data/'

data.read_technology_data(tec_data_path)
data.read_network_data()


# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()
configuration.optimization.objective = 'pareto'

# # Read data
energyhub = EnergyHub(data, configuration)
results = energyhub.quick_solve()
results.write_excel(r'./user_data/test')









# results.write_excel(r'userData/test')

# pl.plot_balance_at_node(results.detailed_results[0], 'electricity')

