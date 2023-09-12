# TODO: Implement option for complete linearization
# TODO: Implement length of time step
# TODO: Implement all technologies
from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import numpy as np

# Save Data File to file
data_save_path = r'.\user_data\data_handle_test'

# TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-01 23:00', resolution=1)
topology.define_carriers(['electricity'])
topology.define_nodes(['onshore'])

topology.define_new_technologies('onshore', ['Storage_OceanBattery'])

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# CLIMATE DATA
from_file = 1
if from_file == 1:
    data.read_climate_data_from_file('onshore', r'.\data\climate_data_onshore.txt')
else:
    lat = 52
    lon = 5.16
    data.read_climate_data_from_api('onshore', lon, lat,save_path='.\data\climate_data_onshore.txt')

# DEMAND
electricity_demand = np.ones(len(topology.timesteps)) * 1
data.read_demand_data('onshore', 'electricity', electricity_demand)

production_prof = np.ones(len(topology.timesteps)) * 0.8
production_prof[3] = 20

data.read_production_profile('onshore', 'electricity', production_prof, 1)


data.read_technology_data()
data.read_network_data()


# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()

# # Read data
energyhub = EnergyHub(data, configuration)
results = energyhub.quick_solve()
# results.write_excel(r'userData/test')
#

# Get results from OceanBattery
def print_var(var):
    name = var.local_name
    value = round(var.value, 10)
    print(name, ': ', value)

ob = energyhub.model.node_blocks['onshore'].tech_blocks_active['Storage_OceanBattery']
print('--- Reservoir ---')
print_var(ob.var_size)
print('--- Pumps ---')
pb = ob.pump_block
for p in pb:
    print_var(pb[p].var_size)
print('--- Turbines ---')
tb = ob.turbine_block
for t in tb:
    print_var(tb[t].var_size)


