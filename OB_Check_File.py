# TODO: Implement option for complete linearization
# TODO: Implement length of time step
# TODO: Implement all technologies
import pandas as pd

from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import numpy as np
from pathlib import Path



# Save Data File to file
data_save_path = Path('./user_data/data_handle_test')

# TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='06-01 23:00', resolution=1)
topology.define_carriers(['electricity'])
topology.define_nodes(['offshore'])
topology.define_new_technologies('offshore', ['Storage_OceanBattery_general'])

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# CLIMATE DATA
from_file = 1
if from_file == 1:
    data.read_climate_data_from_file('offshore', './data/climate_data_offshore.txt')
    # data.read_climate_data_from_file('offshore', './data/climate_data_offshore.txt')
# else:
#     lat = 52
#     lon = 5.16
#     data.read_climate_data_from_api('offshore', lon, lat,save_path='./data/climate_data_offshore.txt')
#     lat = 52.2
#     lon = 4.4
#     data.read_climate_data_from_api('offshore', lon, lat,save_path='./data/climate_data_offshore.txt')


# PRODUCTION
# electricity_production = np.ones(len(topology.timesteps)) * 1000
# data.read_production_profile('offshore', 'electricity', electricity_production, 1)

# DEMAND
# electricity_demand_low = 800
# electricity_demand_high = 1000
# electricity_demand = np.zeros(len(topology.timesteps))
# for i in range(len(topology.timesteps)):
#     if i % 2 == 0:
#         electricity_demand[i] = electricity_demand_low
#     else:
#         electricity_demand[i] = electricity_demand_high
# data.read_demand_data('offshore', 'electricity', electricity_demand)

# IMPORT
el_import = np.ones(len(topology.timesteps)) * 1000
data.read_import_limit_data('offshore', 'electricity', el_import)
# EXPORT
el_export = np.ones(len(topology.timesteps)) * 1000
data.read_export_limit_data('offshore', 'electricity', el_export)

use_el_profile = 1

if use_el_profile ==0:
    el_import_price_low = 100
    el_import_price_high = 1000
    el_import_price = np.zeros(len(topology.timesteps))
    for i in range(len(topology.timesteps)):
        if i % 2 == 0:
            el_import_price[i] = el_import_price_low
        else:
            el_import_price[i] = el_import_price_high
else:
    loadpath_electricityprice = 'C:/Users/6574114/Documents/Research/EHUB-Py/data/ob_input_data/day_ahead_2019.csv'
    el_import_price = pd.read_csv(loadpath_electricityprice)
    el_import_price = el_import_price['Day-ahead Price [EUR/MWh]'][0:8760]
    el_import_price = el_import_price.interpolate()

    # Rescale electricity price profile
    el_import_price = np.array(el_import_price)
    mean_price = np.mean(el_import_price.mean())
    std_deviation = np.std(el_import_price)
    new_std_deviation = std_deviation * 10
    new_mean = 60
    el_import_price = new_mean + (el_import_price - mean_price) * (new_std_deviation / std_deviation)

data.read_import_price_data('offshore', 'electricity', el_import_price)
data.read_export_price_data('offshore', 'electricity', el_import_price)
# el_export_price_low = 500
# el_export_price_high = 1000



# carbontax = np.ones(len(topology.timesteps)) * 11
# carbonsubsidy = np.ones(len(topology.timesteps)) * 11
#
# data.read_carbon_price_data(carbontax, 'tax')
# data.read_carbon_price_data(carbonsubsidy, 'subsidy')
# heat_demand = np.ones(len(topology.timesteps)) * 10
# data.read_demand_data('offshore', 'heat', heat_demand)
# co2 = np.ones(len(topology.timesteps)) * 10000/8760
# data.read_demand_data('offshore', 'CO2', co2)


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
#
#