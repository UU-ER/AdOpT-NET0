# TODO: Implement option for complete linearization
# TODO: Implement length of time step
# TODO: Implement all technologies
import pandas as pd
import numpy as np
from pyomo.environ import *
from pathlib import Path

from src.model_construction.construct_balances import add_system_costs
from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub



# Save Data File to file
data_save_path = Path('./user_data/data_handle_test')

# TOPOLOGY
topology = dm.SystemTopology()
topology.define_time_horizon(year=2001,start_date='05-01 00:00', end_date='05-31 23:00', resolution=1)
topology.define_carriers(['electricity'])
topology.define_nodes(['offshore'])
topology.define_new_technologies('offshore', ['Storage_OceanBattery_specific_3_2bp'])

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

SD_list = np.arange(3, 16, 1)
CAPEX_list = np.arange(0.05, 1.05, 0.05)
# SD_list = np.arange(1, 2, 1)
# CAPEX_list = np.arange(0.1, 0.3, 0.1)


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

# Electricity Price
loadpath_electricityprice = './data/ob_input_data/day_ahead_2019.csv'
el_import_price = pd.read_csv(loadpath_electricityprice)
el_import_price = el_import_price['Day-ahead Price [EUR/MWh]'][0:8760]
el_import_price = el_import_price.interpolate()
el_import_price = np.array(el_import_price)

data.read_import_price_data('offshore', 'electricity', el_import_price)
data.read_export_price_data('offshore', 'electricity', el_import_price)

# Read data
data.read_technology_data()
data.read_network_data()

# Orignial values
capex_original = data.technology_data['offshore']['Storage_OceanBattery_specific_3_2bp'].economics.capex_data['unit_capex']


# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()
# configuration.reporting.case_name = 'MAY19_SD' + str(round(sd, 2)) + '_CAPEX' + str(round(capex_share, 2))

# # Read data
energyhub = EnergyHub(data, configuration)
results = energyhub.quick_solve()

el_import_price_original = np.array(data.node_data['offshore'].data['import_prices']['electricity'])
capex_original = energyhub.model.node_blocks['offshore'].tech_blocks_active['Storage_OceanBattery_specific_3_2bp'].para_unit_capex_reservoir_annual.value

for sd in SD_list:
    for capex_share in CAPEX_list:

        # Change the electricity prices
        mean_price = np.mean(el_import_price_original.mean())
        std_deviation = np.std(el_import_price_original)
        new_std_deviation = std_deviation * sd
        new_mean = 60
        el_import_price = new_mean + (el_import_price_original - mean_price) * (new_std_deviation / std_deviation)

        # Import Prices
        set_t = energyhub.model.set_t_full

        b_node = energyhub.model.node_blocks['offshore']
        # b_node.para_import_price.pprint()
        for t in set_t:
            b_node.para_import_price[t, 'electricity'] = el_import_price[t-1]
            b_node.para_export_price[t, 'electricity'] = el_import_price[t-1]

        # Change the capex
        b_tec = b_node.tech_blocks_active['Storage_OceanBattery_specific_3_2bp']
        # b_tec.para_unit_capex_reservoir_annual.pprint()
        b_tec.para_unit_capex_reservoir_annual = capex_original * capex_share

        energyhub.solver.remove_constraint(b_tec.const_capex_aux)
        b_tec.del_component(b_tec.const_capex_aux)
        b_tec.const_capex_aux = Constraint(expr=b_tec.para_unit_capex_reservoir_annual * b_tec.var_size +
                                                sum(b_tec.var_capex_turbine[turbine] for
                                                                     turbine in b_tec.set_turbine_slots) +
                                                sum(b_tec.var_capex_pump[pump] for pump in b_tec.set_pump_slots) ==
                                                b_tec.var_capex_aux)


        energyhub.configuration.reporting.case_name = 'MAY19_2bp_SD'+ str(round(sd,2)) + '_CAPEX' + str(round(capex_share,2))
        energyhub.model = add_system_costs(energyhub)
        energyhub.solve()
