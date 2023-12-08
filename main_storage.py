import pandas as pd
from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import numpy as np
from pathlib import Path

# Test?
test = 1

# INPUT
factors = {}
factors['demand'] = 0.05
if test == 1:
    factors['offshore'] = [round(x, 2) for x in list(np.arange(0.1, 0.2, 0.05))]
    factors['self_sufficiency'] = [round(x, 2) for x in list(np.arange(0.1, 0.3, 0.1))]
else:
    factors['offshore'] = [round(x, 2) for x in list(np.arange(0.1, 1.05, 0.05))]
    factors['self_sufficiency'] = [round(x, 2) for x in list(np.arange(0.1, 2.1, 0.1))]


gas_price = 43.92 # ERAA
co2_price = 110 # ERAA

time_series = pd.read_csv(Path('./cases/storage/clean_data/time_series.csv'))

# TOPOLOGY
topology = dm.SystemTopology()
if test == 1:
    topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-01 23:00', resolution=1)
else:
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='12-31 23:00', resolution=1)
topology.define_carriers(['electricity', 'gas', 'hydrogen'])
topology.define_nodes({'offshore':[], 'onshore':[]})
topology.define_existing_technologies('onshore', {'PowerPlant_Gas': max(time_series['demand'] * factors['demand']) * 1.5})
# topology.define_new_technologies('offshore', ['Storage_OceanBattery_general'])
# topology.define_new_technologies('onshore', ['Storage_Battery'])

annual_demand = sum(time_series['demand']) * factors['demand']
s_pv = 194522 / (100661 + 194522)
s_wind = 100661 / (100661 + 194522)

e_offshore = sum(time_series['wind_offshore'])
e_onshore = sum(time_series['wind_onshore']) * s_wind + sum(time_series['PV']) * s_pv

for f_offshore in factors['offshore']:
    for f_self_sufficiency in factors['self_sufficiency']:

        f_onshore = 1 - f_offshore

        # capacity required for 1MWh annual generation onshore/offshore
        c_offshore = 1/e_offshore * annual_demand * f_offshore * f_self_sufficiency
        c_onshore = 1/e_onshore * annual_demand * (1-f_offshore) * f_self_sufficiency

        # generation profiles
        p_offshore = c_offshore * time_series['wind_offshore']
        p_onshore = c_onshore * (time_series['wind_onshore'] * s_wind + time_series['PV'] * s_pv)

        distance = dm.create_empty_network_matrix(topology.nodes)
        distance.at['onshore', 'offshore'] = 100
        distance.at['offshore', 'onshore'] = 100

        size = dm.create_empty_network_matrix(topology.nodes)
        size.at['onshore', 'offshore'] = max(p_offshore)
        size.at['offshore', 'onshore'] = max(p_offshore)
        topology.define_existing_network('electricityDC', distance=distance, size=size)

        # Initialize instance of DataHandle
        data = dm.DataHandle(topology)

        # CLIMATE DATA
        from_file = 1
        if from_file == 1:
            data.read_climate_data_from_file('onshore', './data/climate_data_onshore.txt')
            data.read_climate_data_from_file('offshore', './data/climate_data_offshore.txt')
        # else:
        #     lat = 52
        #     lon = 5.16
        #     data.read_climate_data_from_api('offshore', lon, lat,save_path='./data/climate_data_offshore.txt')
        #     lat = 52.2
        #     lon = 4.4
        #     data.read_climate_data_from_api('offshore', lon, lat,save_path='./data/climate_data_offshore.txt')

        # DEMAND
        data.read_demand_data('onshore', 'electricity', (time_series['demand'] * factors['demand']).to_list())

        # PRODUCTION
        data.read_production_profile('offshore', 'electricity', (p_offshore).to_list(), 1)
        data.read_production_profile('onshore', 'electricity', (p_onshore).to_list(), 1)

        # GAS IMPORT
        data.read_import_limit_data('onshore', 'gas', np.ones(len(topology.timesteps)) * max(time_series['demand'] * factors['demand']) * 2)
        data.read_import_price_data('onshore', 'gas', np.ones(len(topology.timesteps)) * gas_price)

        # CO2 price
        data.read_carbon_price_data(np.ones(len(topology.timesteps)) * co2_price, 'tax')

        # READ TECHNOLOGY AND NETWORK DATA
        data.read_technology_data('./cases/storage/technology_data/')
        data.read_network_data('./cases/storage/network_data/')

        # SAVING/LOADING DATA FILE
        configuration = ModelConfiguration()

        configuration.solveroptions.nodefiledir = '//ad.geo.uu.nl/Users/StaffUsers/6574114/gurobifiles/'

        configuration.reporting.save_path = 'userData'
        configuration.reporting.case_name = 'Baseline_SS' + str(f_self_sufficiency) + 'OS' + str(f_offshore)

        # # Read data
        energyhub = EnergyHub(data, configuration)
        results = energyhub.quick_solve()
        #
        # for tec in data.technology_data['offshore']:
        #     size = data.technology_data['offshore'][tec].model_block.report_results()
        #     print(size)
