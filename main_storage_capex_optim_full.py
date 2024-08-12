import pandas as pd
from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.case_offshore_storage.handle_input_data import DataHandleCapexOptimization
from src.case_offshore_storage.energyhub import EnergyhubCapexOptimization as EnergyHub
import numpy as np
from pathlib import Path

from pyomo.environ import *

def determine_time_series(f_demand, f_offshore, f_self_sufficiency):
    time_series = pd.read_csv(Path('./cases/storage/clean_data/time_series.csv'))
    demand = time_series['demand'] * f_demand
    annual_demand = sum(demand)

    s_pv = 194522 / (100661 + 194522)
    s_wind = 100661 / (100661 + 194522)

    e_offshore = sum(time_series['wind_offshore'])
    e_onshore = sum(time_series['wind_onshore']) * s_wind + sum(time_series['PV']) * s_pv

    # capacity required for 1MWh annual generation onshore/offshore
    c_offshore = 1 / e_offshore * annual_demand * f_offshore * f_self_sufficiency
    c_onshore = 1 / e_onshore * annual_demand * (1 - f_offshore) * f_self_sufficiency

    # generation profiles
    p_offshore = c_offshore * time_series['wind_offshore']
    p_onshore = c_onshore * (time_series['wind_onshore'] * s_wind + time_series['PV'] * s_pv)

    return demand, p_onshore, p_offshore


def construct_model_first_time(f_demand, f_offshore, f_self_sufficiency, test):
    gas_price = 43.92  # ERAA
    co2_price = 110  # ERAA

    demand, p_onshore, p_offshore = determine_time_series(f_demand, f_offshore, f_self_sufficiency)

    # TOPOLOGY
    topology = dm.SystemTopology()
    if test == 1:
        topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='02-01 00:00', resolution=1)
    else:
        topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='12-31 23:00', resolution=1)
    topology.define_carriers(['electricity', 'gas', 'hydrogen'])
    topology.define_nodes({'offshore': [], 'onshore': []})
    topology.define_existing_technologies('onshore',
                                          {'PowerPlant_Gas': max(demand) * 1.5})
    topology.define_new_technologies('onshore', ['Storage_Battery_CapexOptimization'])
    # topology.define_new_technologies('onshore', ['Storage_Battery'])

    distance = dm.create_empty_network_matrix(topology.nodes)
    distance.at['onshore', 'offshore'] = 100
    distance.at['offshore', 'onshore'] = 100

    size = dm.create_empty_network_matrix(topology.nodes)
    size.at['onshore', 'offshore'] = max(p_offshore)
    size.at['offshore', 'onshore'] = max(p_offshore)
    topology.define_existing_network('electricityDC', distance=distance, size=size)

    # Initialize instance of DataHandle
    data = DataHandleCapexOptimization(topology)

    # CLIMATE DATA
    data.read_climate_data_from_file('onshore', './data/climate_data_onshore.txt')
    data.read_climate_data_from_file('offshore', './data/climate_data_offshore.txt')

    # DEMAND
    data.read_demand_data('onshore', 'electricity', demand.to_list())

    # PRODUCTION
    data.read_production_profile('offshore', 'electricity', (p_offshore).to_list(), 1)
    data.read_production_profile('onshore', 'electricity', (p_onshore).to_list(), 1)

    # GAS IMPORT
    data.read_import_limit_data('onshore', 'gas',
                                np.ones(len(topology.timesteps)) * max(demand) * 2)
    data.read_import_price_data('onshore', 'gas', np.ones(len(topology.timesteps)) * gas_price)

    # CO2 price
    data.read_carbon_price_data(np.ones(len(topology.timesteps)) * co2_price, 'tax')

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data('./cases/storage/technology_data/')
    data.read_network_data('./cases/storage/network_data/')

    # SAVING/LOADING DATA FILE
    configuration = ModelConfiguration()
    configuration.solveroptions.mipgap = 0

    configuration.solveroptions.nodefiledir = '//ad.geo.uu.nl/Users/StaffUsers/6574114/gurobifiles/'

    # # Read data
    energyhub = EnergyHub(data, configuration, ('onshore', 'Storage_Battery_CapexOptimization'), 348126316.2)
    energyhub.construct_model()
    energyhub.construct_balances()

    return energyhub

def adapt_model(energyhub, f_demand, f_offshore, f_self_sufficiency):
    demand, p_onshore, p_offshore = determine_time_series(f_demand, f_offshore, f_self_sufficiency)

    # Adapt network size
    network_size = max(p_offshore)
    b_netw = energyhub.model.network_block['electricityDC_existing']
    for arc in b_netw.set_arcs:
        b_arc = b_netw.arc_block[arc]
        b_arc.var_flow.setub(network_size * b_netw.para_rated_capacity)  # Set upper bound

        b_arc.del_component('const_flow_size_high')
        def init_size_const_high(const, t):
            return b_arc.var_flow[t] <= network_size * b_netw.para_rated_capacity
        b_arc.const_flow_size_high = Constraint(energyhub.model.set_t_full, rule=init_size_const_high)

        b_netw.del_component('const_cut_bidirectional')
        b_netw.del_component('const_cut_bidirectional_index')
        def init_cut_bidirectional(const, t, node_from, node_to):
            return b_netw.arc_block[node_from, node_to].var_flow[t] + b_netw.arc_block[node_to, node_from].var_flow[t]\
                   <= network_size
        b_netw.const_cut_bidirectional = Constraint(energyhub.model.set_t_full, b_netw.set_arcs_unique, rule=init_cut_bidirectional)


    # Adapt production profiles (onshore)
    energyhub.data.read_production_profile('offshore', 'electricity', (p_offshore).to_list(), 1)
    energyhub.data.read_production_profile('onshore', 'electricity', (p_onshore).to_list(), 1)

    for nodename in ['onshore', 'offshore']:
        b_node = energyhub.model.node_blocks[nodename]
        b_node.del_component('const_generic_production_index')
        b_node.del_component('const_generic_production')
        b_node.del_component('para_production_profile')
        b_node.del_component('para_production_profile_index')
        node_data = energyhub.data.node_data[nodename]

        def init_production_profile(para, t, car):
            return node_data.data['production_profile'][car][t - 1]
        b_node.para_production_profile = Param(energyhub.model.set_t_full, b_node.set_carriers, rule=init_production_profile, mutable=True)

        def init_generic_production(const, t, car):
            return b_node.para_production_profile[t, car] >= b_node.var_generic_production[t, car]
        b_node.const_generic_production = Constraint(energyhub.model.set_t_full, b_node.set_carriers, rule=init_generic_production)

    return energyhub


# Test?
test = 0

# INPUT
factors = {}
factors['demand'] = 0.05
if test == 1:
    factors['offshore'] = [0.1]
    factors['self_sufficiency'] = [2]
else:
    factors['offshore'] = [round(x, 2) for x in list(np.arange(0.1, 1.05, 0.05))]
    factors['self_sufficiency'] = [round(x, 2) for x in list(np.arange(0.1, 2.1, 0.1))]

idx = 1
for f_offshore in factors['offshore']:
    for f_self_sufficiency in factors['self_sufficiency']:
        if idx == 1:
            energyhub = construct_model_first_time(factors['demand'], f_offshore, f_self_sufficiency, test)
        else:
            energyhub = adapt_model(energyhub, factors['demand'], f_offshore, f_self_sufficiency)

        idx = idx + 1

        energyhub.configuration.reporting.save_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/StorageOffshore'
        energyhub.configuration.reporting.case_name = 'Baseline_SS' + str(f_self_sufficiency) + 'OS' + str(f_offshore)
        energyhub.configuration.reporting.save_summary_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/StorageOffshore'
        energyhub.solve()




