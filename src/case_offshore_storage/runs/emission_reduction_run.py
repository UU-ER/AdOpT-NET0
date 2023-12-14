import pandas as pd

from ..input_data import determine_time_series
from ...model_configuration import ModelConfiguration
from ...data_management import DataHandle, SystemTopology, create_empty_network_matrix
from ..energyhub import EnergyhubEmissionOptimization as EnergyHub
import numpy as np
import time

def construct_model(input_data_config, node, technology, cost_limit):

    demand, p_onshore, p_offshore = determine_time_series(input_data_config.f_demand_scaling,
                                                          input_data_config.f_self_sufficiency[0],
                                                          input_data_config.f_offshore_share[0])

    # TOPOLOGY
    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date=input_data_config.start_date, end_date=input_data_config.end_date, resolution=1)
    topology.define_carriers(['electricity', 'gas', 'hydrogen'])
    topology.define_nodes({'offshore': [], 'onshore': []})
    topology.define_existing_technologies('onshore',
                                          {'PowerPlant_Gas': max(demand)})
    topology.define_new_technologies(node, [technology])

    distance = create_empty_network_matrix(topology.nodes)
    distance.at['onshore', 'offshore'] = 100
    distance.at['offshore', 'onshore'] = 100

    size = create_empty_network_matrix(topology.nodes)
    size.at['onshore', 'offshore'] = max(p_offshore)
    size.at['offshore', 'onshore'] = max(p_offshore)
    topology.define_existing_network('electricityDC', distance=distance, size=size)

    # Initialize instance of DataHandle
    data = DataHandle(topology)

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
    data.read_import_price_data('onshore', 'gas', np.ones(len(topology.timesteps)) * input_data_config.price_ng)

    # CO2 price
    data.read_carbon_price_data(np.ones(len(topology.timesteps)) * input_data_config.price_co2, 'tax')

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data('./cases/storage/technology_data/')
    data.read_network_data('./cases/storage/network_data/')

    # SAVING/LOADING DATA FILE
    configuration = ModelConfiguration()

    configuration.scaling = 1
    configuration.scaling_factors.energy_vars = 1e-2
    configuration.scaling_factors.cost_vars = 1e-1
    configuration.scaling_factors.objective = 1e-3

    configuration.solveroptions.nodefiledir = '//ad.geo.uu.nl/Users/StaffUsers/6574114/gurobifiles/'
    configuration.reporting.save_path = input_data_config.save_path + 'EmissionReduction'
    configuration.reporting.case_name = technology
    configuration.solveroptions.mipgap = input_data_config.mipgap

    # # Read data
    energyhub = EnergyHub(data, configuration, (node, technology), cost_limit)
    energyhub.construct_model()
    energyhub.construct_balances()

    return energyhub


def solve_model(energyhub, f_demand, f_offshore, f_self_sufficiency, node, technology, emission_limit):
    # Determine profiles
    demand, p_onshore, p_offshore = determine_time_series(f_demand, f_offshore, f_self_sufficiency)

    # Change model
    energyhub.change_network_size('electricityDC_existing', max(p_offshore))
    energyhub.change_generic_production('offshore', 'electricity', (p_offshore).to_list())
    energyhub.change_generic_production('onshore', 'electricity', (p_onshore).to_list())

    # Solve model
    energyhub.emission_limit = emission_limit
    results = energyhub.solve()
    time.sleep(1)

    if energyhub.solution.solver.termination_condition == 'optimal':
        curtailment_on = sum(p_onshore) - sum(results.energybalance['onshore']['electricity']['Generic_production'])
        curtailment_of = sum(p_offshore) - sum(results.energybalance['offshore']['electricity']['Generic_production'])
        size = energyhub.model.node_blocks[node].tech_blocks_active[technology].var_size.value
        cost = results.summary.loc[0, 'Total_Cost']
        emissions = results.summary.loc[0, 'Net_Emissions']
    else:
        curtailment_on = -1
        curtailment_of = -1
        size = -1
        cost = -1
        emissions = -1
    result_dict = {'Case': 'MaxCapex',
    'Technology': technology,
    'Node': node,
    'Self Sufficiency': f_self_sufficiency,
    'Offshore Share': f_offshore,
    'Cost': cost,
    'Emissions': emissions,
    'Curtailment Onshore': curtailment_on,
    'Curtailment Offshore': curtailment_of,
    'Size': size
    }
    return pd.DataFrame(result_dict, index=[0])
