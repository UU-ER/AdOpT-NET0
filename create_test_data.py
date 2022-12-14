import src.data_management as dm
import pandas as pd
import numpy as np
import sys

# REMOVE THIS
import pytest
import src.data_management as dm
from src.energyhub import energyhub as ehub
from pyomo.environ import units as u
from pyomo.environ import *
import pandas as pd

def create_data_model1():
    """
    Creates dataset for a model with two nodes.
    PV @ node 2
    electricity demand @ node 1
    electricity network in between
    should be infeasible
    """
    data_save_path = './test/test_data/model1.p'
    modeled_year = 2001

    topology = {}
    topology['timesteps'] = pd.date_range(start=str(modeled_year)+'-01-01 00:00', end=str(modeled_year)+'-12-31 23:00', freq='1h')
    topology['timestep_length_h'] = 1
    topology['carriers'] = ['electricity']
    topology['nodes'] = ['test_node1', 'test_node2']
    topology['technologies'] = {}
    topology['technologies']['test_node1'] = []
    topology['technologies']['test_node2'] = ['PV']

    topology['networks'] = {}
    topology['networks']['electricityTest'] = {}
    network_data = dm.create_empty_network_data(topology['nodes'])
    network_data['distance'].at['test_node1', 'test_node2'] = 100
    network_data['distance'].at['test_node2', 'test_node1'] = 100
    network_data['connection'].at['test_node1', 'test_node2'] = 1
    network_data['connection'].at['test_node2', 'test_node1'] = 1
    topology['networks']['electricityTest'] = network_data

    # Initialize instance of DataHandle
    data = dm.DataHandle(topology)

    # CLIMATE DATA
    data.read_climate_data_from_file('test_node1', r'./test/test_data/climate_data_test.p')
    data.read_climate_data_from_file('test_node2', r'./test/test_data/climate_data_test.p')

    # DEMAND
    electricity_demand = np.ones(len(topology['timesteps'])) * 100
    data.read_demand_data('test_node1', 'electricity', electricity_demand)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

def create_data_model2():
    """
    Creates dataset for a model with two nodes.
    PV @ node 2
    electricity demand @ node 1
    electricity network in between
    should be infeasible
    """
    data_save_path = './test/test_data/model2.p'
    modeled_year = 2001

    topology = {}
    topology['timesteps'] = pd.date_range(start=str(modeled_year)+'-01-01 00:00', end=str(modeled_year)+'-12-31 23:00', freq='1h')

    topology['timestep_length_h'] = 1
    topology['carriers'] = ['heat', 'gas']
    topology['nodes'] = ['test_node1']
    topology['technologies'] = {}
    topology['technologies']['test_node1'] = ['Furnace_NG']

    topology['networks'] = {}

    # Initialize instance of DataHandle
    data = dm.DataHandle(topology)

    # CLIMATE DATA
    data.read_climate_data_from_file('test_node1', r'./test/test_data/climate_data_test.p')

    # DEMAND
    heat_demand = np.ones(len(topology['timesteps'])) * 10
    data.read_demand_data('test_node1', 'heat', heat_demand)

    # PRICE DATA
    gas_price = np.ones(len(topology['timesteps'])) * 1
    data.read_import_price_data('test_node1', 'gas', gas_price)

    # IMPORT/EXPORT LIMITS
    gas_import = np.ones(len(topology['timesteps'])) * 100
    data.read_import_limit_data('test_node1', 'gas', gas_import)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

def create_data_technology_type1_PV():
    """
    Creates dataset for test_technology_type1_PV().
    PV @ node 1
    electricity demand @ node 1
    import of electricity at high price
    Size of PV should be around max electricity demand
    """
    data_save_path = './test/test_data/technology_type1_PV.p'
    modeled_year = 2001

    topology = {}
    topology['timesteps'] = pd.date_range(start=str(modeled_year) + '-01-01 00:00',
                                          end=str(modeled_year) + '-12-31 23:00', freq='1h')

    topology['timestep_length_h'] = 1
    topology['carriers'] = ['electricity']
    topology['nodes'] = ['test_node1']
    topology['technologies'] = {}
    topology['technologies']['test_node1'] = ['PV']

    topology['networks'] = {}

    # Initialize instance of DataHandle
    data = dm.DataHandle(topology)

    # CLIMATE DATA
    data.read_climate_data_from_file('test_node1', r'./test/test_data/climate_data_test.p')

    # DEMAND
    demand = np.ones(len(topology['timesteps'])) * 10
    data.read_demand_data('test_node1', 'electricity', demand)

    # PRICE DATA
    price = np.ones(len(topology['timesteps'])) * 10000
    data.read_import_price_data('test_node1', 'electricity', price)

    # IMPORT/EXPORT LIMITS
    import_lim = np.ones(len(topology['timesteps'])) * 10
    data.read_import_limit_data('test_node1', 'electricity', import_lim)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

def create_data_technology_type1_WT():
    """
    Creates dataset for test_technology_type1_PV().
    WT @ node 1
    electricity demand @ node 1
    import of electricity at high price
    Size of PV should be around max electricity demand
    """
    data_save_path = './test/test_data/technology_type1_WT.p'
    modeled_year = 2001

    topology = {}
    topology['timesteps'] = pd.date_range(start=str(modeled_year) + '-01-01 00:00',
                                          end=str(modeled_year) + '-12-31 23:00', freq='1h')

    topology['timestep_length_h'] = 1
    topology['carriers'] = ['electricity']
    topology['nodes'] = ['test_node1']
    topology['technologies'] = {}
    topology['technologies']['test_node1'] = ['WT_1500']

    topology['networks'] = {}

    # Initialize instance of DataHandle
    data = dm.DataHandle(topology)

    # CLIMATE DATA
    data.read_climate_data_from_file('test_node1', r'./test/test_data/climate_data_test.p')

    # DEMAND
    demand = np.ones(len(topology['timesteps'])) * 10
    data.read_demand_data('test_node1', 'electricity', demand)

    # PRICE DATA
    price = np.ones(len(topology['timesteps'])) * 1000
    data.read_import_price_data('test_node1', 'electricity', price)

    # IMPORT/EXPORT LIMITS
    import_lim = np.ones(len(topology['timesteps'])) * 10
    data.read_import_limit_data('test_node1', 'electricity', import_lim)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

def create_data_technology_PWA():
    """
    Creates dataset for test_technology_CONV3_PWA().
    PWA @ node 1
    heat & electricity demand_e @ node 1
    electricity is about half the heat demand
    import of heat, electricity and gas at high price

    """
    data_save_path = './test/test_data/technology_CONV3_PWA.p'
    modeled_year = 2001

    topology = {}
    topology['timesteps'] = pd.date_range(start=str(modeled_year) + '-01-01 00:00',
                                          end=str(modeled_year) + '-01-01 00:00', freq='1h')

    topology['timestep_length_h'] = 1
    topology['carriers'] = ['electricity', 'heat', 'gas', 'hydrogen']
    topology['nodes'] = ['test_node1']
    topology['technologies'] = {}
    topology['technologies']['test_node1'] = ['testPWA']

    topology['networks'] = {}

    # Initialize instance of DataHandle
    data = dm.DataHandle(topology)

    # CLIMATE DATA
    data.read_climate_data_from_file('test_node1', r'./test/test_data/climate_data_test.p')

    # DEMAND
    demand_e = np.ones(len(topology['timesteps'])) * 5
    data.read_demand_data('test_node1', 'electricity', demand_e)
    demand_h = np.ones(len(topology['timesteps'])) * 10
    data.read_demand_data('test_node1', 'heat', demand_h)

    # PRICE DATA
    price = np.ones(len(topology['timesteps'])) * 10000
    data.read_import_price_data('test_node1', 'electricity', price)
    data.read_import_price_data('test_node1', 'gas', price)
    data.read_import_price_data('test_node1', 'heat', price)
    data.read_import_price_data('test_node1', 'hydrogen', price)

    # IMPORT/EXPORT LIMITS
    import_lim = np.ones(len(topology['timesteps'])) * 5
    data.read_import_limit_data('test_node1', 'electricity', import_lim)
    import_lim = np.ones(len(topology['timesteps'])) * 10
    data.read_import_limit_data('test_node1', 'heat', import_lim)
    import_lim = np.ones(len(topology['timesteps'])) * 10
    data.read_import_limit_data('test_node1', 'gas', import_lim)
    import_lim = np.ones(len(topology['timesteps'])) * 1
    data.read_import_limit_data('test_node1', 'hydrogen', import_lim)


    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)


def create_data_network():
    """
    Creates dataset for test_network().
    import electricity @ node 1
    electricity demand @ node 2
    """
    data_save_path = './test/test_data/networks.p'
    modeled_year = 2001

    topology = {}
    topology['timesteps'] = pd.date_range(start=str(modeled_year) + '-01-01 00:00',
                                          end=str(modeled_year) + '-01-01 01:00', freq='1h')

    topology['timestep_length_h'] = 1
    topology['carriers'] = ['electricity', 'hydrogen']
    topology['nodes'] = ['test_node1', 'test_node2']
    topology['technologies'] = {}

    topology['networks'] = {}
    topology['networks']['hydrogenTest'] = {}
    network_data = dm.create_empty_network_data(topology['nodes'])
    network_data['distance'].at['test_node1', 'test_node2'] = 1
    network_data['distance'].at['test_node2', 'test_node1'] = 1
    network_data['connection'].at['test_node1', 'test_node2'] = 1
    network_data['connection'].at['test_node2', 'test_node1'] = 1
    topology['networks']['hydrogenTest'] = network_data

    # Initialize instance of DataHandle
    data = dm.DataHandle(topology)

    # CLIMATE DATA
    data.read_climate_data_from_file('test_node1', r'./test/test_data/climate_data_test.p')
    data.read_climate_data_from_file('test_node2', r'./test/test_data/climate_data_test.p')

    # DEMAND
    demand = np.zeros(len(topology['timesteps']))
    demand[1] = 10
    data.read_demand_data('test_node1', 'hydrogen', demand)

    demand = np.zeros(len(topology['timesteps']))
    demand[0] = 10
    data.read_demand_data('test_node2', 'hydrogen', demand)

    # PRICE DATA
    price = np.ones(len(topology['timesteps'])) * 0
    data.read_import_price_data('test_node1', 'hydrogen', price)
    data.read_import_price_data('test_node2', 'hydrogen', price)
    price = np.ones(len(topology['timesteps'])) * 10
    data.read_import_price_data('test_node1', 'electricity', price)
    data.read_import_price_data('test_node2', 'electricity', price)

    # IMPORT/EXPORT LIMITS
    import_lim = np.zeros(len(topology['timesteps']))
    import_lim[0] = 100
    data.read_import_limit_data('test_node1', 'hydrogen', import_lim)

    import_lim = np.zeros(len(topology['timesteps']))
    import_lim[1] = 100
    data.read_import_limit_data('test_node2', 'hydrogen', import_lim)

    import_lim = np.ones(len(topology['timesteps'])) * 1000
    data.read_import_limit_data('test_node1', 'electricity', import_lim)

    import_lim = np.ones(len(topology['timesteps'])) * 1000
    data.read_import_limit_data('test_node2', 'electricity', import_lim)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

# create_data_model1()
# create_data_model2()
# create_data_technology_type1_PV()
# create_data_technology_type1_WT()
create_data_technology_PWA()
# create_data_network()



