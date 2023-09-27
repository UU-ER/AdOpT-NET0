import numpy as np
import copy
import sys
from src.data_management import *

from pathlib import Path


def create_data_test_data_handle():
    """
    Creates dataset for a model with two nodes.
    PV @ node 2
    electricity demand @ node 1
    electricity network in between
    should be infeasible
    """
    data_save_path = './src/test/test_data/data_handle_test.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='12-31 23:00', resolution=1)
    topology.define_carriers(['electricity'])
    topology.define_nodes(['test_node1'])

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # CLIMATE DATA
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)

    # DEMAND
    electricity_demand = np.ones(len(topology.timesteps)) * 1
    data.read_demand_data('test_node1', 'electricity', electricity_demand)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

def create_data_model1():
    """
    Creates dataset for a model with two nodes.
    PV @ node 2
    electricity demand @ node 1
    electricity network in between
    should be infeasible
    """
    data_save_path = './src/test/test_data/model1.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-31 23:00', resolution=1)
    topology.define_carriers(['electricity'])
    topology.define_nodes(['test_node1', 'test_node2'])
    topology.define_new_technologies('test_node2', ['Photovoltaic'])

    distance = create_empty_network_matrix(topology.nodes)
    distance.at['test_node1', 'test_node2'] = 100
    distance.at['test_node2', 'test_node1'] = 100

    connection = create_empty_network_matrix(topology.nodes)
    connection.at['test_node1', 'test_node2'] = 1
    connection.at['test_node2', 'test_node1'] = 1
    topology.define_new_network('electricityTest', distance=distance, connections=connection)

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # CLIMATE DATA
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)
    data.read_climate_data_from_file('test_node2', climate_data_path)

    # DEMAND
    electricity_demand = np.ones(len(topology.timesteps)) * 100
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
    should be feasible
    """
    data_save_path = './src/test/test_data/model2.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-02 23:00', resolution=1)
    topology.define_carriers(['heat', 'gas'])
    topology.define_nodes(['test_node1'])
    topology.define_new_technologies('test_node1', ['Furnace_NG'])

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # CLIMATE DATA
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)

    # DEMAND
    heat_demand = np.ones(len(topology.timesteps)) * 10
    data.read_demand_data('test_node1', 'heat', heat_demand)

    # PRICE DATA
    gas_price = np.ones(len(topology.timesteps)) * 1
    data.read_import_price_data('test_node1', 'gas', gas_price)

    # IMPORT/EXPORT LIMITS
    gas_import = np.ones(len(topology.timesteps)) * 100
    data.read_import_limit_data('test_node1', 'gas', gas_import)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

def create_data_emissionbalance1():
    """
    Creates dataset for a model with two nodes.
    PV & furnace @ node 1
    electricity & heat demand @ node 1
    offshore wind @ node 2
    electricity network in between
    """
    data_save_path = './src/test/test_data/emissionbalance1.p'
    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-01 01:00', resolution=1)
    topology.define_carriers(['electricity', 'heat', 'gas'])
    topology.define_nodes(['onshore', 'offshore'])
    topology.define_new_technologies('onshore', ['Furnace_NG'])

    distance = create_empty_network_matrix(topology.nodes)
    distance.at['onshore', 'offshore'] = 1
    distance.at['offshore', 'onshore'] = 1

    connection = create_empty_network_matrix(topology.nodes)
    connection.at['onshore', 'offshore'] = 1
    connection.at['offshore', 'onshore'] = 1
    topology.define_new_network('electricityTest', distance=distance, connections=connection)

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # CLIMATE DATA
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('onshore', climate_data_path)
    data.read_climate_data_from_file('offshore', climate_data_path)

    # DEMAND
    electricity_demand = np.ones(len(topology.timesteps)) * 10
    data.read_demand_data('onshore', 'electricity', electricity_demand)
    heat_demand = np.ones(len(topology.timesteps)) * 9
    data.read_demand_data('onshore', 'heat', heat_demand)

    # IMPORT
    gas_import = np.ones(len(topology.timesteps)) * 10
    data.read_import_limit_data('onshore', 'gas', gas_import)
    el_import = np.ones(len(topology.timesteps)) * 100
    data.read_import_limit_data('offshore', 'electricity', el_import)

    # EMISSIONS
    gas_imp_emis = np.ones(len(topology.timesteps)) * 0.4
    gas_imp_emis[1] = 0
    data.read_import_emissionfactor_data('onshore', 'gas', gas_imp_emis)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

def create_data_emissionbalance2():
    """
    Creates dataset for a model with two nodes.
    PV & furnace @ node 1
    electricity demand @ node 1
    """
    data_save_path = './src/test/test_data/emissionbalance2.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-04 01:00', resolution=1)
    topology.define_carriers(['electricity', 'heat', 'gas', 'hydrogen'])
    topology.define_nodes(['test_node1'])
    topology.define_new_technologies('test_node1', ['Storage_Battery', 'Photovoltaic', 'testCONV1_1'])

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # CLIMATE DATA
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)

    # DEMAND
    electricity_demand = np.ones(len(topology.timesteps)) * 10
    data.read_demand_data('test_node1', 'electricity', electricity_demand)

    # IMPORT
    gas_import = np.ones(len(topology.timesteps)) * 100
    data.read_import_limit_data('test_node1', 'gas', gas_import)

    # EMISSIONS
    gas_imp_emis = np.ones(len(topology.timesteps)) * 0
    data.read_import_emissionfactor_data('test_node1', 'gas', gas_imp_emis)

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
    data_save_path = './src/test/test_data/technology_type1_PV.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='12-31 23:00', resolution=1)
    topology.define_carriers(['electricity'])
    topology.define_nodes(['test_node1'])
    topology.define_new_technologies('test_node1', ['Photovoltaic'])

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # CLIMATE DATA
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)

    # DEMAND
    demand = np.ones(len(topology.timesteps)) * 10
    data.read_demand_data('test_node1', 'electricity', demand)

    # PRICE DATA
    price = np.ones(len(topology.timesteps)) * 10000
    data.read_import_price_data('test_node1', 'electricity', price)

    # IMPORT/EXPORT LIMITS
    import_lim = np.ones(len(topology.timesteps)) * 10
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
    Size of WT should be around max electricity demand
    """
    data_save_path = './src/test/test_data/technology_type1_WT.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='12-31 23:00', resolution=1)
    topology.define_carriers(['electricity'])
    topology.define_nodes(['test_node1'])
    topology.define_new_technologies('test_node1', ['TestWindTurbine_Onshore_1500'])

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # CLIMATE DATA
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)

    # DEMAND
    demand = np.ones(len(topology.timesteps)) * 10
    data.read_demand_data('test_node1', 'electricity', demand)

    # PRICE DATA
    price = np.ones(len(topology.timesteps)) * 1000
    data.read_import_price_data('test_node1', 'electricity', price)

    # IMPORT/EXPORT LIMITS
    import_lim = np.ones(len(topology.timesteps)) * 10
    data.read_import_limit_data('test_node1', 'electricity', import_lim)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

def create_data_technology_CONV():
    """
    Creates dataset for test_technology_CONV_PWA().
    heat demand @ node 1
    Technology type 1, 2, 3, gas,H2 -> heat, electricity
    """

    perf_function_type = [1, 2, 3]
    CONV_Type = [1, 2, 3, 4]
    for j in CONV_Type:
        for i in perf_function_type:
            if (j == 4) and i == 3:
                pass
            else:
                data_save_path = './src/test/test_data/technology_CONV' + str(j) + '_' + str(i) + '.p'

                topology = SystemTopology()
                topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-01 01:00', resolution=1)
                topology.define_carriers(['electricity', 'heat', 'gas', 'hydrogen'])
                topology.define_nodes(['test_node1'])
                topology.define_new_technologies('test_node1', ['testCONV' + str(j) + '_' + str(i)])

                # Initialize instance of DataHandle
                data = DataHandle(topology)

                # CLIMATE DATA
                climate_data_path = './src/test/climate_data_test.p'
                data.read_climate_data_from_file('test_node1', climate_data_path)

                # DEMAND
                demand_h = np.ones(len(topology.timesteps))
                demand_h[0]= 0.75
                demand_h[1] = 0.5
                data.read_demand_data('test_node1', 'heat', demand_h)

                # PRICE DATA
                if j != 3:
                    price = np.ones(len(topology.timesteps)) * 1
                    data.read_import_price_data('test_node1', 'gas', price)

                # IMPORT/EXPORT LIMITS
                import_lim = np.ones(len(topology.timesteps)) * 10
                data.read_import_limit_data('test_node1', 'gas', import_lim)
                import_lim = np.ones(len(topology.timesteps)) * 10
                data.read_import_limit_data('test_node1', 'hydrogen', import_lim)
                export_lim = np.ones(len(topology.timesteps)) * 10
                data.read_export_limit_data('test_node1', 'electricity', export_lim)

                # READ TECHNOLOGY AND NETWORK DATA
                data.read_technology_data()
                data.read_network_data()

                # SAVING/LOADING DATA FILE
                data.save(data_save_path)

def create_data_technologySTOR():
    """
    Creates dataset for a model with two nodes.
    WT, battery @ node 1
    electricity demand @ node 1
    two periods, rated wind speed at first, no wind at second. battery to balance
    """
    data_save_path = './src/test/test_data/technologySTOR.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-01 01:00', resolution=1)
    topology.define_carriers(['electricity'])
    topology.define_nodes(['test_node1'])
    topology.define_new_technologies('test_node1', ['TestWindTurbine_Onshore_1500', 'testSTOR'])

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # CLIMATE DATA
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)
    data.node_data['test_node1'].data['climate_data']['ws10'][0] = 15
    data.node_data['test_node1'].data['climate_data']['ws10'][1] = 0

    # DEMAND
    electricity_demand = np.ones(len(topology.timesteps)) * 0.1
    data.read_demand_data('test_node1', 'electricity', electricity_demand)

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
    data_save_path = './src/test/test_data/networks.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-01 01:00', resolution=1)
    topology.define_carriers(['electricity', 'hydrogen'])
    topology.define_nodes(['test_node1', 'test_node2'])

    distance = create_empty_network_matrix(topology.nodes)
    distance.at['test_node1', 'test_node2'] = 1
    distance.at['test_node2', 'test_node1'] = 1

    connection = create_empty_network_matrix(topology.nodes)
    connection.at['test_node1', 'test_node2'] = 1
    connection.at['test_node2', 'test_node1'] = 1
    topology.define_new_network('hydrogenTest', distance=distance, connections=connection)

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # CLIMATE DATA
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)
    data.read_climate_data_from_file('test_node2', climate_data_path)

    # DEMAND
    demand = np.zeros(len(topology.timesteps))
    demand[1] = 10
    data.read_demand_data('test_node1', 'hydrogen', demand)

    demand = np.zeros(len(topology.timesteps))
    demand[0] = 10
    data.read_demand_data('test_node2', 'hydrogen', demand)

    # PRICE DATA
    price = np.ones(len(topology.timesteps)) * 0
    data.read_import_price_data('test_node1', 'hydrogen', price)
    data.read_import_price_data('test_node2', 'hydrogen', price)
    price = np.ones(len(topology.timesteps)) * 10
    data.read_import_price_data('test_node1', 'electricity', price)
    data.read_import_price_data('test_node2', 'electricity', price)

    # IMPORT/EXPORT LIMITS
    import_lim = np.zeros(len(topology.timesteps))
    import_lim[0] = 100
    data.read_import_limit_data('test_node1', 'hydrogen', import_lim)

    import_lim = np.zeros(len(topology.timesteps))
    import_lim[1] = 100
    data.read_import_limit_data('test_node2', 'hydrogen', import_lim)

    import_lim = np.ones(len(topology.timesteps)) * 1000
    data.read_import_limit_data('test_node1', 'electricity', import_lim)

    import_lim = np.ones(len(topology.timesteps)) * 1000
    data.read_import_limit_data('test_node2', 'electricity', import_lim)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

def create_data_addtechnology():
    """
    Creates dataset for test_addtechnology().
    electricity demand @ node 2
    battery at node 2
    first, WT at node 1, later PV at node 2
    """
    data_save_path = './src/test/test_data/addtechnology.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-31 23:00', resolution=1)
    topology.define_carriers(['electricity'])
    topology.define_nodes(['test_node1', 'test_node2'])
    topology.define_new_technologies('test_node1', ['TestWindTurbine_Onshore_1500'])
    topology.define_new_technologies('test_node2', ['Storage_Battery'])

    distance = create_empty_network_matrix(topology.nodes)
    distance.at['test_node1', 'test_node2'] = 1
    distance.at['test_node2', 'test_node1'] = 1

    connection = create_empty_network_matrix(topology.nodes)
    connection.at['test_node1', 'test_node2'] = 1
    connection.at['test_node2', 'test_node1'] = 1
    topology.define_new_network('electricitySimple', distance=distance, connections=connection)

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # CLIMATE DATA
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)
    data.read_climate_data_from_file('test_node2', climate_data_path)

    # DEMAND
    demand = np.ones(len(topology.timesteps)) * 10
    data.read_demand_data('test_node2', 'electricity', demand)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

def create_data_time_algorithms():
    """
    Creates dataset for a model with one node.
    Temporal resolution 31 days.
    electricity demand @ node 1
    Technologies are PV and storage
    """
    data_save_path = './src/test/test_data/time_algorithms.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='03-31 23:00', resolution=1)
    topology.define_carriers(['electricity', 'gas', 'hydrogen'])
    topology.define_nodes(['test_node1','test_node2'])
    topology.define_new_technologies('test_node1', ['GasTurbine_simple', 'Storage_Battery'])
    topology.define_new_technologies('test_node2', ['Photovoltaic', 'TestWindTurbine_Onshore_1500'])

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # NETWORKS
    distance = create_empty_network_matrix(topology.nodes)
    distance.at['test_node1', 'test_node2'] = 1
    distance.at['test_node2', 'test_node1'] = 1
    connection = create_empty_network_matrix(topology.nodes)
    connection.at['test_node1', 'test_node2'] = 1
    connection.at['test_node2', 'test_node1'] = 1
    topology.define_new_network('electricityTest', distance=distance, connections=connection)

    # CLIMATE DATA
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)
    data.read_climate_data_from_file('test_node2', climate_data_path)

    # DEMAND
    electricity_demand = np.ones(len(topology.timesteps)) * 100
    data.read_demand_data('test_node1', 'electricity', electricity_demand)

    # IMPORT
    gas_import = np.ones(len(topology.timesteps)) * 10
    data.read_import_limit_data('test_node1', 'gas', gas_import)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

def create_data_optimization_types():

    data_save_path = './src/test/test_data/optimization_types.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-31 23:00', resolution=1)
    topology.define_carriers(['electricity', 'gas', 'hydrogen'])
    topology.define_nodes(['test_node1'])
    topology.define_new_technologies('test_node1', ['Photovoltaic', 'GasTurbine_simple'])

    data = DataHandle(topology)

    # CLIMATE DATA
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)

    # DEMAND
    demand = np.ones(len(topology.timesteps)) * 10
    data.read_demand_data('test_node1', 'electricity', demand)

    # PRICE DATA
    price = np.ones(len(topology.timesteps)) * 0
    data.read_import_price_data('test_node1', 'gas', price)

    # IMPORT/EXPORT LIMITS
    import_lim = np.ones(len(topology.timesteps)) * 10000
    data.read_import_limit_data('test_node1', 'gas', import_lim)

    data.read_technology_data()
    data.read_network_data()

    data.technology_data['test_node1']['Photovoltaic'].economics.capex_data['unit_capex'] = 200
    data.technology_data['test_node1']['GasTurbine_simple'].economics.capex_data['unit_capex'] = 10
    data.save(data_save_path)


def create_data_existing_technologies():

    def create_topology():
        # TOPOLOGY
        topology = SystemTopology()
        topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-31 23:00', resolution=1)
        topology.define_carriers(['electricity'])
        topology.define_nodes(['test_node1'])
        topology.define_new_technologies('test_node1', ['Photovoltaic'])
        return topology

    def create_data(topology):
        data = DataHandle(topology)

        # CLIMATE DATA
        climate_data_path = './src/test/climate_data_test.p'
        data.read_climate_data_from_file('test_node1', climate_data_path)

        # DEMAND
        demand = np.ones(len(topology.timesteps)) * 10
        data.read_demand_data('test_node1', 'electricity', demand)

        # PRICE DATA
        price = np.ones(len(topology.timesteps)) * 1000
        data.read_import_price_data('test_node1', 'electricity', price)

        # IMPORT/EXPORT LIMITS
        import_lim = np.ones(len(topology.timesteps)) * 10
        data.read_import_limit_data('test_node1', 'electricity', import_lim)

        # READ TECHNOLOGY AND NETWORK DATA
        data.read_technology_data()
        data.read_network_data()
        return data

    topology1 = create_topology()
    topology2 = create_topology()
    topology3 = create_topology()

    topology2.define_existing_technologies('test_node1', {'Storage_Battery': 4})
    topology3.define_existing_technologies('test_node1', {'Storage_Battery': 3000})

    data_save_path1 = './src/test/test_data/existing_tecs1.p'
    data_save_path2 = './src/test/test_data/existing_tecs2.p'
    data_save_path3 = './src/test/test_data/existing_tecs3.p'

    data1 = create_data(topology1)
    data1.save(data_save_path1)
    data2 = create_data(topology2)
    data2.save(data_save_path2)
    data3 = create_data(topology3)
    data3.technology_data['test_node1']['Storage_Battery_existing'].decommission = 1
    data3.save(data_save_path3)

def create_data_existing_networks():
    def create_topology():
        topology = SystemTopology()
        topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-01 01:00', resolution=1)
        topology.define_carriers(['electricity'])
        topology.define_nodes(['test_node1', 'test_node2'])
        return topology

    def create_data(topology):
        # Initialize instance of DataHandle
        data = DataHandle(topology)

        # CLIMATE DATA
        climate_data_path = './src/test/climate_data_test.p'
        data.read_climate_data_from_file('test_node1', climate_data_path)
        data.read_climate_data_from_file('test_node2', climate_data_path)

        # DEMAND
        demand = np.ones(len(topology.timesteps)) * 10
        data.read_demand_data('test_node1', 'electricity', demand)

        # IMPORT/EXPORT LIMITS
        import_lim = np.ones(len(topology.timesteps)) * 100
        data.read_import_limit_data('test_node2', 'electricity', import_lim)

        # READ TECHNOLOGY AND NETWORK DATA
        data.read_technology_data()
        data.read_network_data()
        return data

    topology1 = create_topology()

    topology2 = create_topology()
    distance = create_empty_network_matrix(topology1.nodes)
    distance.at['test_node1', 'test_node2'] = 1
    distance.at['test_node2', 'test_node1'] = 1
    connection = create_empty_network_matrix(topology1.nodes)
    connection.at['test_node1', 'test_node2'] = 1
    connection.at['test_node2', 'test_node1'] = 1
    topology2.define_new_network('electricityTest', distance=distance, connections=connection)

    topology3 = create_topology()
    size_initial = create_empty_network_matrix(topology1.nodes)
    size_initial.at['test_node1', 'test_node2'] = 100
    size_initial.at['test_node2', 'test_node1'] = 100
    topology3.define_existing_network('electricityTest', size=size_initial, distance=distance)

    data1 = create_data(topology1)
    data2 = create_data(topology2)
    data2.network_data['electricityTest'].economics.opex_fixed = 1
    data3 = create_data(topology3)
    data3.network_data['electricityTest_existing'].economics.opex_fixed = 1
    data4 = copy.deepcopy(data3)
    data4.network_data['electricityTest_existing'].decommission = 1

    data_save_path1 = './src/test/test_data/existing_netw1.p'
    data_save_path2 = './src/test/test_data/existing_netw2.p'
    data_save_path3 = './src/test/test_data/existing_netw3.p'
    data_save_path4 = './src/test/test_data/existing_netw4.p'

    data1.save(data_save_path1)
    data2.save(data_save_path2)
    data3.save(data_save_path3)
    data4.save(data_save_path4)

def create_test_data_dac():
    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-05 23:00', resolution=1)
    topology.define_carriers(['electricity', 'heat', 'CO2'])
    topology.define_nodes(['test_node1'])
    topology.define_new_technologies('test_node1',
                                     ['DAC_Adsorption', 'Storage_CO2', 'Photovoltaic'])

    data = DataHandle(topology)

    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)

    # DEMAND
    co2demand = np.ones(len(topology.timesteps)) * 0.01
    data.read_demand_data('test_node1', 'CO2', co2demand)

    # IMPORT
    heat_import = np.ones(len(topology.timesteps)) * 100
    data.read_import_limit_data('test_node1', 'heat', heat_import)

    data.read_technology_data()
    data.read_network_data()

    data_save_path = './src/test/test_data/dac.p'

    data.save(data_save_path)


def create_data_technologyOpen_Hydro():
    """
    Creates dataset for a model with two nodes.
    WT, battery @ node 1
    electricity demand @ node 1
    two periods, rated wind speed at first, no wind at second. battery to balance
    """
    data_save_path = './src/test/test_data/technologyOpenHydro.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-01 01:00', resolution=1)
    topology.define_carriers(['electricity'])
    topology.define_nodes(['test_node1'])
    topology.define_new_technologies('test_node1', ['TestWindTurbine_Onshore_1500', 'TestPumpedHydro_Open'])

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # CLIMATE DATA
    inflow = np.ones(len(topology.timesteps)) * 10
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)
    data.read_hydro_natural_inflow('test_node1', 'TestPumpedHydro_Open', inflow)
    data.node_data['test_node1'].data['climate_data']['ws10'][0] = 15
    data.node_data['test_node1'].data['climate_data']['ws10'][1] = 0

    # DEMAND
    electricity_demand = np.ones(len(topology.timesteps)) * 1
    data.read_demand_data('test_node1', 'electricity', electricity_demand)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

    data_save_path = './src/test/test_data/technologyOpenHydro_max_discharge.p'

    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-01 01:00', resolution=1)
    topology.define_carriers(['electricity'])
    topology.define_nodes(['test_node1'])
    topology.define_new_technologies('test_node1', ['TestWindTurbine_Onshore_1500', 'TestPumpedHydro_Open_max_discharge'])

    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # CLIMATE DATA
    inflow = np.ones(len(topology.timesteps)) * 10
    climate_data_path = './src/test/climate_data_test.p'
    data.read_climate_data_from_file('test_node1', climate_data_path)
    data.read_hydro_maximum_discharge('test_node1', 'TestPumpedHydro_Open_max_discharge', np.ones(len(data.topology.timesteps)) * 0)
    data.read_hydro_natural_inflow('test_node1', 'TestPumpedHydro_Open_max_discharge', inflow)
    data.node_data['test_node1'].data['climate_data']['ws10'][0] = 15
    data.node_data['test_node1'].data['climate_data']['ws10'][1] = 0

    # DEMAND
    electricity_demand = np.ones(len(topology.timesteps)) * 1
    data.read_demand_data('test_node1', 'electricity', electricity_demand)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)



def create_data_carbon_tax():
    """
    Creates dataset for a model with one node and a carbon tax.
    furnace @ node 1
    heat demand @ node 1
    """
    data_save_path = './src/test/test_data/carbon_tax.p'
    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-01 01:00', resolution=1)
    topology.define_carriers(['heat', 'gas'])
    topology.define_nodes(['onshore'])
    topology.define_new_technologies('onshore', ['Furnace_NG'])



    # Initialize instance of DataHandle
    data = DataHandle(topology)

    # DEMAND

    heat_demand = np.ones(len(topology.timesteps)) * 9
    data.read_demand_data('onshore', 'heat', heat_demand)

    # CARBON TAX
    carbontax = np.ones(len(topology.timesteps)) * 10
    data.read_carbon_price_data(carbontax, 'tax')

    # IMPORT
    gas_import = np.ones(len(topology.timesteps)) * 20
    data.read_import_limit_data('onshore', 'gas', gas_import)

    # READ TECHNOLOGY AND NETWORK DATA
    data.read_technology_data()
    data.read_network_data()

    # SAVING/LOADING DATA FILE
    data.save(data_save_path)

def create_data_carbon_subsidy():
    """
    Creates dataset for a model with one node, DAC and a carbon subsidy
    """
    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-02 00:00', resolution=1)
    topology.define_carriers(['electricity', 'heat', 'CO2'])
    topology.define_nodes(['test_node1'])
    topology.define_new_technologies('test_node1',
                                     ['DAC_Adsorption', 'Storage_CO2'])

    data = DataHandle(topology)

    data.read_climate_data_from_file('test_node1', './src/test/climate_data_test.p')

    # DEMAND
    co2demand = np.ones(len(topology.timesteps)) * 0.01
    data.read_demand_data('test_node1', 'CO2', co2demand)

    # data.read_export_limit_data('test_node1', 'CO2', np.ones(len(topology.timesteps)) * 0.01)

    # CARBON SUBSIDY


    # IMPORT
    heat_import = np.ones(len(topology.timesteps)) * 100
    data.read_import_limit_data('test_node1', 'heat', heat_import)
    electricity_import = np.ones(len(topology.timesteps)) * 100
    data.read_import_limit_data('test_node1', 'electricity', electricity_import)

    #carbon_import = np.ones(len(topology.timesteps)) * 1
    #data.read_import_limit_data('test_node1', 'CO2', carbon_import)
    #carbon_export = np.ones(len(topology.timesteps)) * 1
    #data.read_import_limit_data('test_node1', 'CO2', carbon_export)

    data.read_technology_data()
    data.read_network_data()


    data_save_path = './src/test/test_data/carbon_subsidy.p'

    data.save(data_save_path)




create_data_test_data_handle()
create_data_model1()
create_data_model2()
create_data_emissionbalance1()
create_data_emissionbalance2()
create_data_technology_type1_PV()
create_data_technology_type1_WT()
create_data_technology_CONV()
create_data_network()
create_data_addtechnology()
create_data_technologySTOR()
create_data_time_algorithms()
create_data_optimization_types()
create_data_existing_technologies()
create_data_existing_networks()
create_test_data_dac()
create_data_technologyOpen_Hydro()
create_data_carbon_tax()
create_data_carbon_subsidy()