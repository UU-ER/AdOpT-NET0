import pytest
import pandas as pd
import src.data_management as dm
import sys
import os

def create_topology_sample():
    """
    Create a topology sample for the test_load_technology function
    """
    topology = dm.SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-04 23:00', resolution=1)
    topology.define_carriers(['electricity'])
    topology.define_nodes(['testnode'])

    return topology

def test_initializer():
    """
    tests the datahandle initilization
    """
    topology = dm.SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-04 23:00', resolution=1)
    topology.define_carriers(['electricity', 'heat', 'gas'])
    topology.define_nodes(['testnode', 'offshore'])
    topology.define_new_technologies('testnode', ['PV', 'Furnace_NG', 'battery'])
    topology.define_new_technologies('offshore', ['WT_OS_11000'])

    distance = dm.create_empty_network_matrix(topology.nodes)
    distance.at['onshore', 'offshore'] = 100
    distance.at['offshore', 'onshore'] = 100

    connection = dm.create_empty_network_matrix(topology.nodes)
    connection.at['onshore', 'offshore'] = 1
    connection.at['offshore', 'onshore'] = 1
    topology.define_new_network('electricitySimple', distance=distance, connections=connection)

    data = dm.DataHandle(topology)


def test_load_technologies():
    """
    Tests the loading of all technologies contained in the technology folder
    """
    topology = create_topology_sample()
    data = dm.DataHandle(topology)
    lat = 52
    lon = 5.16
    data.read_climate_data_from_file('testnode', './test/test_data/climate_data_test.p')

    directory = os.fsencode('./data/technology_data')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".json"):
            print(filename.replace('.json', ''))
            data.topology.technologies_new['testnode'] = [filename.replace('.json', '')]
            data.read_technology_data()
            continue
        else:
            continue

def test_k_means_clustering():
    """
    Test the k-means clustering process
    """
    data = dm.load_data_handle(r'./test/test_data/k_means.p')
    clustered_data = dm.ClusteredDataHandle()
    nr_days_cluster = 40
    clustered_data.cluster_data(data, nr_days_cluster)