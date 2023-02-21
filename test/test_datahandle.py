import pytest
import pandas as pd
import src.data_management as dm
import sys
import os

def create_topology_sample():
    """
    Create a topology sample for the test_load_technology function
    """
    modeled_year = 2001
    topology = {}
    topology['timesteps'] = pd.date_range(start=str(modeled_year) + '-01-01 00:00',
                                          end=str(modeled_year) + '-12-31 23:00', freq='1h')
    topology['timestep_length_h'] = 1
    topology['carriers'] = ['electricity']
    topology['nodes'] = ['testnode']
    topology['technologies'] = {}
    topology['technologies']['testnode'] = []

    topology['networks'] = {}
    return topology

def test_initializer():
    """
    tests the datahandle initilization
    """
    modeled_year = 2001
    topology = {}
    topology['timesteps'] = pd.date_range(start=str(modeled_year) + '-01-01 00:00',
                                          end=str(modeled_year) + '-12-31 23:00', freq='1h')
    topology['timestep_length_h'] = 1
    topology['carriers'] = ['electricity', 'heat', 'gas']
    topology['nodes'] = ['testnode', 'offshore']
    topology['technologies'] = {}
    topology['technologies']['testnode'] = ['PV', 'Furnace_NG', 'battery']
    topology['technologies']['offshore'] = ['WT_OS_11000']

    topology['networks'] = {}
    topology['networks']['electricity'] = {}
    network_data = dm.create_empty_network_data(topology['nodes'])
    network_data['distance'].at['onshore', 'offshore'] = 100
    network_data['distance'].at['offshore', 'onshore'] = 100
    network_data['connection'].at['onshore', 'offshore'] = 1
    network_data['connection'].at['offshore', 'onshore'] = 1
    topology['networks']['electricity']['AC'] = network_data

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
            data.topology['technologies']['testnode'] = [filename.replace('.json', '')]
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