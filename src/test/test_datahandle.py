import pytest
import pandas as pd
import sys
import os

from src.data_management import *
from src.model_configuration import ModelConfiguration
from src.energyhub import EnergyHub

@pytest.mark.quicktest
def create_topology_sample():
    """
    Create a topology sample for the test_load_technology function
    """
    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-04 23:00', resolution=1)
    topology.define_carriers(['electricity'])
    topology.define_nodes(['testnode'])

    return topology

@pytest.mark.quicktest
def test_initializer():
    """
    tests the datahandle initilization
    """
    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-04 23:00', resolution=1)
    topology.define_carriers(['electricity', 'heat', 'gas'])
    topology.define_nodes(['testnode', 'offshore'])
    topology.define_new_technologies('testnode', ['Photovoltaic', 'Furnace_NG', 'Storage_Battery'])
    topology.define_new_technologies('offshore', ['WindTurbine_Offshore_11000'])

    distance = create_empty_network_matrix(topology.nodes)
    distance.at['onshore', 'offshore'] = 100
    distance.at['offshore', 'onshore'] = 100

    connection = create_empty_network_matrix(topology.nodes)
    connection.at['onshore', 'offshore'] = 1
    connection.at['offshore', 'onshore'] = 1
    topology.define_new_network('electricitySimple', distance=distance, connections=connection)

    data = DataHandle(topology)

@pytest.mark.quicktest
def test_load_technologies():
    """
    Tests the loading of all technologies contained in the technology folder
    """
    topology = SystemTopology()
    topology.define_time_horizon(year=2001, start_date='01-01 00:00', end_date='01-01 00:00', resolution=1)
    topology.define_carriers(['electricity', 'heat'])
    topology.define_nodes(['testnode'])

    data = DataHandle(topology)
    lat = 52
    lon = 5.16
    data.read_climate_data_from_api('testnode', lon, lat)
    # data.read_climate_data_from_file('testnode', r'./src/test/climate_data_test.p')

    directory = os.fsencode('./data/Technology_Data')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.json'):
            print(filename.replace('.json', ''))
            data.topology.technologies_new['testnode'] = [filename.replace('.json', '')]
            data.read_technology_data()
            continue
        else:
            continue
