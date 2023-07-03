from cases.NorthSea_helpers.read_input_data import *
import numpy as np

from src.model_configuration import ModelConfiguration
from src.energyhub import EnergyHub
from cases.NorthSea_helpers.utilities import *


def determine_deficit(node, settings):
    nodes = read_nodes(settings)
    nodes.all = [node]
    nodes.onshore_nodes = [node]
    nodes.offshore_nodes = []
    topology = define_topology(settings, nodes)
    topology = define_installed_capacities(settings, nodes, topology)
    topology = define_networks(settings, topology)
    topology = define_new_technologies(settings, nodes, topology)

    data = define_data_handle(topology, nodes)
    data = define_generic_production(settings, nodes, data)
    data = define_hydro_inflow(settings, nodes, data)
    data = define_demand(settings, nodes, data)
    data = define_imports(settings, nodes, data)

    # Read data
    tec_data_path = settings.data_path + '/Technology_Data/'
    write_to_technology_data(tec_data_path, settings)
    data.read_technology_data(path=tec_data_path)
    data.read_network_data()
    data = define_charging_efficiencies(settings, nodes, data)

    # Solve
    configuration = define_configuration()

    configuration.optimization.objective = 'costs'

    energyhub = EnergyHub(data, configuration)
    results = energyhub.quick_solve()
    return max(results.detailed_results[0].energybalance[node]['electricity']['Import'])

# General Settings
settings = Settings()

settings.new_technologies_stage = None
settings.networks.existing_electricity = 1
settings.networks.new_electricityAC = 0
settings.networks.new_electricityDC = 0
settings.networks.new_hydrogen = 0

node = 'onDE'

print(determine_deficit(node, settings))






