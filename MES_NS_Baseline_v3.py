import cases.MES_NorthSea.utilities as dm
import numpy as np
from src.model_configuration import ModelConfiguration
from src.energyhub import EnergyHub
import pandas as pd

# General Settings
settings = dm.Settings(test=1)

# for stage in ['Baseline', 'Battery_on', 'Battery_off', 'Battery_all', 'Electrolysis_on', 'ElectricityGrid']:
for stage in ['ElectricityGrid']:

    settings.new_technologies_stage = stage
    if stage == 'ElectricityGrid':
        settings.networks.new_electricityAC = 1
        settings.networks.new_electricityDC = 1

    # Configuration
    configuration = dm.define_configuration()

    # Set Data
    nodes = dm.read_nodes(settings)
    topology = dm.define_topology(settings, nodes)
    topology = dm.define_installed_capacities(settings, nodes, topology)
    topology = dm.define_networks(settings, topology)
    topology = dm.define_new_technologies(settings, nodes, topology)

    if len(topology.timesteps) < 8760:
        settings.save_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20230912/TESTResult_' + settings.new_technologies_stage
    else:
        settings.save_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20230912/Result_' + settings.new_technologies_stage

    data = dm.define_data_handle(topology, nodes)
    data = dm.define_generic_production(settings, nodes, data)
    data = dm.define_hydro_inflow(settings, nodes, data)
    data = dm.define_demand(settings, nodes, data)
    data = dm.define_imports_exports(settings, nodes, data)

    # Read data
    dm.write_to_technology_data(settings)
    data.read_technology_data(load_path = settings.tec_data_path)
    data.read_network_data(load_path=settings.netw_data_path)
    data = dm.define_charging_efficiencies(settings, nodes, data)

    # Solve
    energyhub = EnergyHub(data, configuration)
    results = energyhub.quick_solve()
    # results.write_excel(settings.save_path)
