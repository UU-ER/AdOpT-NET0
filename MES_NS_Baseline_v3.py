import mes_north_sea.optimization.utilities as pp
import numpy as np
from src.model_configuration import ModelConfiguration
from src.energyhub import EnergyHub
import pandas as pd
import random

# General Settings
settings = pp.Settings(test=1)
pp.write_to_technology_data(settings)
pp.write_to_network_data(settings)


# for stage in ['Baseline',
#               'Battery_on',
#               'Battery_off',
#               'Battery_all',
#               'Electrolysis_on',
#               'ElectricityGrid_all',
#               'ElectricityGrid_on',
#               'ElectricityGrid_off',
#               'ElectricityGrid_noBorder',
#               ]:
for stage in ['ElectricityGrid_all',
              'ElectricityGrid_on',
              'ElectricityGrid_off',
              'ElectricityGrid_noBorder',
              ]:

    settings.new_technologies_stage = stage

    # Configuration
    configuration = pp.define_configuration()

    # Set Data
    nodes = pp.read_nodes(settings)
    topology = pp.define_topology(settings, nodes)
    topology = pp.define_installed_capacities(settings, nodes, topology)
    topology = pp.define_networks(settings, topology)
    topology = pp.define_new_technologies(settings, nodes, topology)

    data = pp.define_data_handle(topology, nodes)
    data = pp.define_generic_production(settings, nodes, data)
    data = pp.define_hydro_inflow(settings, nodes, data)
    data = pp.define_demand(settings, nodes, data)
    data = pp.define_imports_exports(settings, nodes, data)

    # Read data
    data.read_technology_data(load_path = settings.tec_data_path)
    data.read_network_data(load_path=settings.netw_data_path)
    data = pp.define_charging_efficiencies(settings, nodes, data)

    # Alter capex of electrolysis to remove symmetry
    for node in data.technology_data:
        if 'Electrolyser_PEM' in data.technology_data[node]:
            data.technology_data[node]['Electrolyser_PEM'].economics.capex_data['unit_capex'] = data.technology_data[node]['Electrolyser_PEM'].economics.capex_data['unit_capex'] * random.uniform(0.99, 1.01)

    configuration.reporting.save_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/baseline_demand/'
    configuration.reporting.save_summary_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/baseline_demand/'
    # configuration.reporting.save_path = './userData/'
    # configuration.reporting.save_summary_path = './userData/'
    # Solve
    if stage == 'Baseline':
        configuration.optimization.objective = 'costs'
    if len(topology.timesteps) < 8760:
        configuration.reporting.case_name = 'TEST' + stage
    else:
        configuration.reporting.case_name = stage

    energyhub = EnergyHub(data, configuration)
    results = energyhub.quick_solve()
