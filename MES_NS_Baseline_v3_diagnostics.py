import mes_north_sea.optimization.utilities as pp
import numpy as np
from src.model_configuration import ModelConfiguration
from src.energyhub import EnergyHub
import pandas as pd
from types import SimpleNamespace

# General Settings
settings = pp.Settings(test=0)

# for stage in ['Baseline', 'Battery_on', 'Battery_off', 'Battery_all', 'Electrolysis_on', 'ElectricityGrid']:
for stage in ['Baseline']:

    settings.new_technologies_stage = stage
    if stage == 'ElectricityGrid':
        settings.networks.new_electricityAC = 1
        settings.networks.new_electricityDC = 1

    # Configuration
    configuration = pp.define_configuration()

    # Set Data
    data_path = settings.data_path
    nodes = SimpleNamespace()

    node_data = data_path + '/nodes/nodes.xlsx'
    node_list = pd.read_excel(node_data, sheet_name='Nodes_used')
    for country in node_list['Country'].unique():
        nodes_per_country = node_list[node_list['Country'] == country]
        nodes.onshore_nodes = nodes_per_country[nodes_per_country['Type'] == 'onshore']['Node'].values.tolist()
        nodes.offshore_nodes = nodes_per_country[nodes_per_country['Type'].apply(lambda x: x.startswith('offshore'))][
            'Node'].values.tolist()
        nodes.all = nodes_per_country['Node'].values.tolist()

        print(country)

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
        pp.write_to_technology_data(settings)
        data.read_technology_data(load_path = settings.tec_data_path)
        data.read_network_data(load_path=settings.netw_data_path)
        data = pp.define_charging_efficiencies(settings, nodes, data)

        # Solve
        if stage == 'Baseline':
            configuration.optimization.objective = 'costs'

        configuration.reporting.save_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20231201/'

        if len(topology.timesteps) < 8760:
            configuration.reporting.case_name = 'TEST' + stage
        else:
            configuration.reporting.case_name = stage
        configuration.reporting.case_name = country

        energyhub = EnergyHub(data, configuration)
        results = energyhub.quick_solve()
