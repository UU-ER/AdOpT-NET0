import src.data_management as dm
import pandas as pd
import numpy as np
from types import SimpleNamespace
import copy
from src.model_configuration import ModelConfiguration

class Settings():

    def __init__(self):
        self.year = 2030
        self.scenario = 'GA'
        self.climate_year = 2008
        self.start_date = '01-01 00:00'
        self.end_date = '01-05 23:00'
        self.data_path = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/DataPreprocessing/00_CleanData/'
        self.save_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20230906/MES_NS_Benchmark'

        self.node_aggregation_type = {
            'onshore': [],
            'offshore': []}
        self.node_aggregation = {}

        self.new_technologies_stage = None

        # Network Settings
        self.networks = SimpleNamespace()
        self.networks.existing_electricity = 0
        self.networks.new_electricityAC = 0
        self.networks.new_electricityDC = 0
        self.networks.new_hydrogen = 0


def read_nodes(settings):
    """
    Reads onshore and offshore nodes from file
    """

    data_path = settings.data_path
    nodes = SimpleNamespace()

    node_data = data_path + '/Nodes/nodes.xlsx'
    node_list = pd.read_excel(node_data, sheet_name='Nodes_used')
    nodes.onshore_nodes = node_list[node_list['Type'] == 'onshore']['Node'].values.tolist()
    nodes.offshore_nodes = node_list[node_list['Type'] == 'offshore']['Node'].values.tolist()
    nodes.all = node_list['Node'].values.tolist()
    #
    # # Remove aggregated nodes
    # for node in settings.node_aggregation:
    #     nodes.all = [x for x in nodes.all if x not in settings.node_aggregation[node]]
    #     nodes.onshore_nodes = [x for x in nodes.onshore_nodes if x not in settings.node_aggregation[node]]
    #     nodes.offshore_nodes = [x for x in nodes.offshore_nodes if x not in settings.node_aggregation[node]]
    #
    # # Add new nodes
    # nodes.all.extend(list(settings.node_aggregation.keys()))
    # nodes.onshore_nodes.extend(list(settings.node_aggregation_type['onshore']))
    # nodes.onshore_nodes.extend(list(settings.node_aggregation_type['offshore']))

    return nodes


def define_topology(settings, nodes):
    """
    Defines topology
    """

    start_date = settings.start_date
    end_date = settings.end_date
    year = settings.year

    # Define Topology
    topology = dm.SystemTopology()
    topology.define_time_horizon(year=year, start_date=start_date, end_date=end_date, resolution=1)

    # Carriers
    topology.define_carriers(['electricity', 'gas', 'hydrogen'])

    # Nodes
    topology.define_nodes(nodes.all)

    return topology


def define_installed_capacities(settings, nodes, topology):
    data_path = settings.data_path

    new_tecs = pd.read_excel(data_path + '/InstalledCapacities_nonRE/InstalledCapacities_nonRE.xlsx',
                             sheet_name='Capacities at node',
                             index_col=0)

    for node in nodes.onshore_nodes:
        tecs_at_node = {'PowerPlant_Gas': round(new_tecs['Gas'][node],0),
                        'PowerPlant_Nuclear': round(new_tecs['Nuclear'][node],0),
                        'Storage_PumpedHydro_Closed': round(new_tecs['Hydro closed'][node],0),
                        'Storage_PumpedHydro_Open': round(new_tecs['Hydro open'][node], 0),
                        'Storage_PumpedHydro_Reservoir': round(new_tecs['Hydro reservoir'][node], 0)
                        }

        tecs_at_node = {k: v for k,v in tecs_at_node.items() if v > 0}

        topology.define_existing_technologies(node, tecs_at_node)

    return topology


def define_networks(settings, topology):
    """
    Defines the networks
    """
    data_path = settings.data_path + 'Networks/'


    def get_network_data(file_path):
        network = pd.read_csv(file_path, sep=';')
        network_data = {}
        network_data['size_matrix'] = dm.create_empty_network_matrix(topology.nodes)
        network_data['distance_matrix'] = dm.create_empty_network_matrix(topology.nodes)
        network_data['max_size_matrix'] = dm.create_empty_network_matrix(topology.nodes)
        for idx, row in network.iterrows():
            if (row['node0'] in topology.nodes) & (row['node1'] in topology.nodes):
                network_data['size_matrix'].at[row['node0'], row['node1']] = row['s_nom']*1000
                network_data['size_matrix'].at[row['node1'], row['node0']] = row['s_nom']*1000
                network_data['distance_matrix'].at[row['node0'], row['node1']] = row['length']
                network_data['distance_matrix'].at[row['node1'], row['node0']] = row['length']
                network_data['max_size_matrix'].at[row['node1'], row['node0']] = row['s_nom_max']
                network_data['max_size_matrix'].at[row['node1'], row['node0']] = row['s_nom_max']

        return network_data

    file_name = 'pyhub_el_ac.csv'
    ac_data = get_network_data(data_path + file_name)

    file_name = 'pyhub_el_dc.csv'
    dc_data = get_network_data(data_path + file_name)

    if settings.networks.existing_electricity:
        # Networks existing
        topology.define_existing_network('electricityAC', size=ac_data['size_matrix'], distance=ac_data['distance_matrix'])
        topology.define_existing_network('electricityDC', size=dc_data['size_matrix'], distance=dc_data['distance_matrix'])

    if settings.networks.new_electricityAC:
        # Networks - New Electricity AC
        topology.define_new_network('electricityAC', connections=ac_data['size_matrix'], distance=ac_data['distance_matrix'])

    if settings.networks.new_electricityDC:
        # Networks - New Electricity DC
        topology.define_new_network('electricityDC', connections=dc_data['size_matrix'], distance=dc_data['distance_matrix'])

    #
    # if settings.networks.new_hydrogen:
    #     # Networks - Hydrogen
    #     network_data = read_network_data(topology.nodes, settings.node_aggregation, data_path + '/Networks/NetworkDataHydrogen.xlsx', 0)
    #     topology.define_new_network('hydrogenPipeline_int', connections=network_data['connection'],
    #                                 distance=network_data['distance'])

    return topology


def define_data_handle(topology, nodes):
    data = dm.DataHandle(topology)
    for node in nodes.all:
        data.read_climate_data_from_file(node, r'.\data\climate_data_onshore.txt')

    return data


def define_generic_production(settings, nodes, data):

    data_path = settings.data_path

    for node in nodes.all:
        profile = pd.read_csv(data_path + '/ProductionProfiles_RE/' +
                              node + '_' +
                              str(settings.climate_year) + '.csv', index_col=0)
        data.read_production_profile(node, 'electricity', profile['total'].to_numpy(), 1)

    return data


def define_hydro_inflow(settings, nodes, data):

    data_path = settings.data_path

    # Hydro Inflow
    reservoir_inflow = pd.read_csv(data_path + 'Hydro_Inflows\HydroInflowReservoir' + str(settings.climate_year) + '.csv', index_col=0)

    opencycle_inflow = pd.read_csv(data_path + 'Hydro_Inflows\HydroInflowPump storage - Open Loop' + str(settings.climate_year) + '.csv', index_col=0)

    for node in nodes.onshore_nodes:
        if node in reservoir_inflow.columns:
            data.read_hydro_natural_inflow(node, 'Storage_PumpedHydro_Reservoir',
                                               reservoir_inflow[node].tolist())
        if node in opencycle_inflow.columns:
            data.read_hydro_natural_inflow(node, 'Storage_PumpedHydro_Open', opencycle_inflow[node].tolist())

    return data


def define_demand(settings, nodes, data):

    scenario = settings.scenario
    climate_year = settings.climate_year
    data_path = settings.data_path + 'demand/'

    demand_el = pd.read_csv(data_path + 'TotalDemand_'+scenario+'_' + str(climate_year) + '.csv')
    for node in nodes.onshore_nodes:
        data.read_demand_data(node, 'electricity', demand_el[node].to_numpy())

    return data


def define_imports_exports(settings, nodes, data):

    data_path = settings.data_path + 'ImportExport/ImportExport.xlsx'
    import_export = pd.read_excel(data_path, index_col=0)

    # IMPORT PRICES
    import_carrier_price = {'gas': 180,
                            'electricity':100000,
                            'hydrogen': 100000
                            }

    for node in nodes.all:
        for car in import_carrier_price:
            data.read_import_price_data(node, car, np.ones(len(data.topology.timesteps)) * import_carrier_price[car])

    for node in nodes.all:
        for car in import_carrier_price:
            data.read_import_limit_data(node, car,
                                        np.ones(len(data.topology.timesteps)) * import_export['Import_'+car][node])
            data.read_export_limit_data(node, car,
                                        np.ones(len(data.topology.timesteps)) * import_export['Export_' + car][node])

    # Emission Factor
    import_emissions = {'electricity': 0.3,
                        'hydrogen': -0.183
                        }
    for node in nodes.onshore_nodes:
        for car in import_emissions:
            data.read_import_emissionfactor_data(node, car, np.ones(len(data.topology.timesteps)) * import_emissions[car])

    return data


def define_charging_efficiencies(settings, nodes, data):
    installed_capacities = get_installed_capacities(settings, nodes)

    for node in nodes.onshore_nodes:
        storage_at_node = installed_capacities[node]['HydroStorage_charging']
        for storage in storage_at_node:
            data.technology_data[node][storage + '_existing'].fitted_performance.coefficients['charge_max'] = \
            installed_capacities[node]['HydroStorage_charging'][storage]['max_charge']
            data.technology_data[node][storage + '_existing'].fitted_performance.coefficients['discharge_max'] = \
            installed_capacities[node]['HydroStorage_charging'][storage]['max_discharge']


    return data


def define_new_technologies(settings, nodes, topology):

    data_path = settings.data_path

    new_tecs = pd.read_excel(data_path + '/NewTechnologies/NewTechnologies.xlsx', index_col=0)
    stage = settings.new_technologies_stage

    if not stage == None:
        for node in settings.node_aggregation:
            tec_list = new_tecs[stage][settings.node_aggregation[node]].str.cat(sep = ", ")
            tec_list = tec_list.split(', ')
            tec_list = list(dict.fromkeys(tec_list))

            new_tecs.at[node, stage] = ', '.join(str(s) for s in tec_list)

        for node in nodes.all:
            if not isinstance(new_tecs[stage][node], float):
                new_technologies = new_tecs[stage][node].split(', ')
                topology.define_new_technologies(node, new_technologies)

    return topology


def define_configuration():
    # Configuration
    configuration = ModelConfiguration()
    configuration.solveroptions.solver = 'gurobi_persistent'
    configuration.solveroptions.mipgap = 0.01
    configuration.solveroptions.lpwarmstart = 2
    configuration.solveroptions.numericfocus = 3
    configuration.optimization.save_log_files = 1
    configuration.optimization.monte_carlo.on = 0
    configuration.optimization.monte_carlo.N = 5
    configuration.optimization.typicaldays = 0
    configuration.solveroptions.timelim = 20

    configuration.solveroptions.intfeastol = 1e-3
    configuration.solveroptions.feastol = 1e-3
    configuration.solveroptions.numericfocus = 3
    configuration.optimization.objective = 'costs'
    configuration.optimization.pareto_points = 8

    return configuration