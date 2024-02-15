import src.data_management as dm
import pandas as pd
import numpy as np
from types import SimpleNamespace
import copy
from src.model_configuration import ModelConfiguration
import os
import json


class Settings():

    def __init__(self, test):
        self.test = test
        self.year = 2030
        self.scenario = 'NT'
        self.climate_year = 2008
        if test:
            self.start_date = '05-01 00:00'
            self.end_date = '05-03 00:00'
        else:
            self.start_date = '01-01 00:00'
            self.end_date = '12-31 23:00'
        self.data_path = './mes_north_sea/clean_data/'
        self.save_path = ''
        self.tec_data_path = self.data_path + 'technology_data'
        self.netw_data_path = self.data_path + 'network_data'

        self.node_aggregation_type = {
            'onshore': [],
            'offshore': []}
        self.node_aggregation = {}

        self.new_technologies_stage = None

        # Network Settings
        self.networks = SimpleNamespace()
        self.networks.existing_electricity = 1
        self.networks.new_electricityAC = 0
        self.networks.new_electricityDC = 0
        self.networks.new_hydrogen = 0


def read_nodes(settings):
    """
    Reads onshore and offshore nodes from file
    """

    data_path = settings.data_path
    nodes = SimpleNamespace()

    node_data = data_path + '/nodes/nodes.xlsx'
    node_list = pd.read_excel(node_data, sheet_name='Nodes_used')
    nodes.onshore_nodes = node_list[node_list['Type'] == 'onshore']['Node'].values.tolist()
    nodes.offshore_nodes = node_list[node_list['Type'].apply(lambda x: x.startswith('offshore'))]['Node'].values.tolist()
    nodes.all = {}
    for row in node_list.iterrows():
        node_data = {}
        node_data['lon'] = row[1]['x']
        node_data['lat'] = row[1]['y']
        nodes.all[row[1]['Node']] = node_data

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

    new_tecs = pd.read_csv(data_path + 'installed_capacities/capacities_node.csv',
                             index_col=0)

    for node in nodes.onshore_nodes:
        new_at_node = new_tecs[new_tecs['Node'] == node][['Technology', 'Capacity our work']].set_index('Technology').to_dict()['Capacity our work']
        tecs_at_node = {'PowerPlant_Gas': round(new_at_node.get('Gas', 0),0),
                        'PowerPlant_Nuclear': round(new_at_node.get('Nuclear', 0),0),
                        'PowerPlant_Oil': round(new_at_node.get('Oil', 0),0),
                        'PowerPlant_Coal': round(new_at_node.get('Coal & Lignite', 0),0),
                        'Storage_PumpedHydro_Closed': round(new_at_node.get('Hydro - Pump Storage Closed Loop (Energy)', 0),0),
                        'Storage_PumpedHydro_Open': round(new_at_node.get('Hydro - Pump Storage Open Loop (Energy)', 0),0),
                        'Storage_PumpedHydro_Reservoir': round(new_at_node.get('Hydro - Reservoir (Energy)', 0),0),
                        }

        tecs_at_node = {k: v for k,v in tecs_at_node.items() if v > 0}

        topology.define_existing_technologies(node, tecs_at_node)

    return topology


def define_networks(settings, topology):
    """
    Defines the networks
    """
    data_path = settings.data_path + 'networks/'
    stage = settings.new_technologies_stage

    def get_network_data(file_path):
        network = pd.read_csv(file_path, sep=';')
        network_data = {}
        network_data['size_matrix'] = dm.create_empty_network_matrix(topology.nodes)
        network_data['distance_matrix'] = dm.create_empty_network_matrix(topology.nodes)
        network_data['max_size_matrix'] = dm.create_empty_network_matrix(topology.nodes)
        network_data['connection_matrix'] = dm.create_empty_network_matrix(topology.nodes)
        for idx, row in network.iterrows():
            if (row['node0'] in topology.nodes) & (row['node1'] in topology.nodes):
                network_data['size_matrix'].at[row['node0'], row['node1']] = row['s_nom']*1000
                network_data['size_matrix'].at[row['node1'], row['node0']] = row['s_nom']*1000
                network_data['distance_matrix'].at[row['node0'], row['node1']] = row['length']
                network_data['distance_matrix'].at[row['node1'], row['node0']] = row['length']
                network_data['max_size_matrix'].at[row['node1'], row['node0']] = row['s_nom_max']*1000 - row['s_nom']*1000
                network_data['max_size_matrix'].at[row['node0'], row['node1']] = row['s_nom_max']*1000 - row['s_nom']*1000
                if row['s_nom_max'] > 0:
                    network_data['connection_matrix'].at[row['node1'], row['node0']] = 1
                    network_data['connection_matrix'].at[row['node0'], row['node1']] = 1

        return network_data

    if stage == 'ElectricityGrid_all':
        file_name_ac = 'pyhub_el_ac_all.csv'
        file_name_dc = 'pyhub_el_dc_all.csv'
    elif stage == 'ElectricityGrid_on':
        file_name_ac = 'pyhub_el_ac_on.csv'
        file_name_dc = 'pyhub_el_dc_on.csv'
    elif stage == 'ElectricityGrid_off':
        file_name_ac = 'pyhub_el_ac_off.csv'
        file_name_dc = 'pyhub_el_dc_off.csv'
    elif stage == 'ElectricityGrid_noBorder':
        file_name_ac = 'pyhub_el_ac_noBorder.csv'
        file_name_dc = 'pyhub_el_dc_noBorder.csv'
    else:
        file_name_ac = 'pyhub_el_ac_all.csv'
        file_name_dc = 'pyhub_el_dc_all.csv'


    ac_data = get_network_data(data_path + file_name_ac)
    dc_data = get_network_data(data_path + file_name_dc)

    if settings.networks.existing_electricity:
        # Networks existing
        topology.define_existing_network('electricityAC', size=ac_data['size_matrix'], distance=ac_data['distance_matrix'])
        topology.define_existing_network('electricityDC', size=dc_data['size_matrix'], distance=dc_data['distance_matrix'])


    if 'ElectricityGrid' in stage:
        # Networks - New Electricity AC
        topology.define_new_network('electricityAC', connections=ac_data['connection_matrix'],
                                    distance=ac_data['distance_matrix'],
                                    size_max_arcs=ac_data['max_size_matrix'])

        # Networks - New Electricity DC
        topology.define_new_network('electricityDC_int', connections=dc_data['connection_matrix'],
                                    distance=dc_data['distance_matrix'],
                                    size_max_arcs=round(dc_data['max_size_matrix']/2000,0))

    #
    # if settings.networks.new_hydrogen:
    #     # Networks - Hydrogen
    #     network_data = read_network_data(topology.nodes, settings.node_aggregation, data_path + '/Networks/NetworkDataHydrogen.xlsx', 0)
    #     topology.define_new_network('hydrogenPipeline_int', connections=network_data['connection'],
    #                                 distance=network_data['distance'])

    return topology


def define_data_handle(topology, nodes):
    data = dm.DataHandle(topology)
    for node in nodes.all.keys():
        data.read_climate_data_from_file(node, r'.\data\climate_data_onshore.txt')

    return data


def define_generic_production(settings, nodes, data):

    data_path = settings.data_path
    profiles = pd.read_csv(data_path + 'production_profiles_re/production_profiles_re.csv', index_col=0, header=[0, 1])

    for node in nodes.all.keys():
        profile = profiles.loc[:, (node, 'total')].to_numpy().round(1)
        data.read_production_profile(node, 'electricity', profile, 1)

    return data


def define_hydro_inflow(settings, nodes, data):

    data_path = settings.data_path

    # Hydro Inflow
    inflows = pd.read_csv(data_path + 'hydro_inflows\hydro_inflows.csv', index_col=0, header=[0, 1])

    for col in inflows.columns:
        node = col[0]
        tec = col[1]
        if tec == 'Hydro - Reservoir (Energy)':
            data.read_hydro_natural_inflow(node, 'Storage_PumpedHydro_Reservoir',
                                           inflows[col].tolist())
        elif tec == 'Hydro - Pump Storage Open Loop (Energy)':
            data.read_hydro_natural_inflow(node, 'Storage_PumpedHydro_Open', inflows[col].tolist())

    return data


def define_demand(settings, nodes, data):

    climate_year = settings.climate_year
    data_path = settings.data_path + 'demand/'

    demand_el = pd.read_csv(data_path + 'TotalDemand_NT_' + str(climate_year) + '.csv', index_col=0)
    for node in nodes.onshore_nodes:
        data.read_demand_data(node, 'electricity', demand_el[node].to_numpy())

    return data


def define_imports_exports(settings, nodes, data):

    if settings.test == 1:
        data_path = settings.data_path + 'import_export/ImportExport_unlimited.xlsx'
    else:
        data_path = settings.data_path + 'import_export/ImportExport_realistic.xlsx'

    import_export = pd.read_excel(data_path, index_col=0)

    carbontax = 80

    # IMPORT/EXPORT PRICES
    import_carrier_price = {'gas': 40,
                            'electricity': 1000
                            }
    export_carrier_price = {'hydrogen': import_carrier_price['gas'] + carbontax * 0.18,
                            }

    for node in nodes.all.keys():
        for car in import_carrier_price:
            data.read_import_price_data(node, car, np.ones(len(data.topology.timesteps)) * import_carrier_price[car])
        for car in export_carrier_price:
            data.read_export_price_data(node, car, np.ones(len(data.topology.timesteps)) * export_carrier_price[car])

    for node in nodes.all.keys():
        for car in import_carrier_price:
            data.read_import_limit_data(node, car,
                                        np.ones(len(data.topology.timesteps)) * import_export['Import_'+car][node])
        for car in export_carrier_price:
            data.read_export_limit_data(node, car,
                                        np.ones(len(data.topology.timesteps)) * import_export['Export_' + car][node])

    # Emission Factor
    import_emissions = {'electricity': 0.3}
    for node in nodes.onshore_nodes:
        for car in import_emissions:
            data.read_import_emissionfactor_data(node, car, np.ones(len(data.topology.timesteps)) * import_emissions[car])

    export_emissions = {'hydrogen': -0.18}
    for node in nodes.onshore_nodes:
        for car in export_emissions:
            data.read_export_emissionfactor_data(node, car, np.ones(len(data.topology.timesteps)) * export_emissions[car])

    # Emission Price
    data.read_carbon_price_data(carbontax * np.ones(len(data.topology.timesteps)), 'tax')

    return data


def define_charging_efficiencies(settings, nodes, data):
    data_path = settings.data_path

    new_tecs = pd.read_csv(data_path + 'installed_capacities/capacities_node.csv',
                           index_col=0)

    for node in nodes.onshore_nodes:
        new_at_node = \
        new_tecs[new_tecs['Node'] == node][['Technology', 'Capacity our work']].set_index('Technology').to_dict()[
            'Capacity our work']

        charging = {
            'Storage_PumpedHydro_Closed': round(new_at_node.get('Hydro - Pump Storage Closed Loop (Pumping)', 0), 0),
            'Storage_PumpedHydro_Open': round(new_at_node.get('Hydro - Pump Storage Open Loop (Pumping)', 0), 0),
            'Storage_PumpedHydro_Reservoir': 0,
        }

        discharging = {
            'Storage_PumpedHydro_Closed': round(new_at_node.get('Hydro - Pump Storage Closed Loop (Turbine)', 0), 0),
            'Storage_PumpedHydro_Open': round(new_at_node.get('Hydro - Pump Storage Open Loop (Turbine)', 0), 0),
            'Storage_PumpedHydro_Reservoir': round(new_at_node.get('Hydro - Reservoir (Turbine)', 0), 0),
            }

        capacity = {
            'Storage_PumpedHydro_Closed': round(new_at_node.get('Hydro - Pump Storage Closed Loop (Energy)', 0), 0),
            'Storage_PumpedHydro_Open': round(new_at_node.get('Hydro - Pump Storage Open Loop (Energy)', 0), 0),
            'Storage_PumpedHydro_Reservoir': round(new_at_node.get('Hydro - Reservoir (Energy)', 0), 0),
            }

        storage_tecs_at_node = {k: v for k,v in capacity.items() if v > 0}
        for storage in storage_tecs_at_node:
            data.technology_data[node][storage + '_existing'].fitted_performance.coefficients['charge_max'] = \
            -charging[storage]/capacity[storage]
            data.technology_data[node][storage + '_existing'].fitted_performance.coefficients['discharge_max'] = \
            discharging[storage]/capacity[storage]

    return data


def define_new_technologies(settings, nodes, topology):

    data_path = settings.data_path

    new_tecs = pd.read_excel(data_path + 'new_technologies/NewTechnologies.xlsx', index_col=0, sheet_name='NewTechnologies')
    stage = settings.new_technologies_stage

    if not stage == None:
        for node in nodes.all.keys():
            if not isinstance(new_tecs[stage][node], float):
                new_technologies = new_tecs[stage][node].split(', ')
                topology.define_new_technologies(node, new_technologies)

    return topology


def define_configuration():
    # Configuration
    configuration = ModelConfiguration()
    configuration.solveroptions.solver = 'gurobi_persistent'
    configuration.solveroptions.mipgap = 0.02
    configuration.solveroptions.lpwarmstart = 2
    configuration.solveroptions.numericfocus = 3
    configuration.optimization.save_log_files = 1
    configuration.optimization.monte_carlo.on = 0
    configuration.optimization.monte_carlo.N = 5
    configuration.optimization.typicaldays.N = 0
    configuration.solveroptions.timelim = 50

    configuration.solveroptions.intfeastol = 1e-3
    configuration.solveroptions.feastol = 1e-3
    configuration.solveroptions.numericfocus = 3
    configuration.optimization.objective = 'pareto'
    configuration.optimization.pareto_points = 4

    return configuration

def write_to_technology_data(settings):
    data_path = settings.data_path
    year = settings.year
    tec_data_path = settings.tec_data_path

    financial_data = pd.read_excel(data_path + 'cost_technologies/TechnologyCost.xlsx', sheet_name='ToModel', skiprows=1)
    financial_data = financial_data[financial_data['Year'] == year]

    for filename in os.listdir(tec_data_path):
        with open(os.path.join(tec_data_path, filename), 'r') as openfile:
            # Reading from json file
            tec_data = json.load(openfile)

        new_financial_data = financial_data[financial_data['Technology'] == filename.replace('.json', '')]
        tec_data['Economics']['unit_CAPEX'] = float(round(new_financial_data['Investment Cost'].values[0],2))
        tec_data['Economics']['OPEX_variable'] = float(round(new_financial_data['OPEX Variable'].values[0],3))
        tec_data['Economics']['OPEX_fixed'] = float(round(new_financial_data['OPEX Fixed'].values[0],3))
        tec_data['Economics']['lifetime'] = float(round(new_financial_data['Lifetime'].values[0],0))
        tec_data['TechnologyPerf']['emission_factor'] = float(round(new_financial_data['Emission factor'].values[0],3))
        if 'performance' in tec_data['TechnologyPerf']:
            performance_parameters = {'eta_in': 'Charging Efficiency', 'eta_out': 'Discharging Efficiency', 'lambda': 'Lambda'}
            for para in performance_parameters.keys():
                if para in tec_data['TechnologyPerf']['performance']:
                    tec_data['TechnologyPerf']['performance'][para] = float(
                        round(new_financial_data[performance_parameters[para]].values[0], 3))
            if 'out' in tec_data['TechnologyPerf']['performance']:
                if tec_data['tec_type'] == 'CONV1':
                    tec_data['TechnologyPerf']['performance']['out'] = [0, float(
                            round(new_financial_data['Conv. Efficiency'].values[0], 3))]
                else:
                    for car in tec_data['TechnologyPerf']['performance']['out']:
                        tec_data['TechnologyPerf']['performance']['out'][car] = [0, float(
                            round(new_financial_data['Conv. Efficiency'].values[0], 3))]




        with open(os.path.join(tec_data_path, filename), 'w') as outfile:
            json.dump(tec_data, outfile, indent=2)


def write_to_network_data(settings):
    data_path = settings.data_path
    year = settings.year
    netw_data_path = settings.netw_data_path

    financial_data = pd.read_excel(data_path + 'cost_networks/NetworkCost.xlsx', sheet_name='ToModel', skiprows=1)
    financial_data = financial_data[financial_data['Year'] == year]

    for filename in os.listdir(netw_data_path):
        if filename.replace('.json', '') in financial_data['Network'].to_list():
            with open(os.path.join(netw_data_path, filename), 'r') as openfile:
                # Reading from json file
                netw_data = json.load(openfile)

            new_financial_data = financial_data[financial_data['Network'] == filename.replace('.json', '')]
            netw_data['Economics']['gamma1'] = float(round(new_financial_data['gamma1'].values[0],2))
            netw_data['Economics']['gamma2'] = float(round(new_financial_data['gamma2'].values[0],2))
            netw_data['Economics']['gamma3'] = float(round(new_financial_data['gamma3'].values[0],2))
            netw_data['Economics']['gamma4'] = float(round(new_financial_data['gamma4'].values[0],2))
            netw_data['Economics']['OPEX_variable'] = float(round(new_financial_data['OPEX Variable'].values[0],3))
            netw_data['Economics']['OPEX_fixed'] = float(round(new_financial_data['OPEX Fixed'].values[0],3))
            netw_data['Economics']['lifetime'] = float(round(new_financial_data['Lifetime'].values[0],0))
            netw_data['NetworkPerf']['loss'] = float(round(new_financial_data['loss'].values[0],8))
            netw_data['NetworkPerf']['rated_capacity'] = float(round(new_financial_data['rated power'].values[0],0))

            with open(os.path.join(netw_data_path, filename), 'w') as outfile:
                json.dump(netw_data, outfile, indent=2)