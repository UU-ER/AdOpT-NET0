import src.data_management as dm
import pandas as pd
from types import SimpleNamespace
from cases.NorthSea_helpers.read_input_data import *
import copy
from src.model_configuration import ModelConfiguration

class Settings():

    def __init__(self):
        self.year = 2030
        self.scenario = 'GA'
        self.climate_year = 2009
        self.start_date = '01-01 00:00'
        self.end_date = '01-01 23:00'
        self.data_path = r'./cases/NorthSea_v3'
        self.save_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20230614/MES_NS_Benchmark'

        self.node_aggregation_type = {
            'onshore': ['onNL_C', 'onOther', 'onNL_NE', 'onNL_SW', 'onNL_NW'],
            'offshore': []}
        self.node_aggregation = {
            'onNL_C': ['onNL_SE', 'onNL_CE', 'onNL_E'],
            'onNL_NE': ['onNL_NE', 'ofNL_GE_A', 'ofNL_GE_B'],
            'onNL_SW': ['onNL_SW', 'ofNL_BO_A', 'ofNL_BO_B'],
            'onNL_NW': ['onNL_NW', 'ofNL_LU', 'ofNL_PA', 'ofNL_EG'],
            'onOther': ['onBE', 'onDE', 'onDKW', 'onNOS']}

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

    node_data = data_path + '/Nodes/Nodes.xlsx'
    node_list = pd.read_excel(node_data, sheet_name='Nodes_used')
    nodes.onshore_nodes = node_list[node_list['Type'] == 'onshore']['Node'].values.tolist()
    nodes.offshore_nodes = node_list[node_list['Type'] == 'offshore']['Node'].values.tolist()
    nodes.all = node_list['Node'].values.tolist()

    # Remove aggregated nodes
    for node in settings.node_aggregation:
        nodes.all = [x for x in nodes.all if x not in settings.node_aggregation[node]]
        nodes.onshore_nodes = [x for x in nodes.onshore_nodes if x not in settings.node_aggregation[node]]
        nodes.offshore_nodes = [x for x in nodes.offshore_nodes if x not in settings.node_aggregation[node]]

    # Add new nodes
    nodes.all.extend(list(settings.node_aggregation.keys()))
    nodes.onshore_nodes.extend(list(settings.node_aggregation_type['onshore']))
    nodes.onshore_nodes.extend(list(settings.node_aggregation_type['offshore']))

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


def get_installed_capacities(settings, nodes):
    installed_capacities = {}
    node_data = settings.data_path + '/Nodes/Nodes.xlsx'
    node_list = pd.read_excel(node_data, sheet_name='Nodes_used')
    all_onshore_nodes = node_list[node_list['Type'] == 'onshore']['Node'].values.tolist()

    for node in nodes.onshore_nodes:
        if node in settings.node_aggregation:
            installed_capacities[node] = {}

            for node_to_agg in settings.node_aggregation[node]:
                if node_to_agg in all_onshore_nodes:
                    if not installed_capacities[node]:
                        installed_capacities[node] = read_installed_capacity_eraa(node_to_agg, settings.data_path)
                    else:
                        new_cap = read_installed_capacity_eraa(node_to_agg, settings.data_path)
                        for tec_type in installed_capacities[node]:
                            for tec in installed_capacities[node][tec_type]:
                                if tec in new_cap[tec_type]:
                                    if not tec_type == 'HydroStorage_charging':
                                        installed_capacities[node][tec_type][tec] = installed_capacities[node][tec_type][tec] + new_cap[tec_type][tec]
                                    else:
                                        for charging in installed_capacities[node][tec_type][tec]:
                                            installed_capacities[node][tec_type][tec][charging] = \
                                            installed_capacities[node][tec_type][tec][charging] + new_cap[tec_type][tec][charging]
        else:
            installed_capacities[node] = read_installed_capacity_eraa(node, settings.data_path)
    return installed_capacities


def define_installed_capacities(settings, nodes, topology):

    installed_capacities = get_installed_capacities(settings, nodes)
    for node in nodes.onshore_nodes:
        topology.define_existing_technologies(node, installed_capacities[node]['Conventional'])

    return topology


def define_networks(settings, topology):
    """
    Defines the networks
    """
    data_path = settings.data_path

    if settings.networks.existing_electricity:
        # Networks - Existing Electricity
        network_data = read_network_data(topology.nodes, settings.node_aggregation, data_path + '/Networks/NetworkDataElectricity_existing.xlsx', 1)
        topology.define_existing_network('electricityAC', size=network_data['size'], distance=network_data['distance'])

    if settings.networks.new_electricityAC:
        # Networks - New Electricity AC
        network_data = read_network_data(topology.nodes, settings.node_aggregation, data_path + '/Networks/NetworkDataElectricity_AC.xlsx', 0)
        topology.define_new_network('electricityAC_int', connections=network_data['connection'],
                                    distance=network_data['distance'])

    if settings.networks.new_electricityDC:
        # Networks - New Electricity DC
        network_data = read_network_data(topology.nodes, settings.node_aggregation, data_path + '/Networks/NetworkDataElectricity_DC.xlsx', 0)
        topology.define_new_network('electricityDC_int', connections=network_data['connection'],
                                    distance=network_data['distance'])

    if settings.networks.new_hydrogen:
        # Networks - Hydrogen
        network_data = read_network_data(topology.nodes, settings.node_aggregation, data_path + '/Networks/NetworkDataHydrogen.xlsx', 0)
        topology.define_new_network('hydrogenPipeline_int', connections=network_data['connection'],
                                    distance=network_data['distance'])

    return topology


def define_data_handle(topology, nodes):
    data = dm.DataHandle(topology)
    for node in nodes.all:
        data.read_climate_data_from_file(node, r'.\data\climate_data_onshore.txt')

    return data


def define_generic_production(settings, nodes, data):

    data_path = settings.data_path
    node_data = data_path + '/Nodes/Nodes.xlsx'
    node_list = pd.read_excel(node_data, sheet_name='Nodes_used')
    offshore_nodes = node_list[node_list['Type'] == 'offshore']['Node'].values.tolist()

    # all generic production profiles
    profiles_on = pd.read_csv(
        data_path + '\ProductionProfiles\Production_Profiles' + str(settings.climate_year) + '.csv',
        index_col=0)
    profiles_of = calculate_production_profiles_offshore(offshore_nodes)

    profiles = pd.concat([profiles_on, profiles_of], axis=1)

    # Aggregate Nodes
    for node in settings.node_aggregation:
        list_of_nodes = [s + '_tot' for s in settings.node_aggregation[node]]
        profiles[node] = profiles[list_of_nodes].sum(axis=1)

    for node in nodes.all:
        if node + '_tot' in profiles:
            data.read_production_profile(node, 'electricity', profiles[node + '_tot'].to_numpy(), 1)

    return data


def define_hydro_inflow(settings, nodes, data):

    data_path = settings.data_path

    # Hydro Inflow
    reservoir_inflow = pd.read_csv(
        data_path + '\Hydro_Inflows\HydroInflowReservoir' + str(settings.climate_year) + '.csv', index_col=0)

    # Aggregate Nodes
    for node in settings.node_aggregation:
        list_of_nodes = [s + '_tot' for s in settings.node_aggregation[node]]
        list_of_nodes = [value for value in list_of_nodes if value in list(reservoir_inflow.columns)]
        reservoir_inflow[node] = reservoir_inflow[list_of_nodes].sum(axis=1)

    opencycle_inflow = pd.read_csv(
        data_path + '\Hydro_Inflows\HydroInflowPump storage - Open Loop' + str(settings.climate_year) + '.csv',
        index_col=0)

    # Aggregate Nodes
    for node in settings.node_aggregation:
        list_of_nodes = [s + '_tot' for s in settings.node_aggregation[node]]
        list_of_nodes = [value for value in list_of_nodes if value in list(opencycle_inflow.columns)]
        opencycle_inflow[node] = opencycle_inflow[list_of_nodes].sum(axis=1)

    for node in nodes.all:
        if node in reservoir_inflow:
            data.read_hydro_natural_inflow(node, 'Storage_PumpedHydro_Reservoir',
                                           reservoir_inflow[node].values.tolist())
            data.read_hydro_natural_inflow(node, 'Storage_PumpedHydro_Open', opencycle_inflow[node].values.tolist())

    return data


def define_demand(settings, nodes, data):

    data_path = settings.data_path

    demand_el = read_demand_data_eraa(settings.scenario, settings.year, settings.climate_year, data_path + '/Demand_Electricity')
    for node in settings.node_aggregation:
        node_list_with_demand = [x for x in settings.node_aggregation[node] if x in demand_el]
        demand_el[node] = demand_el[node_list_with_demand].sum(axis=1)
    for node in nodes.all:
        if node in demand_el:
            data.read_demand_data(node, 'electricity', demand_el[node].to_numpy())

    demand_h2 = read_demand_data_eraa(settings.scenario, settings.year, settings.climate_year, data_path + '/Demand_Hydrogen')
    for node in settings.node_aggregation:
        node_list_with_demand = [x for x in settings.node_aggregation[node] if x in demand_el]
        demand_h2[node] = demand_h2[node_list_with_demand].sum(axis=1)
    for node in nodes.all:
        if node in demand_h2:
            data.read_demand_data(node, 'hydrogen', demand_h2[node].to_numpy() / 100)

    return data


def define_imports(settings, nodes, data):

    data_path = settings.data_path

    # IMPORT PRICES
    import_carrier_price = {'gas': 180,
                            'electricity':100000,
                            'hydrogen': 200
                            }
    for node in nodes.onshore_nodes:
        for car in import_carrier_price:
            data.read_import_price_data(node, car, np.ones(len(data.topology.timesteps)) * import_carrier_price[car])

    # IMPORT LIMITS
    import_limit = {'gas': 1000000,
                    'hydrogen': 100000
                    }

    for node in nodes.onshore_nodes:
        for car in import_limit:
            data.read_import_limit_data(node, car, np.ones(len(data.topology.timesteps)) * import_limit[car])

    # Electricity
    import_limit = pd.read_excel(data_path + '/Networks/ImportLimits.xlsx', index_col=0, sheet_name='ToPython')
    factor = 10

    car = 'electricity'
    for node in settings.node_aggregation:
        import_limit.at[node, car] = import_limit[car][settings.node_aggregation[node]].sum()

    for node in nodes.onshore_nodes:
        data.read_import_limit_data(node, car, np.ones(len(data.topology.timesteps)) * import_limit[car][node] * factor)

    # Emission Factor
    import_emissions = {'electricity': 0.3,
                            'hydrogen': 0.183/0.6 * 0.2
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
    configuration.optimization.objective = 'pareto'
    configuration.optimization.pareto_points = 8

    return configuration