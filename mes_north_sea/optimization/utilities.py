# import src.data_management as dm
import pandas as pd
import numpy as np
from types import SimpleNamespace
import copy
# from src.model_configuration import ModelConfiguration
import os
import json
import adopt_net0 as adopt
from pathlib import Path


class Settings():

    def __init__(self, test):
        self.test = test
        self.year = 2030
        self.scenario = 'NT'
        self.climate_year = 2008
        self.simplify_networks = 1
        self.only_belgium = 0
        self.validation = 0
        self.test_nodes = 0
        self.model_h2 = 1

        if test:
            self.start_date = '05-01 00:00'
            self.end_date = '05-01 05:00'
        else:
            self.start_date = '01-01 00:00'
            self.end_date = '12-31 23:00'

        self.data_path = Path(__file__).resolve().parent.parent / 'clean_data'
        self.save_path = ''
        self.tec_data_path = self.data_path / 'technology_data'
        self.netw_data_path = self.data_path / 'network_data'
        self.variable_h2_demand = 0
        self.co2_tax = None

        self.node_aggregation_type = {
            'onshore': [],
            'offshore': []}
        self.node_aggregation = {}

        self.new_technologies_stage = None

        # Demand
        self.demand_factor = 1


def write_to_technology_data(settings):
    data_path = settings.data_path
    year = 2030
    tec_data_path = settings.tec_data_path

    financial_data = pd.read_excel(data_path / 'cost_technologies/TechnologyCost.xlsx', sheet_name='ToModel', skiprows=1)
    financial_data = financial_data[financial_data['Year'] == year]

    for filename in os.listdir(tec_data_path):
        with open(os.path.join(tec_data_path, filename), 'r') as openfile:
            # Reading from json file
            tec_data = json.load(openfile)

        tec = filename.replace('.json', '')
        tec = tec.replace('_noh2', '')

        new_financial_data = financial_data[financial_data['Technology'] == tec]
        tec_data['Economics']['unit_capex'] = float(round(new_financial_data['Investment Cost'].values[0],2))
        tec_data['Economics']['opex_variable'] = float(round(new_financial_data['OPEX Variable'].values[0],3))
        tec_data['Economics']['opex_fixed'] = float(round(new_financial_data['OPEX Fixed'].values[0],3))
        tec_data['Economics']['lifetime'] = float(round(new_financial_data['Lifetime'].values[0],0))
        tec_data['Performance']['emission_factor'] = float(round(new_financial_data['Emission factor'].values[0],3))
        if 'performance' in tec_data['Performance']:
            performance_parameters = {'eta_in': 'Charging Efficiency', 'eta_out': 'Discharging Efficiency', 'lambda': 'Lambda'}
            for para in performance_parameters.keys():
                if para in tec_data['Performance']['performance']:
                    tec_data['Performance']['performance'][para] = float(
                        new_financial_data[performance_parameters[para]].values[0])
            if 'out' in tec_data['Performance']['performance']:
                if tec_data['tec_type'] == 'CONV1':
                    tec_data['Performance']['performance']['out'] = [0, float(
                            round(new_financial_data['Conv. Efficiency'].values[0], 3))]
                else:
                    for car in tec_data['Performance']['performance']['out']:
                        tec_data['Performance']['performance']['out'][car] = [0, float(
                            round(new_financial_data['Conv. Efficiency'].values[0], 3))]




        with open(os.path.join(tec_data_path, filename), 'w') as outfile:
            json.dump(tec_data, outfile, indent=2)


def write_to_network_data(settings):
    data_path = settings.data_path
    year = settings.year
    netw_data_path = settings.netw_data_path

    financial_data = pd.read_excel(data_path / 'cost_networks/NetworkCost.xlsx', sheet_name='ToModel', skiprows=1)
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
            netw_data['Economics']['opex_variable'] = float(round(new_financial_data['OPEX Variable'].values[0],3))
            netw_data['Economics']['opex_fixed'] = float(round(new_financial_data['OPEX Fixed'].values[0],3))
            netw_data['Economics']['lifetime'] = float(round(new_financial_data['Lifetime'].values[0],0))
            netw_data['Performance']['loss'] = float(round(new_financial_data['loss'].values[0],8))
            netw_data['Performance']['rated_capacity'] = float(round(new_financial_data['rated power'].values[0],0))

            with open(os.path.join(netw_data_path, filename), 'w') as outfile:
                json.dump(netw_data, outfile, indent=2)


def read_nodes(settings):
    """
    Reads onshore and offshore nodes from file
    """

    data_path = settings.data_path
    nodes = SimpleNamespace()

    if settings.year == 2030:
        node_data = data_path / "nodes" / "nodes.xlsx"
    elif settings.year == 2040:
        node_data = data_path  / "nodes" / "nodes_2040.xlsx"

    if settings.test_nodes:
        node_data = data_path  / "nodes" / "nodes_test.xlsx"


    node_list = pd.read_excel(node_data, sheet_name='Nodes_used')
    nodes.onshore_nodes = node_list[node_list['Type'] == 'onshore']['Node'].values.tolist()
    nodes.offshore_nodes = node_list[node_list['Type'].apply(lambda x: x.startswith('offshore'))]['Node'].values.tolist()
    nodes.storage_nodes = node_list[node_list['Type'] == 'New_CO2_Storage_2040']['Node'].values.tolist()
    nodes.all = {}
    for row in node_list.iterrows():
        node_data = {}
        node_data['lon'] = row[1]['x']
        node_data['lat'] = row[1]['y']
        nodes.all[row[1]['Node']] = node_data

    return nodes


def define_topology(settings, input_data_path, nodes):
    with open(input_data_path / "Topology.json", "r") as json_file:
        topology = json.load(json_file)
    # Nodes
    topology["nodes"] = list(nodes.all.keys())
    # Carriers:
    if settings.model_h2:
        topology["carriers"] = ['electricity', 'gas', 'hydrogen']
    else:
        topology["carriers"] = ['electricity', 'gas', 'CO2captured', 'heat']

    if settings.year == 2040:
        topology["start_date"] = str(2041) + "-" + settings.start_date
        topology["end_date"] = str(2041) + "-" + settings.end_date
    else:
        topology["start_date"] = str(settings.year) + "-" + settings.start_date
        topology["end_date"] = str(settings.year) + "-" + settings.end_date
    # Investment periods:
    topology["investment_periods"] = ["period1"]
    # Save json template
    with open(input_data_path / "Topology.json", "w") as json_file:
        json.dump(topology, json_file, indent=4)



def define_configuration(input_data_path, settings):
    # Configuration
    with open(input_data_path / "ConfigModel.json", "r") as json_file:
        configuration = json.load(json_file)

    configuration["optimization"]["typicaldays"]["N"]["value"] = 0
    configuration["optimization"]["typicaldays"]["method"]["value"] = 2
    configuration["optimization"]["monte_carlo"]["N"]["value"] = 0
    configuration["optimization"]["monte_carlo"]["sd"]["value"] = 0.5
    configuration["optimization"]["monte_carlo"]["on_what"]["value"] = ["Technologies", "Networks", "Import", "Export"]

    configuration["solveroptions"]["solver"]["value"] = 'gurobi'
    configuration["solveroptions"]["mipgap"]["value"] = 0.02
    configuration["solveroptions"]["lpwarmstart"]["value"] = 0
    configuration["solveroptions"]["numericfocus"]["value"] = 3
    configuration["solveroptions"]["timelim"]["value"] = 7*24
    configuration["solveroptions"]["method"]["value"] = 2
    configuration["solveroptions"]["threads"]["value"] = 30
    configuration["solveroptions"]["crossover"] = {}
    configuration["solveroptions"]["crossover"]["value"] = 0
    configuration["solveroptions"]["nodemethod"] = {}
    configuration["solveroptions"]["nodemethod"]["value"] = -1
    configuration["solveroptions"]["intfeastol"]["value"] = 1e-3
    configuration["solveroptions"]["feastol"]["value"] = 1e-3

    configuration["reporting"]["save_summary_path"]["value"] = "//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/" + str(settings.year) + "_250423"
    configuration["reporting"]["save_path"]["value"] = "//Soliscom.uu.nl/geo/USERS/StaffUsers/6574114/EhubResults/MES NorthSea/" + str(settings.year) + "_250423"

    configuration["scaling"]["scaling_on"]["value"] = 1
    configuration["scaling"]["scaling_factors"]["energy_vars"]["value"] = 1e-2
    configuration["scaling"]["scaling_factors"]["cost_vars"]["value"] = 1e-2
    configuration["scaling"]["scaling_factors"]["objective"]["value"] = 1

    with open(input_data_path / "ConfigModel.json", "w") as json_file:
        json.dump(configuration, json_file, indent=4)


def define_node_locations(input_data_path, nodes):
    node_location = pd.read_csv(input_data_path / "NodeLocations.csv", sep=';', index_col=0, header=0)
    for _, node in node_location.iterrows():
        node.loc["lon"] = nodes.all[node.name]["lon"]
        node.loc["lat"] = nodes.all[node.name]["lat"]
        node.loc["alt"] = 10
    node_location = node_location.reset_index()
    node_location.to_csv(input_data_path / "NodeLocations.csv", sep=';', index=False)


def define_installed_capacities(input_data_path, settings, nodes):
    data_path = settings.data_path
    new_tecs = pd.read_csv(data_path /'installed_capacities/capacities_node.csv',
                           index_col=0)
    for node in nodes.onshore_nodes:
        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "r") as json_file:
            technologies = json.load(json_file)

        new_at_node = \
            new_tecs[new_tecs['Node'] == node][['Technology', 'Capacity our work']].set_index('Technology').to_dict()[
                'Capacity our work']

        if settings.model_h2:
            gas_plant = 'PowerPlant_Gas'
        else:
            gas_plant = 'PowerPlant_Gas_noh2'

        tecs_at_node = {
                        gas_plant: round(new_at_node.get('Gas', 0), 0),
                        'PowerPlant_Nuclear': round(new_at_node.get('Nuclear', 0), 0),
                        'PowerPlant_Oil': round(new_at_node.get('Oil', 0), 0),
                        'PowerPlant_Coal': round(new_at_node.get('Coal & Lignite', 0), 0),
                        'Storage_PumpedHydro_Closed': round(
                            new_at_node.get('Hydro - Pump Storage Closed Loop (Energy)', 0),
                            0),
                        'Storage_PumpedHydro_Open': round(new_at_node.get('Hydro - Pump Storage Open Loop (Energy)', 0),
                                                          0),
                        'Storage_PumpedHydro_Reservoir': round(new_at_node.get('Hydro - Reservoir (Energy)', 0), 0),
                        }

        technologies["existing"] = {k: v for k, v in tecs_at_node.items() if v > 0}

        with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "w") as json_file:
            json.dump(technologies, json_file, indent=4)



def define_new_technologies(input_data_path, settings, nodes):

    data_path = settings.data_path

    if settings.year == 2030:
        new_tecs = pd.read_excel(data_path /'new_technologies/NewTechnologies.xlsx', index_col=0,
                                 sheet_name='NewTechnologies')

    elif settings.year == 2040:
        new_tecs = pd.read_excel(data_path /'new_technologies/NewTechnologies_2040.xlsx', index_col=0,
                                 sheet_name='NewTechnologies')

    stage = settings.new_technologies_stage

    if not stage == None:
        for node in nodes.all.keys():
            if node in nodes.storage_nodes:  # skip storage nodes
                continue
            if not isinstance(new_tecs[stage][node], float):
                with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "r") as json_file:
                    technologies = json.load(json_file)

                technologies["new"] = new_tecs[stage][node].split(', ')

                with open(input_data_path / "period1" / "node_data" / node / "Technologies.json", "w") as json_file:
                    json.dump(technologies, json_file, indent=4)

def define_storage(input_data_path, settings, nodes):
    data_path = settings.data_path
    limits = pd.read_csv(data_path / "storage_limits" / "CO2_storage_limits_2040.csv", sep=',',thousands=',')
    limits.columns = limits.columns.str.strip()
    limits = limits.set_index("Node")
    nodes.storage_nodes = limits.index.tolist()

    for node in nodes.storage_nodes:
        tec_data_path = (input_data_path / "period1" / "node_data" / node / "technology_data" / "PermanentStorage_CO2_simple.json")
        if not tec_data_path.exists():
            continue
        with open(tec_data_path, "r") as f:
            tec_data = json.load(f)
        if node in limits.index:
            tec_data["size_max"] = float(limits.loc[node, "size_max"])
            if "Flexibility" not in tec_data:
                tec_data["Flexibility"] = {}
            tec_data["Flexibility"]["injection_rate_max"] = float(
                limits.loc[node, "injection_rate_max"])

        with open(tec_data_path, "w") as f:
            json.dump(tec_data, f, indent=2)

def define_networks(input_data_path, settings):
    """
    Defines the networks
    """

    stage = settings.new_technologies_stage
    # CO2 network
    if settings.year == 2030 or settings.year == 2040:
        new_co2_networks = ["CO2_Pipeline"]
    else:
        new_co2_networks = []


    # # H2 network
    # if ('Hydrogen' in stage) or (stage == 'All') or (stage == 'All_RE_offshore_only'):
    #     if stage != 'Hydrogen_H4':
    #         new_h2_networks = ["hydrogenPipelineOffshore", "hydrogenPipelineOnshore_new", "hydrogenPipelineOnshore_re"]
    #     else:
    #         new_h2_networks = []
    # else:
    #     new_h2_networks = []

    # El networks
    new_el_networks = []
    if settings.year == 2040:
        if settings.simplify_networks:
            new_el_networks = ["electricityDC"]
        else:
            new_el_networks = ["electricityDC_int"]

    if ('ElectricityGrid' in stage) or (stage == 'All') or (stage == 'All_wind_offshore_only'):
        if settings.simplify_networks:
            new_el_networks = ["electricityAC", "electricityDC"]
        else:
            new_el_networks = ["electricityAC", "electricityDC_int"]

    with open(input_data_path / "period1" / "Networks.json", "r") as json_file:
        networks = json.load(json_file)
    networks["new"] = new_el_networks + new_co2_networks
    networks["existing"] = ["electricityAC", "electricityDC"]

    with open(input_data_path / "period1" / "Networks.json", "w") as json_file:
        json.dump(networks, json_file, indent=4)


def define_network_topology(input_data_path, settings, nodes):

    data_path = settings.data_path /'networks/'
    stage = settings.new_technologies_stage

    def get_network_data(file_path, nodes):
        network = pd.read_csv(file_path, sep=None, engine='python')

        network_data = {}
        network_data['size_matrix'] = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";", index_col=0).astype(float)
        network_data['distance_matrix'] = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";", index_col=0).astype(float)
        network_data['max_size_matrix'] = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";", index_col=0).astype(float)
        network_data['connection_matrix'] = pd.read_csv(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";", index_col=0).astype(float)
        for idx, row in network.iterrows():
            if (row.node0 in nodes.all.keys()) & (row.node1 in nodes.all.keys()):
                network_data['size_matrix'].loc[row['node0'], row['node1']] = row['s_nom']*1000
                network_data['size_matrix'].loc[row['node1'], row['node0']] = row['s_nom']*1000
                network_data['distance_matrix'].loc[row['node0'], row['node1']] = row['length']
                network_data['distance_matrix'].loc[row['node1'], row['node0']] = row['length']
                network_data['max_size_matrix'].loc[row['node1'], row['node0']] = row['s_nom_max']*1000 - row['s_nom']*1000
                network_data['max_size_matrix'].loc[row['node0'], row['node1']] = row['s_nom_max']*1000 - row['s_nom']*1000
                if row['s_nom_max'] > 0:
                    network_data['connection_matrix'].loc[row['node1'], row['node0']] = 1
                    network_data['connection_matrix'].loc[row['node0'], row['node1']] = 1

        return network_data

    # AC GRIDS
    # Existing AC grid
    file_name_ac = 'pyhub_el_ac_all.csv'
    ac_data = get_network_data(data_path /file_name_ac, nodes)
    os.makedirs(input_data_path / "period1" / "network_topology" / "existing" / "electricityAC", exist_ok=True)
    ac_data['connection_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "electricityAC" / "connection.csv",
        sep=";")
    ac_data['distance_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "electricityAC" / "distance.csv",
        sep=";")
    ac_data['size_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "electricityAC" / "size.csv",
        sep=";")



    if stage == 'ElectricityGrid_on':
        file_name_ac = 'pyhub_el_ac_on.csv'
    elif stage == 'ElectricityGrid_off':
        file_name_ac = 'pyhub_el_ac_off.csv'
    elif stage == 'ElectricityGrid_noBorder':
        file_name_ac = 'pyhub_el_ac_noBorder.csv'
    else:
        file_name_ac = 'pyhub_el_ac_all.csv'

    # New AC grid
    ac_data = get_network_data(data_path /file_name_ac, nodes)
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / "electricityAC", exist_ok=True)
    ac_data['connection_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "electricityAC" / "connection.csv",
        sep=";")
    ac_data['distance_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "electricityAC" / "distance.csv",
        sep=";")
    ac_data['size_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "new" / "electricityAC" / "size_max_arcs.csv",
        sep=";")

    # DC GRIDS
    # Existing DC grid
    if settings.year == 2030:
        file_name_dc = 'pyhub_el_dc_all.csv'
    elif settings.year == 2040:
        file_name_dc = 'pyhub_el_dc_all_2040.csv'
    dc_data = get_network_data(data_path /file_name_dc, nodes)
    os.makedirs(input_data_path / "period1" / "network_topology" / "existing" / "electricityDC", exist_ok=True)
    dc_data['connection_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "electricityDC" / "connection.csv",
        sep=";")
    dc_data['distance_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "electricityDC" / "distance.csv",
        sep=";")
    dc_data['size_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "existing" / "electricityDC" / "size.csv",
        sep=";")

    # New DC grid
    if stage == 'ElectricityGrid_on':
        if settings.year == 2030:
            file_name_dc = 'pyhub_el_dc_on.csv'
        elif settings.year == 2040:
            file_name_dc = 'pyhub_el_dc_on_2040.csv'
    elif stage == 'ElectricityGrid_off':
        if settings.year == 2030:
            file_name_dc = 'pyhub_el_dc_off.csv'
        elif settings.year == 2040:
            file_name_dc = 'pyhub_el_dc_off_2040.csv'
    elif stage == 'ElectricityGrid_noBorder':
        if settings.year == 2030:
            file_name_dc = 'pyhub_el_dc_noBorder.csv'
        elif settings.year == 2040:
            file_name_dc = 'pyhub_el_dc_noBorder_2040.csv'
    else:
        if settings.year == 2030:
            file_name_dc = 'pyhub_el_dc_all.csv'
        elif settings.year == 2040:
            file_name_dc = 'pyhub_el_dc_all_2040.csv'

    dc_data = get_network_data(data_path /file_name_dc, nodes)

    if ('ElectricityGrid' in stage) or (stage == 'All') or (stage == 'All_wind_offshore_only'):
        pass
    else:
        if settings.year == 2040:
            dc_data = get_network_data(data_path /'pyhub_el_dc_re_only_2040.csv', nodes)

    if settings.simplify_networks:
        dc_netw_name = 'electricityDC'
        size_dc = round(dc_data['max_size_matrix'], 0)
    else:
        dc_netw_name = 'electricityDC_int'
        size_dc = round(dc_data['max_size_matrix'] / 2000, 0)

    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / dc_netw_name, exist_ok=True)
    dc_data['connection_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "new" / dc_netw_name / "connection.csv",
        sep=";")
    dc_data['distance_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "new" / dc_netw_name / "distance.csv",
        sep=";")
    size_dc.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / dc_netw_name / "size_max_arcs.csv",
        sep=";")

    # co2 networks onshore and offshore
    if settings.year == 2030:
        file_name_offshore = 'pyhub_co2_offshore.csv'
    elif settings.year == 2040:
        file_name_offshore = 'pyhub_co2_offshore_2040.csv'

    data_offshore = get_network_data(data_path / file_name_offshore, nodes)
    data_onshore = get_network_data(data_path / 'pyhub_co2_onshore_new.csv', nodes)

    merged_connection = data_offshore['connection_matrix'].combine(
        data_onshore['connection_matrix'], func=lambda a, b: (a + b).clip(upper=1)
    )
    merged_distance = data_offshore['distance_matrix'].combine(
        data_onshore['distance_matrix'], func=lambda a, b: a + b
    )

    netw_name = "CO2_Pipeline"
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / netw_name, exist_ok=True)
    merged_connection.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / netw_name / "connection.csv",
        sep=";"
    )
    merged_distance.to_csv(
        input_data_path / "period1" / "network_topology" / "new" / netw_name / "distance.csv",
        sep=";"
    )

    # # H2 NETWORKS
    # # offshore
    # if settings.year == 2030:
    #     file_name = 'pyhub_h2_offshore.csv'
    # elif settings.year == 2040:
    #     file_name = 'pyhub_h2_offshore_2040.csv'
    #
    # data = get_network_data(data_path /file_name, nodes)
    # netw_name = "hydrogenPipelineOffshore"
    # os.makedirs(input_data_path / "period1" / "network_topology" / "new" / netw_name, exist_ok=True)
    # data['connection_matrix'].to_csv(
    #     input_data_path / "period1" / "network_topology" / "new" / netw_name / "connection.csv",
    #     sep=";")
    # data['distance_matrix'].to_csv(
    #     input_data_path / "period1" / "network_topology" / "new" / netw_name / "distance.csv",
    #     sep=";")

    # onshore new
    file_name = 'pyhub_h2_onshore_new.csv'
    data = get_network_data(data_path /file_name, nodes)
    netw_name = "hydrogenPipelineOnshore_new"
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / netw_name, exist_ok=True)
    data['connection_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "new" / netw_name / "connection.csv",
        sep=";")
    data['distance_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "new" / netw_name / "distance.csv",
        sep=";")
    # data['size_matrix'].to_csv(
    #     input_data_path / "period1" / "network_topology" / "new" / netw_name / "size_max_arcs.csv",
    #     sep=";")

    # onshore repurposed
    file_name = 'pyhub_h2_onshore_re.csv'
    data = get_network_data(data_path /file_name, nodes)
    netw_name = "hydrogenPipelineOnshore_re"
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / netw_name, exist_ok=True)
    data['connection_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "new" / netw_name / "connection.csv",
        sep=";")
    data['distance_matrix'].to_csv(
        input_data_path / "period1" / "network_topology" / "new" / netw_name / "distance.csv",
        sep=";")
    # data['size_matrix'].to_csv(
    #     input_data_path / "period1" / "network_topology" / "new" / netw_name / "size_max_arcs.csv",
    #     sep=";")

def define_demand(input_data_path, settings, nodes):

    climate_year = settings.climate_year
    model_year = settings.year

    demand_el = pd.read_csv(settings.data_path / 'demand' / f"TotalDemand_NT_{str(model_year)}_{str(climate_year)}.csv", index_col=0)
    for node in nodes.onshore_nodes:
        demand = pd.DataFrame()
        demand["Demand"] = demand_el[node]
        adopt.fill_carrier_data(input_data_path, value_or_data=demand, columns=['Demand'], carriers=['electricity'], nodes = [node])

def define_generic_production(input_data_path, settings, nodes):
    climate_year = settings.climate_year

    generic_production = pd.read_csv(settings.data_path / "production_profiles_re" / f"production_profiles_re{str(climate_year)}.csv", index_col=0, header=[0, 1])
    generic_production_no2009 = pd.read_csv(settings.data_path / "production_profiles_re" / "production_profiles_re.csv", index_col=0, header=[0, 1])
    for node in nodes.all.keys():
        profile = pd.DataFrame()
        if (node, 'total') in generic_production.columns:
            if settings.validation == 1 and node == "NO1" and climate_year == 2009:
                profile["Generic production"] = generic_production_no2009.loc[:, (node, 'total')].to_numpy().round(1)
            else:
                profile["Generic production"] = generic_production.loc[:, (node, 'total')].to_numpy().round(1)

            adopt.fill_carrier_data(input_data_path, value_or_data=profile, columns=['Generic production'],
                                    carriers=['electricity'], nodes=[node])
        else:
            print(node)

def define_hydro_inflow(input_data_path, settings):
    climate_year = settings.climate_year

    inflows = pd.read_csv(settings.data_path / "hydro_inflows" / f"hydro_inflows{str(climate_year)}.csv", index_col=0, header=[0, 1])

    for col in inflows.columns:
        node = col[0]
        tec = col[1]
        climate_data = pd.read_csv(input_data_path / "period1" / "node_data" / node / "ClimateData.csv", sep=";", index_col=0)

        climate_data[tec] = inflows[col].tolist()[:len(climate_data)]

        if tec == 'Hydro - Reservoir (Energy)':
            climate_data["Storage_PumpedHydro_Reservoir_existing_inflow"] = inflows[col].tolist()[:len(climate_data)]
        elif tec == 'Hydro - Pump Storage Open Loop (Energy)':
            climate_data["Storage_PumpedHydro_Open_existing_inflow"] = inflows[col].tolist()[:len(climate_data)]

        climate_data.to_csv(input_data_path / "period1" / "node_data" / node / "ClimateData.csv", sep=";")

def define_capacity_factors(input_data_path, settings, nodes):
    climate_year = settings.climate_year

    data_path = settings.data_path /'capacity_factors'


    cfs = {}
    if settings.validation:
        cfs["offshore_wind"] = pd.read_csv(data_path / f"wind_offshore'{str(2008)}.csv", index_col=0)
    else:
        cfs["offshore_wind"] = pd.read_csv(data_path / f"wind_offshore{str(climate_year)}.csv", index_col=0)
    cfs["onshore_wind"] = pd.read_csv(data_path / f"wind_onshore{str(climate_year)}.csv", index_col=0)
    cfs["pv"] = pd.read_csv(data_path / f"pv{str(climate_year)}.csv", index_col=0)

    for profile in cfs.keys():
        for node in cfs[profile].columns:
            climate_data = pd.read_csv(input_data_path / "period1" / "node_data" / node / "ClimateData.csv", sep=";", index_col=0)
            climate_data[profile] = cfs[profile][node].to_numpy()[:len(climate_data)]
            climate_data.to_csv(input_data_path / "period1" / "node_data" / node / "ClimateData.csv", sep=";")

    weather_data = pd.read_csv(settings.data_path / "weather_data" / f"weather_{climate_year}.csv", index_col=0)

    for node in nodes.all.keys():
        climate_csv = input_data_path / "period1" / "node_data" / node / "ClimateData.csv"
        climate_data = pd.read_csv(climate_csv, sep=";", index_col=0)
        climate_data['temp_air'] = weather_data[f"{node}_temp"].to_numpy()[:len(climate_data)]
        climate_data['rh'] = weather_data[f"{node}_rh"].to_numpy()[:len(climate_data)]
        climate_data.to_csv(climate_csv, sep=";")

def define_max_renewable_capacities(input_data_path, settings):

    data_path = settings.data_path

    max_caps = pd.read_csv(data_path / "max_re_cap"/ "max_re_nodes.csv", index_col=0)

    for node, caps in max_caps.iterrows():

        tec_data_path = input_data_path / "period1" / "node_data" / node / "technology_data"

        if (tec_data_path / "Onshore_Wind.json").exists():
            with open(os.path.join(tec_data_path, "Onshore_Wind.json"), 'r') as openfile:
                # Reading from json file
                tec_data = json.load(openfile)

            tec_data["size_max"] = min(caps['RemainingPotential_Wind_on'], 100000)

            with open(os.path.join(tec_data_path, "Onshore_Wind.json"), 'w') as outfile:
                json.dump(tec_data, outfile, indent=2)

        if (tec_data_path / "Offshore_Wind.json").exists():
            with open(os.path.join(tec_data_path, "Offshore_Wind.json"), 'r') as openfile:
                # Reading from json file
                tec_data = json.load(openfile)

            tec_data["size_max"] = min(caps['RemainingPotential_Wind_off'], 250000 / 2)

            with open(os.path.join(tec_data_path, "Offshore_Wind.json"), 'w') as outfile:
                json.dump(tec_data, outfile, indent=2)

        if (tec_data_path / "PV.json").exists():
            with open(os.path.join(tec_data_path, "PV.json"), 'r') as openfile:
                # Reading from json file
                tec_data = json.load(openfile)

            tec_data["size_max"] = min(caps['RemainingPotential_PV'], 400000)

            with open(os.path.join(tec_data_path, "PV.json"), 'w') as outfile:
                json.dump(tec_data, outfile, indent=2)


def define_imports_exports(input_data_path, settings, nodes):

    if settings.test == 1:
        if settings.year == 2030:
            data_path = settings.data_path /"import_export"/"ImportExport_unlimited.xlsx"
            carbontax = 80

        elif settings.year == 2040:
            data_path = settings.data_path / "import_export"/"ImportExport_unlimited_2040.xlsx"
            carbontax = settings.co2_tax

    else:
        if settings.year == 2030:
            data_path = settings.data_path / "import_export"/"ImportExport_realistic.xlsx"
            carbontax = 80

        elif settings.year == 2040:
            data_path = settings.data_path /"import_export"/"ImportExport_unlimited_2040.xlsx"
            carbontax = settings.co2_tax

    import_export = pd.read_excel(data_path, index_col=0)


    # IMPORT/EXPORT PRICES
    import_carrier_price = {'gas': 40,
                            'electricity': 1000
                            }
    if settings.model_h2:
        export_carrier_price = {'hydrogen': import_carrier_price['gas'] + carbontax * 0.108,
                            }
    else:
        export_carrier_price = {}

    if settings.variable_h2_demand:
        hydrogen_demand = pd.read_csv(settings.data_path / "demand" / f"HydrogenDemand_NT_{str(settings.climate_year)}.csv", index_col=0)

    for node in nodes.all.keys():
        for car in import_carrier_price:
            adopt.fill_carrier_data(input_data_path, value_or_data=import_carrier_price[car], columns=['Import price'],
                                    carriers=[car], nodes=[node])
            adopt.fill_carrier_data(input_data_path, value_or_data=import_export['Import_'+car][node], columns=['Import limit'],
                                    carriers=[car], nodes=[node])
        for car in export_carrier_price:
            adopt.fill_carrier_data(input_data_path, value_or_data=export_carrier_price[car], columns=['Export price'],
                                    carriers=[car], nodes=[node])
            adopt.fill_carrier_data(input_data_path, value_or_data=import_export['Export_'+car][node], columns=['Export limit'],
                                    carriers=[car], nodes=[node])

        if settings.variable_h2_demand and settings.year==2040:
            demand = pd.DataFrame()
            demand["Export limit"] = hydrogen_demand[node]

            if settings.model_h2:
                adopt.fill_carrier_data(input_data_path, value_or_data=demand, columns=['Export limit'],
                                        carriers=["hydrogen"], nodes=[node])






    # Emission Factors
    import_emissions = {'electricity': 0.8}
    for car in import_emissions:
        adopt.fill_carrier_data(input_data_path, value_or_data=import_emissions[car],
                                columns=['Import emission factor'], carriers=[car])

    export_emissions = {'hydrogen': -0.108}
    for car in export_emissions:
        if settings.model_h2:
            adopt.fill_carrier_data(input_data_path, value_or_data=export_emissions[car],
                                columns=['Export emission factor'], carriers=[car], nodes = nodes.onshore_nodes)

    # Emission Price
    for node in nodes.all.keys():
        carbon_cost_path = input_data_path / "period1" / "node_data" / node / "CarbonCost.csv"
        carbon_cost_template = pd.read_csv(carbon_cost_path, sep=';', index_col=0, header=0)
        carbon_cost_template['price'] = carbontax
        carbon_cost_template = carbon_cost_template.reset_index()
        carbon_cost_template.to_csv(carbon_cost_path, sep=';', index=False)


def define_charging_efficiencies(settings, nodes, m):
    data_path = settings.data_path

    new_tecs = pd.read_csv(data_path /"installed_capacities"/"capacities_node.csv",
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
            if storage == 'Storage_PumpedHydro_Closed':
                m.data.technology_data["period1"][node][storage + '_existing'].processed_coeff.time_independent['charge_rate'] = \
                -charging[storage]/capacity[storage]

                m.data.technology_data["period1"][node][storage + '_existing'].processed_coeff.time_independent['discharge_rate'] = \
                discharging[storage]/capacity[storage]
            else:
                m.data.technology_data["period1"][node][storage + '_existing'].processed_coeff.time_independent[
                    'charge_max'] = \
                    -charging[storage] / capacity[storage]

                m.data.technology_data["period1"][node][storage + '_existing'].processed_coeff.time_independent[
                    'discharge_max'] = \
                    discharging[storage] / capacity[storage]

    return m
