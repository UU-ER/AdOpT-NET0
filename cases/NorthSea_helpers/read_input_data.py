import pandas as pd
import pickle
import numpy as np
from src.data_management.components.fit_technology_performance import perform_fitting_WT
import src.data_management as dm
import json
import os
import copy

def read_demand_data_eraa(scenario, year, climate_year, data_path):
    """
    reads demand data for respective climate year and region from TYNDP dataset

    :param str scenario: scenario to read
    :param int climate_year: climate year to read
    :param int year: year to read
    :return pandas.series demand_profile: demand profile as series
    """
    load_path = data_path + '/Demand_'
    file_name = scenario + '_' + str(year) + '_ClimateYear' + str(climate_year) + '.csv'
    demand_profiles = pd.read_csv(load_path + file_name)

    return demand_profiles

def read_installed_capacity_eraa(region, data_path):
    """
    reads production profiles for respective climate year and region from ERAA dataset (year 2030).

    It adds solar, wind onshore and wind offshore respectively for the whole region.

    :param int climate_year: climate year to read
    :param str region: region to read
    :return dict capacity_factors: production profile as series
    """
    data_path = data_path + '/InstalledCapacity/ERAA_InstalledCapacity.xlsx'

    instcap = pd.read_excel(data_path, index_col= 2, sheet_name='Sheet1')

    Conv = {}
    RE = {}
    Stor = {}
    if instcap[region]['Nuclear'] >0:
        Conv['PowerPlant_Nuclear'] = round(instcap[region]['Nuclear'], 0)

    if instcap[region]['Gas'] > 0:
        Conv['PowerPlant_Gas'] = round(instcap[region]['Gas'], 0)

    if instcap[region]['Hard Coal'] > 0:
        Conv['PowerPlant_Coal'] = round(instcap[region]['Hard Coal'], 0)

    if instcap[region]['SteamReformer'] > 0:
        Conv['SteamReformer'] = round(instcap[region]['SteamReformer'], 0)

    if instcap[region]['Batteries'] > 0:
        Conv['Storage_Battery'] = round(instcap[region]['Batteries'], 0)
        Stor['Storage_Battery'] = {}
        Stor['Storage_Battery']['max_charge'] = round(\
            - instcap[region]['Batteries (Offtake)'] /\
            instcap[region]['Batteries'], 3)
        Stor['Storage_Battery']['max_discharge'] = round(\
            instcap[region]['Batteries (Injection)'] / \
            instcap[region]['Batteries'], 3)

    if instcap[region]['Hydro - Pump Storage Closed Loop'] > 0:
        Conv['Storage_PumpedHydro_Closed'] = round(instcap[region]['Hydro - Pump Storage Closed Loop'], 0)
        Stor['Storage_PumpedHydro_Closed'] = {}
        Stor['Storage_PumpedHydro_Closed']['max_charge'] = round(\
            - instcap[region]['Hydro - Pump Storage Closed Loop (Pumping)'] /\
            instcap[region]['Hydro - Pump Storage Closed Loop'], 3)
        Stor['Storage_PumpedHydro_Closed']['max_discharge'] = round(\
            instcap[region]['Hydro - Pump Storage Closed Loop (Turbine)'] / \
            instcap[region]['Hydro - Pump Storage Closed Loop'], 3)

    if instcap[region]['Hydro - Pump Storage Open Loop'] > 0:
        Conv['Storage_PumpedHydro_Open'] = round(instcap[region]['Hydro - Pump Storage Open Loop'], 0)
        Stor['Storage_PumpedHydro_Open'] = {}
        Stor['Storage_PumpedHydro_Open']['max_charge'] = round(\
            - instcap[region]['Hydro - Pump Storage Open Loop (Pumping)'] /\
            instcap[region]['Hydro - Pump Storage Open Loop'], 3)
        Stor['Storage_PumpedHydro_Open']['max_discharge'] = round(\
            instcap[region]['Hydro - Pump Storage Open Loop (Turbine)'] / \
            instcap[region]['Hydro - Pump Storage Open Loop'], 3)

    if instcap[region]['Hydro - Reservoir'] > 0:
        Conv['Storage_PumpedHydro_Reservoir']= round(instcap[region]['Hydro - Reservoir'], 0)
        Stor['Storage_PumpedHydro_Reservoir'] = {}
        Stor['Storage_PumpedHydro_Reservoir']['max_charge'] = 0
        Stor['Storage_PumpedHydro_Reservoir']['max_discharge'] = round(\
            instcap[region]['Hydro - Reservoir (Turbine)'] / \
            instcap[region]['Hydro - Pump Storage Open Loop'], 3)

    RE['Wind_On'] = round(instcap[region]['Wind Onshore'], 0)
    RE['Wind_Of'] = round(instcap[region]['Wind Offshore'], 0)
    RE['PV'] = round(instcap[region]['Solar (Photovoltaic)'], 0)

    installed_capacities = {}
    installed_capacities['RE'] = RE
    installed_capacities['Conventional'] = Conv
    installed_capacities['HydroStorage_charging'] = Stor

    return installed_capacities


def calculate_production_profiles_offshore(offshore_nodes):
    # Get output from wind parks
    load_path = r'./data/climate_data/'
    data_path = r'./cases/NorthSea_v2/Offshore_Parks/WindParkData.xlsx'
    park_data = pd.read_excel(data_path, sheet_name='Parks')
    park_data_2030 = park_data[park_data.YEAR <= 2030]

    # Output per transformer station (easy)
    output_per_substation = pd.DataFrame()
    for station in offshore_nodes:
        parks_at_station = park_data_2030[park_data_2030['Transformer Platform'] == station]
        output = np.zeros(8760)
        for idx in parks_at_station.index:
            name = parks_at_station['NAME'][idx]
            name = name.replace('(', '')
            name = name.replace(')', '')
            name = name.replace(' ', '_')
            load_file = load_path + name

            with open(load_file, 'rb') as handle:
                data = pickle.load(handle)

            cap_factors = perform_fitting_WT(data['dataframe'], 'WindTurbine_Offshore_11000', 0)
            cap_factors = cap_factors.coefficients['capfactor']
            output = output + cap_factors * parks_at_station['POWER_MW'][idx]
        output_per_substation[station + '_tot'] = output

    return output_per_substation


def read_network_data(nodes, aggregation, data_file, existing):
    def get_matrix(data_from_excel, type):

        result_matrix = dm.create_empty_network_matrix(nodes)

        # set all equal
        for node1 in nodes_to_read:
            for node2 in nodes_to_read:
                if pd.isna(data_from_excel[node1][node2]) and not pd.isna(data_from_excel[node2][node1]):
                    data_from_excel[node1][node2] = data_from_excel[node2][node1]
                if not pd.isna(data_from_excel[node1][node2]) and pd.isna(data_from_excel[node2][node1]):
                    data_from_excel[node2][node1] = data_from_excel[node1][node2]

        for aggregate_node in aggregation.keys():
            if (type == 'size') or (type == 'connection'):
                a = data_from_excel.loc[aggregation[aggregate_node]].sum()
            elif type == 'distance' :
                a = data_from_excel.loc[aggregation[aggregate_node]].mean()
            data_from_excel = data_from_excel.append(pd.DataFrame([a], index=[aggregate_node], columns=data_from_excel.columns))
            data_from_excel[aggregate_node] = a
            data_from_excel = data_from_excel[~data_from_excel.index.duplicated(keep='first')]

        # Read data for not aggregated nodes
        # not_aggregated = [x for x in nodes if x not in list(aggregation.keys())]
        for node1 in nodes:
            for node2 in nodes:
                if node1 == node2:
                    data_from_excel.at[node1, node2] = 0
                    data_from_excel.at[node2, node1] = 0
                if not data_from_excel.at[node1, node2] == data_from_excel.at[node2, node1]:
                    print(data_from_excel.at[node1, node2] - data_from_excel.at[node2, node1])
                    print(node2)

        for node1 in nodes:
            for node2 in nodes:

                if pd.isna(data_from_excel[node1][node2]):
                    result_matrix.at[node1, node2] = 0
                    result_matrix.at[node2, node1] = 0
                else:
                    if not type == 'connection':
                        result_matrix.at[node1, node2] = data_from_excel[node1][node2]
                        result_matrix.at[node2, node1] = data_from_excel[node1][node2]
                    else:
                        if data_from_excel[node1][node2] > 0:
                            result_matrix.at[node1, node2] = 1
                            result_matrix.at[node2, node1] = 1
                        else:
                            result_matrix.at[node1, node2] = 0
                            result_matrix.at[node2, node1] = 0

        return result_matrix

    data = {}

    nodes_to_read = copy.deepcopy(nodes)
    for aggregate_node in aggregation:
        if aggregate_node in nodes_to_read:
            nodes_to_read.remove(aggregate_node)
        nodes_to_read.extend(aggregation[aggregate_node])

    if existing:
        size_from_excel = pd.read_excel(data_file, index_col=0, sheet_name='NetworkSize')
        data['size'] = get_matrix(size_from_excel, 'size')
    else:
        connection_from_excel = pd.read_excel(data_file, index_col=0, sheet_name='NetworkConnection')
        data['connection'] = get_matrix(connection_from_excel, 'connection')

    distance_from_excel = pd.read_excel(data_file, index_col=0, sheet_name='NetworkLength')
    data['distance'] = get_matrix(distance_from_excel, 'distance')

    return data


def write_to_technology_data(tec_data_path, year):
    tec_data_path = r'./cases/NorthSea_v2/Technology_Data/'
    financial_data_path = r'./cases/NorthSea_v2/Cost_Technologies/TechnologyCost.xlsx'

    financial_data = pd.read_excel(financial_data_path, sheet_name='ToModel' ,skiprows=1)
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

        with open(os.path.join(tec_data_path, filename), 'w') as outfile:
            json.dump(tec_data, outfile, indent=2)



