import pandas as pd
import numpy as np
from src.data_management.components.fit_technology_performance import *
from types import SimpleNamespace

def load_nodes():
    """Loads node definition of model from file"""
    path = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/DataPreprocessing/00_CleanData/Nodes/nodes.xlsx'
    nodes = pd.read_excel(path, sheet_name='Nodes_all')
    nodes = nodes.set_index('Node')
    return nodes

def aggregate_offshore_wind_output(wind_parks_at_node, year):
    output = np.zeros(8760)
    path_load_profile = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/DataPreprocessing/00_CleanData/OffshoreWindFarmProfiles/'
    # Loop through each wind park
    for index, wind_park in wind_parks_at_node.iterrows():
        name = wind_park['NAME']
        capacity = wind_park['POWER_MW']
        cf = np.loadtxt(path_load_profile + name.replace("/", "-") + '_' + str(year) + '.csv', delimiter=',')
        cf = cf[0:8760]
        output = output + cf * capacity * 0.95
    return output

def aggregate_offshore_wind_per_node(node, year):
    path_wind_park_data = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/WindFarms_v3.csv'

    # Load wind park data and power curves to dataframe
    wind_park_data = pd.read_csv(path_wind_park_data, delimiter=',')
    wind_park_data = wind_park_data[wind_park_data['NODE_2'] == node]

    output = aggregate_offshore_wind_output(wind_park_data, year)

    return output


def get_installed_capacities_no():
    nodes_norway = ['NOM1', 'NON1', 'NOS0']
    installed_capacities_raw = pd.read_excel(path + '00_Summaries/CapacitiesERAA.xlsx',
                                             sheet_name='Tabelle1',
                                             skiprows=[0])
    installed_capacities = {}
    for node_norway in nodes_norway:
        installed_capacities[node_norway] = {}
        installed_capacities[node_norway]['PV'] = installed_capacities_raw[node_norway][18]
        installed_capacities[node_norway]['Wind_on'] = installed_capacities_raw[node_norway][15]

    return installed_capacities

def get_production_profile_no(installed_capacities_no, climate_year):
    cf_load_path_wind = 'E:/00_Data/00_RenewableGeneration/ENTSOE_ERAA/Wind onshore/PECD_Wind_Onshore_2030_edition 2022.1.xlsx'
    cf_load_path_PV = 'E:/00_Data/00_RenewableGeneration/ENTSOE_ERAA/Solar/PECD_LFSolarPV_2030_edition 2022.1.xlsx'
    production_profile = pd.DataFrame()
    production_profile['PV'] = np.zeros(8760)
    production_profile['Wind_on'] = np.zeros(8760)
    production_profile['Wind_off'] = np.zeros(8760)
    production_profile['Biomass'] = np.zeros(8760)
    production_profile['RunOfRiver'] = np.zeros(8760)
    for node in installed_capacities_no:
        cf_PV = pd.read_excel(cf_load_path_PV, sheet_name=node, skiprows=10)
        cf_PV = cf_PV.fillna(0)
        cf_PV.columns = cf_PV.columns.astype(str)
        cf_wind_on = pd.read_excel(cf_load_path_wind, sheet_name=node, skiprows=10)
        cf_wind_on = cf_wind_on.fillna(0)
        cf_wind_on.columns = cf_wind_on.columns.astype(str)
        production_profile['PV'] = production_profile['PV'] + installed_capacities_no[node]['PV'] * cf_PV[str(climate_year)]
        production_profile['Wind_on'] = production_profile['Wind_on'] + installed_capacities_no[node]['Wind_on'] * cf_wind_on[str(climate_year)]

    return production_profile

def read_run_of_river_production(node, climate_year):
    data_path = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/DataPreprocessing/00_CleanData/'
    columns = [i for i in range(16,16+37)]
    column_names = ['Week', *range(1982, 2018)]
    new_tecs = pd.read_excel(data_path + '/InstalledCapacities_nonRE/InstalledCapacities_nonRE.xlsx',
                             sheet_name='Capacities at node',
                             index_col=0)
    #determine share of runofriver capacity at node
    country = new_tecs['Country'][node]
    share = new_tecs['Hydro river'][node] / new_tecs.groupby('Country').sum()['Hydro river'][country]

    #read inflow in respective climate year
    def divide_dataframe(df, n):
        # Divide each entry by 24
        divided_df = df / n

        # Concatenate the dataframe
        concatenated_df = pd.DataFrame(divided_df.values.repeat(n, axis=0))

        return concatenated_df

    run_of_river_output = np.zeros(8760)
    if country != 'DK':
        if country != 'NO':
            data_path = r'E:/00_Data/00_RenewableGeneration/ENTSOE_ERAA/Hydro Inflows/PEMMDB_' + country + '00_Hydro Inflow_2030.xlsx'
            temp = pd.read_excel(data_path, sheet_name='Run of River', skiprows=12, usecols=columns, names=column_names)
            temp = divide_dataframe(temp[climate_year][0:365], 24) * 1000 * share
            temp = temp.fillna(0)
            run_of_river_output = run_of_river_output + temp.to_numpy().flatten()
        else:
            nodes_norway = ['NOM1', 'NON1', 'NOS0']
            for node_no in nodes_norway:
                data_path = r'E:/00_Data/00_RenewableGeneration/ENTSOE_ERAA/Hydro Inflows/PEMMDB_' + node_no + '_Hydro Inflow_2030.xlsx'
                temp = pd.read_excel(data_path, sheet_name='Run of River', skiprows=12, usecols=columns, names=column_names)
                temp = divide_dataframe(temp[climate_year][0:365], 24) * 1000 * share
                temp = temp.fillna(0)
                run_of_river_output = run_of_river_output + temp.to_numpy().flatten()

    return run_of_river_output

path = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/' \
             'DOSTA - HydrogenOffshore/DataPreprocessing/00_CleanData/'

save_path = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/DataPreprocessing/00_CleanData/ProductionProfiles_RE/'
path_biomass = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/PowerPlants/nonREinstalledNUTS2_v2.xlsx'
capacity_biomass = pd.read_excel(path_biomass, sheet_name='Capacities at node')


# read node files NUTS2
nodes_path = path + 'Nodes/nuts2nodes.csv'
nuts_nodes = pd.read_csv(nodes_path)
all_nodes = load_nodes()

onshore_nodes = all_nodes[all_nodes['Type'] == 'onshore'].index
offshore_nodes = all_nodes[all_nodes['Type'].str.startswith('offshore')].index

# get capacities at node
capacity_path = path + 'InstalledCapacities_RE/InstalledCapacities_RE.csv'
capacities = pd.read_csv(capacity_path)

# merge the two tables:
capacities = pd.merge(nuts_nodes, capacities, on='NUTS_ID')
capacities = capacities[['NUTS_ID', 'Node', 'lon', 'lat', 'Capacity_PV_2030', 'Capacity_Wind_on_2030']]

# climate_years = [1995, 2008, 2009]
climate_years = [2008, 2009]

# Production Profiles @ onshore nodes
for year in climate_years:
    for node in onshore_nodes:
        print(node)
        if node != 'NO1':
            nuts_at_node = capacities[capacities['Node'] == node]
            production_profile = pd.DataFrame()
            production_profile['PV'] = np.zeros(8760)
            production_profile['Wind_on'] = np.zeros(8760)
            production_profile['Wind_off'] = aggregate_offshore_wind_per_node(node, year)
            production_profile['Biomass'] = np.ones(8760) * 0.53 * capacity_biomass[capacity_biomass['Node'] == node]['Biomass'].iloc[0]
            for nuts2_region in nuts_at_node['NUTS_ID']:
                capacities_at_nuts = nuts_at_node[nuts_at_node['NUTS_ID'] == nuts2_region]
                location = SimpleNamespace()
                location.lon = capacities_at_nuts['lon'].iloc[0]
                location.lat = capacities_at_nuts['lat'].iloc[0]
                location.altitude = 10
                PV_cap = capacities_at_nuts['Capacity_PV_2030'].iloc[0]
                wind_cap = capacities_at_nuts['Capacity_Wind_on_2030'].iloc[0]
                climate_data = pd.read_csv(path + 'ClimateData/' + nuts2_region + '_' + str(year) + '.csv', index_col=0)
                climate_data.index = pd.to_datetime(climate_data.index)
                climate_data = climate_data[0:8760]

                # PV
                cf_PV = perform_fitting_PV(climate_data, location)
                cf_PV = cf_PV.coefficients['capfactor']
                cf_PV[cf_PV<=0] = 0
                production_profile.index = cf_PV.index
                production_profile['PV'] = production_profile['PV'] + cf_PV * PV_cap

                # onshore wind
                cf_wind = perform_fitting_WT(climate_data, 'WindTurbine_Onshore_4000', 100)
                production_profile['Wind_on'] = production_profile['Wind_on'] + cf_wind.coefficients['capfactor'] * wind_cap * 0.95

                # Run of river
                production_profile['RunOfRiver'] = read_run_of_river_production(node, year)

        else:
            # for Norway
            installed_capacities_no = get_installed_capacities_no()
            production_profile = get_production_profile_no(installed_capacities_no, year)

            production_profile['RunOfRiver'] = read_run_of_river_production(node, year)

        production_profile['total'] = production_profile['PV'] + \
                                      production_profile['Wind_on'] + \
                                      production_profile['Wind_off'] + \
                                      production_profile['Biomass'] + \
                                      production_profile['RunOfRiver']

        production_profile.to_csv(save_path + node + '_' + str(year) + '.csv')

#
# # Production Profiles @ offshore nodes
# for year in climate_years:
#     for node in offshore_nodes:
#         production_profile = pd.DataFrame()
#         production_profile['PV'] = np.zeros(8760)
#         production_profile['Wind_on'] = np.zeros(8760)
#         production_profile['Wind_off'] = aggregate_offshore_wind_per_node(node, year)
#         production_profile['Biomass'] =  np.zeros(8760)
#         production_profile['RunOfRiver'] = np.zeros(8760)
#         production_profile['total'] = production_profile['PV'] + \
#                                       production_profile['Wind_on'] + \
#                                       production_profile['Wind_off'] + \
#                                       production_profile['Biomass'] + \
#                                       production_profile['RunOfRiver']
#         production_profile.to_csv(save_path + node + '_' + str(year) + '.csv')
#

