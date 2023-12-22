# Run of river, PV, Wind onshore, Wind offshore, Biomass
import numpy as np
import pandas as pd
from types import SimpleNamespace

from mes_north_sea.preprocessing.utilities import Configuration, to_latex, CalculateReGeneration
from src.data_management.import_data import import_jrc_climate_data as read_climate_data

def divide_dataframe(df, n):
    divided_df = df / n
    concatenated_df = pd.DataFrame(divided_df.values.repeat(n, axis=0))
    return concatenated_df

c = Configuration()

# Build DF
nuts = c.nodekeys_nuts['NUTS_ID'].unique()
sources = ['PV', 'Wind onshore', 'Run of River', 'Wind offshore', 'Biomass']

multi_index = pd.MultiIndex.from_product([nuts, sources], names=['Nuts', 'Profile'])
production_profiles_nuts = pd.DataFrame(columns=multi_index)

# PV & wind
cap_nuts = pd.read_csv(c.clean_data_path + 'clean_data/installed_capacities/capacities_nuts.csv', index_col=0)

ReCalc = CalculateReGeneration()
for idx, nuts_region in cap_nuts.iterrows():
    location = SimpleNamespace()
    print(nuts_region['NUTS_ID'])
    if nuts_region['NUTS_ID'] == 'DK01':
        location.lat = 55.8
        location.lon = 12.43
    elif nuts_region['NUTS_ID'] == 'DK05':
        location.lat = 56.96
        location.lon = 9.817
    else:
        location.lat = nuts_region['lat']
        location.lon = nuts_region['lon']
    location.altitude = 10

    climate_data = read_climate_data(location.lon, location.lat, c.climate_year, location.altitude)

    ReCalc.fit_technology_performance(climate_data['dataframe'], 'PV', location)
    production_profiles_nuts[nuts_region['NUTS_ID'], 'PV'] = \
        ReCalc.fitted_performance.coefficients['capfactor'] * nuts_region['Capacity_PV_2030']

    ReCalc.fit_technology_performance(climate_data['dataframe'], 'Wind', location)
    production_profiles_nuts[nuts_region['NUTS_ID'], 'Wind onshore'] = \
        ReCalc.fitted_performance.coefficients['capfactor'] * nuts_region['Capacity_Wind_on_2030']


production_profiles_nodes = production_profiles_nuts.T.reset_index().merge(c.nodekeys_nuts[['NUTS_ID', 'Node']],
                                 left_on=production_profiles_nuts.T.reset_index()['Nuts'],
                                 right_on=['NUTS_ID'])
production_profiles_nodes = production_profiles_nodes.drop(columns=['NUTS_ID', 'Nuts'])
production_profiles_nodes = production_profiles_nodes.groupby(['Node', 'Profile']).sum().T

# Run of River
cap_nodes = pd.read_csv(c.clean_data_path + 'clean_data/installed_capacities/capacities_node.csv')
cap_nodes = cap_nodes[['Country', 'Node', 'Technology', 'Capacity our work']]
cap_ror = cap_nodes[cap_nodes['Technology'] == 'Hydro - Run of River (Turbine)']

cap_ror_national = cap_ror[['Country', 'Capacity our work']].groupby('Country').sum().rename(columns={'Capacity our work': 'National Capacity'})
cap_ror = cap_ror.merge(cap_ror_national, right_on='Country', left_on='Country')
cap_ror['Share'] =  cap_ror['Capacity our work'] / cap_ror['National Capacity']

load_path_inflows = 'E:/00_Data/00_RenewableGeneration/ENTSOE_ERAA/Hydro Inflows/PEMMDB_'

regions = {'DE': ['DE00'], 'BE': ['BE00'], 'UK': ['UK00'], 'NL': ['NL00'], 'NO': ['NOS0', 'NOM1', 'NON1']}
for idx, row in cap_ror.iterrows():
    total_inflow = np.zeros(8760)
    for bidding_zone in regions[row['Country']]:
        data_path = load_path_inflows + bidding_zone + '_Hydro Inflow_' + str(c.year) + '.xlsx'
        temp = pd.read_excel(data_path, sheet_name='Run of River', skiprows=12,
                             usecols=[i for i in range(16, 16 + 37)], names=['Week', *range(1982, 2018)])
        ror_flow = divide_dataframe(temp[c.climate_year], 24) * 1000
        ror_flow = np.array(ror_flow.fillna(0)[0:8760][0])
        total_inflow = ror_flow + total_inflow
    production_profiles_nodes.loc[:, (row['Node'], 'Run of River')] = total_inflow * cap_ror.set_index('Node')['Share'][row['Node']]

# Biomass
cap_bio = cap_nodes[cap_nodes['Technology'] == 'Biofuels']

for idx, row in cap_bio.iterrows():
    production_profiles_nodes.loc[:, (row['Node'],'Biomass')] = row['Capacity our work'] * 0.53

# Wind offshore
load_path_profile = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/00_raw_data/offshore_wind_farm_profiles/'
cap_offshore = pd.read_csv(c.load_path_offshore_farms)
onshore_nodes = c.nodekeys_nuts['Node'].unique()

for node in cap_offshore['NODE_2'].unique():
    cap_at_node = cap_offshore[cap_offshore['NODE_2'] == node]
    total_generation = np.zeros([8760,1])
    for idx, park in cap_at_node.iterrows():
        total_generation = total_generation + \
                           park['POWER_MW'] * pd.read_csv(load_path_profile + park['NAME'].replace('/', '-') + '_' + str(c.climate_year) + '.csv', header=None).to_numpy()
    production_profiles_nodes.loc[:, (node,'Wind offshore')] = total_generation.flatten()

totals = production_profiles_nodes.T.groupby('Node').sum().T
totals.columns = pd.MultiIndex.from_product([totals.columns, ['total']], names=['Node', 'Total'])

production_profiles_nodes = pd.concat([production_profiles_nodes, totals], axis=1)

production_profiles_nodes.to_csv(c.clean_data_path + 'clean_data/production_profiles_re/production_profiles_re.csv')


