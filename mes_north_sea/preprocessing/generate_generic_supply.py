# Run of river, PV, Wind onshore, Wind offshore, Biomass
import numpy as np
import pandas as pd
from types import SimpleNamespace

from mes_north_sea.preprocessing.utilities import Configuration, to_latex, CalculateReGeneration

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

    climate_data = pd.read_csv(c.load_path_climate_data + nuts_region['NUTS_ID'] + '_' + str(c.climate_year) + '.csv', index_col=0)
    climate_data.index = pd.date_range(start=str(c.climate_year)+'-01-01 00:00', end=str(c.climate_year)+'-12-31 23:00', freq='1h')

    ReCalc.fit_technology_performance(climate_data, 'PV', location)
    production_profiles_nuts[nuts_region['NUTS_ID'], 'PV'] = \
        ReCalc.fitted_performance.coefficients['capfactor'] * nuts_region['Capacity_PV_2030']

    ReCalc.fit_technology_performance(climate_data, 'Wind', location)
    production_profiles_nuts[nuts_region['NUTS_ID'], 'Wind onshore'] = \
        ReCalc.fitted_performance.coefficients['capfactor'] * nuts_region['Capacity_Wind_on_2030']


production_profiles_nodes = production_profiles_nuts.T.reset_index().merge(c.nodekeys_nuts[['NUTS_ID', 'Node']],
                                 left_on=production_profiles_nuts.T.reset_index()['Nuts'],
                                 right_on=['NUTS_ID'])
production_profiles_nodes = production_profiles_nodes.drop(columns=['NUTS_ID', 'Nuts'])
production_profiles_nodes = production_profiles_nodes.groupby(['Node', 'Profile']).sum().T

# Norway
scenario = 'National Trends'
climate_year = 'CY 1995'
year = 2030
parameter = 'Capacity (MW)'
tyndp_caps = pd.read_excel(c.load_path_tyndp_cap, sheet_name='Capacity & Dispatch')
tyndp_caps = tyndp_caps[tyndp_caps['Scenario'] == scenario]
tyndp_caps = tyndp_caps[tyndp_caps['Year'] == year]
tyndp_caps = tyndp_caps[tyndp_caps['Climate Year'] == climate_year]
tyndp_caps = tyndp_caps[tyndp_caps['Parameter'] == parameter]
tyndp_caps['Node'] = tyndp_caps['Node'].replace('DKKF', 'DK00')
tyndp_caps['Node'] = tyndp_caps['Node'].replace('UKNI', 'UK00')
tyndp_caps['Country'] = tyndp_caps['Node'].str[0:2]
tyndp_caps = tyndp_caps.rename(columns={'Value': 'Capacity TYNDP', 'Fuel': 'Technology'})
tyndp_caps = tyndp_caps[tyndp_caps['Country'] == 'NO']

pv_gen = np.zeros(8760)
wind_gen = np.zeros(8760)
for bidding_zone in tyndp_caps['Node'].unique():
    cap = tyndp_caps.loc[(tyndp_caps['Node'] == bidding_zone) & (tyndp_caps['Technology'] == 'Solar'), 'Capacity TYNDP']
    cap_factor = pd.read_excel('C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/00_raw_data/capacity_factors_no/PECD_LFSolarPV_2030_edition 2022.1.xlsx',
                               sheet_name=bidding_zone, skiprows=10, header=[0])
    try:
        prod = cap_factor[str(c.climate_year)] * float(cap)
    except:
        prod = cap_factor[c.climate_year] * float(cap)
    pv_gen = pv_gen + np.array(prod.fillna(0)[0:8760])

for bidding_zone in tyndp_caps['Node'].unique():
    cap = tyndp_caps.loc[(tyndp_caps['Node'] == bidding_zone) & (tyndp_caps['Technology'] == 'Wind Onshore'), 'Capacity TYNDP']
    cap_factor = pd.read_excel('C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/00_raw_data/capacity_factors_no/PECD_Wind_Onshore_2030_edition 2022.1.xlsx',
                               sheet_name=bidding_zone, skiprows=10, header=[0])
    try:
        prod = cap_factor[str(c.climate_year)] * float(cap)
    except:
        prod = cap_factor[c.climate_year] * float(cap)
    wind_gen = wind_gen + np.array(prod.fillna(0)[0:8760])


production_profiles_nodes[('NO1', 'PV')] = pv_gen
production_profiles_nodes[('NO1', 'Wind onshore')] = wind_gen

# Run of River
cap_nodes = pd.read_csv(c.clean_data_path + 'clean_data/installed_capacities/capacities_node.csv')
cap_nodes = cap_nodes[['Country', 'Node', 'Technology', 'Capacity our work']]
cap_ror = cap_nodes[cap_nodes['Technology'] == 'Hydro - Run of River (Turbine)']

cap_ror_national = cap_ror[['Country', 'Capacity our work']].groupby('Country').sum().rename(columns={'Capacity our work': 'National Capacity'})
cap_ror = cap_ror.merge(cap_ror_national, right_on='Country', left_on='Country')
cap_ror['Share'] =  cap_ror['Capacity our work'] / cap_ror['National Capacity']


regions = {'DE': ['DE00'], 'BE': ['BE00'], 'UK': ['UK00'], 'NL': ['NL00'], 'NO': ['NOS0', 'NOM1', 'NON1']}
for idx, row in cap_ror.iterrows():
    total_inflow = np.zeros(8760)
    for bidding_zone in regions[row['Country']]:
        data_path = c.load_path_hydro_inflow + bidding_zone + '_Hydro Inflow_' + str(c.year) + '.xlsx'
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


