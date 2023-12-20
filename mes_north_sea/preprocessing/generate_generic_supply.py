# Run of river, PV, Wind onshore, Biomass
import pandas as pd
from types import SimpleNamespace

from mes_north_sea.preprocessing.utilities import Configuration, to_latex, CalculateReGeneration
from src.data_management.import_data import import_jrc_climate_data as read_climate_data

c = Configuration()

# Build DF
nuts = c.nodekeys_nuts['NUTS_ID'].unique()
sources = ['PV', 'Wind onshore', 'total']

multi_index = pd.MultiIndex.from_product([nuts, sources], names=['Nuts', 'Profile'])
production_profiles_nuts = pd.DataFrame(columns=multi_index)

# PV & wind
pv_cap = pd.read_csv(c.savepath_cap_re_per_nutsregion, index_col=0)
pv_cap = pv_cap.merge(c.nodekeys_nuts[['NUTS_ID', 'lon', 'lat']], right_on='NUTS_ID', left_on='NUTS_ID')

ReCalc = CalculateReGeneration()
for idx, nuts_region in pv_cap.iterrows():
    location = SimpleNamespace()
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

    production_profiles_nuts[nuts_region['NUTS_ID'], 'total'] = \
        production_profiles_nuts[nuts_region['NUTS_ID'], 'Wind onshore'] + production_profiles_nuts[nuts_region['NUTS_ID'], 'PV']


production_profiles_nodes = production_profiles_nuts.T.reset_index().merge(c.nodekeys_nuts[['NUTS_ID', 'Node']],
                                 left_on=production_profiles_nuts.T.reset_index()['Nuts'],
                                 right_on=['NUTS_ID'])
production_profiles_nodes = production_profiles_nodes.drop(columns=['NUTS_ID', 'Nuts'])
production_profiles_nodes = production_profiles_nodes.groupby(['Node', 'Profile']).sum().T

# Run of River
cap = pd.read_csv(c.savepath_cap_per_node)
cap = cap[['Node', 'Technology', 'Capacity our work']]
cap = cap[cap['Technology'] == 'Run of River']


# Is there really no run of river at all other nodes???


production_profiles_nodes.sum()