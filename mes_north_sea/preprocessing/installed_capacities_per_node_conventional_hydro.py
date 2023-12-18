import pandas as pd
from mes_north_sea.preprocessing.utilities import Configuration, to_latex

def calculate_national_caps(names, c, eraa_caps):
    cap = {}
    for type in names:
        cap[type] = 0
    caps_national = pd.DataFrame(columns=cap.keys())
    for country in c.countries:
        cap = {key: 0 for key in cap}  # Reset cap for each country
        for bidding_zone in c.countries[country]:
            for type in names:
                cap[type] = cap[type] + eraa_caps.at[type, bidding_zone]
        caps_national = caps_national.append(pd.Series(cap, name=country))

    multiindex = pd.MultiIndex.from_product([caps_national.index, ['ENTSO-E']], names=['Country', 'Source'])
    caps_national.index = multiindex

    return caps_national

def replace_column_names(column_name):
    for old_part, new_part in name_mapping.items():
        column_name = column_name.replace(old_part, new_part)
    return column_name

def replace_in_multiindex(value):
    for old_value, new_value in name_mapping.items():
        if old_value in value:
            return value.replace(old_value, new_value)
    return value

def replace_substrings(text, mapping):
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text

c = Configuration()

scenario = 'National Trends'
climate_year = 'CY 1995'
year = 2030
parameter = 'Capacity (MW)'

tyndp_installed = pd.read_excel(c.load_path_tyndp_cap, sheet_name='Capacity & Dispatch')
tyndp_installed = tyndp_installed[tyndp_installed['Scenario']== scenario]
tyndp_installed = tyndp_installed[tyndp_installed['Year']== year]
tyndp_installed = tyndp_installed[tyndp_installed['Climate Year']== climate_year]
tyndp_installed = tyndp_installed[tyndp_installed['Parameter']== parameter]
tyndp_installed['Node'] = tyndp_installed['Node'].replace('DKKF', 'DK00')
tyndp_installed['Node'] = tyndp_installed['Node'].replace('UKNI', 'UK00')

fuels = tyndp_installed['Fuel'].unique()

# National Capacity TYNDP
# 'Biofuels', 'Coal & Lignite', 'Gas', 'Nuclear', 'Oil', 'Other RES', 'Solar', 'Wind Offshore','Wind Onshore'
cap = {}
for fuel in fuels:
    cap[fuel] = 0

cap_national = pd.DataFrame(columns=cap.keys())
for country in c.countries:
    cap = {key: 0 for key in cap}  # Reset cap for each country
    for bidding_zone in c.countries[country]:
        tyndp_installed_at_bidding_zone = tyndp_installed[tyndp_installed['Node'] == bidding_zone]
        tyndp_installed_at_bidding_zone = tyndp_installed_at_bidding_zone.groupby('Fuel').sum()
        for fuel in fuels:
            cap[fuel] = cap[fuel] + tyndp_installed_at_bidding_zone.at[fuel, 'Value']

    cap_national = cap_national.append(pd.Series(cap, name=country))

cap_national.at['BE', 'Nuclear'] = 2077
cap_national['Gas'] = cap_national['Gas'] + cap_national['Other Non RES']
cap_national = cap_national.drop(columns=['Other Non RES'])
cap_national['Other RES'] = cap_national['Biofuels'] + cap_national['Other RES']
cap_national = cap_national.drop(columns=['Biofuels'])
cap_national = cap_national.drop(columns=['Hydro'])
multiindex = pd.MultiIndex.from_product([cap_national.index, ['ENTSO-E']], names=['Country', 'Source'])
cap_national.index = multiindex

# National Capacities ERAA
eraa_caps = pd.read_excel(c.load_path_tyndp_cap_hydro, sheet_name = 'TY 2030', skiprows=3, index_col=0)

names_energy = ['Hydro - Reservoir',
 'Hydro - Pump Storage Open Loop',
 'Hydro - Pump Storage Closed Loop'
]
names_capacity = ['Hydro - Run of River (Turbine)',
 'Hydro - Reservoir (Turbine)',
 'Hydro - Pump Storage Open Loop (Turbine)',
 'Hydro - Pump Storage Closed Loop (Turbine)',
 'Hydro - Pump Storage Open Loop (Pumping)',
 'Hydro - Pump Storage Closed Loop (Pumping)'
]

eraa_caps_energy = eraa_caps.loc[names_energy]
name_mapping = {
    'Hydro - Pump Storage Closed Loop': 'Closed Loop',
    'Hydro - Reservoir': 'Reservoir',
    'Hydro - Run of River': 'Run of River',
    'Hydro - Pump Storage Open Loop': 'Open Loop'
}
eraa_caps_capacity = eraa_caps.loc[names_capacity]

# NATIONAL
# Energy Capacities
eraa_caps_energy_national = calculate_national_caps(names_energy, c, eraa_caps_energy)
eraa_caps_energy_national = eraa_caps_energy_national.rename(columns=replace_column_names)
multiindex = pd.MultiIndex.from_product([eraa_caps_energy_national.columns, ['Capacity']], names=['Technology', 'Type'])
eraa_caps_energy_national.columns = multiindex

# Pump/Turbine Capacities
eraa_caps_capacity_national = calculate_national_caps(names_capacity, c, eraa_caps_capacity)
eraa_caps_capacity_national = eraa_caps_capacity_national.rename(columns=replace_column_names)

substrings = {
    'Pump': 'Pumping',
    'Turbine': 'Turbine',
}
multiindex = pd.MultiIndex.from_tuples([], names=['Technology', 'Type'])
for key, value in substrings.items():
    matching_columns = [col for col in eraa_caps_capacity_national.columns if value in col]
    multiindex = multiindex.append(pd.MultiIndex.from_product([matching_columns, [key]], names=['Technology', 'Type']))

non_matching_columns = [col for col in eraa_caps_capacity_national.columns if all(sub not in col for sub in substrings.values())]
multiindex = multiindex.append(
    pd.MultiIndex.from_product([non_matching_columns, ['Capacity']], names=['Technology', 'Type']))

eraa_caps_capacity_national.columns = multiindex

cap_national.columns = pd.MultiIndex.from_product([cap_national.columns, ['Capacity']], names=['Technology', 'Type'])

eraa_caps_national = pd.merge(eraa_caps_energy_national, eraa_caps_capacity_national, left_index=True, right_index=True, how='outer')
cap_national = pd.merge(cap_national, eraa_caps_national, left_index=True, right_index=True, how='outer')

cap_national = pd.DataFrame(cap_national.stack(level=['Technology', 'Type']))
cap_national.columns=['Capacity TYNDP']


# Capacities per node PyPSA
pypsa_installed = pd.read_csv(c.load_path_pypsa_cap_all)
pypsa_installed = pypsa_installed.rename(columns={'NODE_NAME': 'Node', 'Others renewable': 'Other RES'})

pypsa_installed['ENTSOE Category'] = pypsa_installed['ENTSOE Category'].apply(lambda x: replace_substrings(x, name_mapping))

pypsa_installed['Country'] = pypsa_installed['Country'].replace('GB', 'UK')
pypsa_installed = pypsa_installed[['Country', 'Node', 'Capacity', 'ENTSOE Category']]

pypsa_installed = pypsa_installed.groupby(['Country', 'ENTSOE Category', 'Node']).sum()
pypsa_installed = pypsa_installed.rename({'GB': 'UK'})
pypsa_installed['Share'] = pypsa_installed.groupby(['Country', 'ENTSOE Category'])['Capacity'].transform(lambda x: x / x.sum())
pypsa_installed = pypsa_installed.rename(columns={'Capacity': 'Capacity PyPsa'})

pypsa_installed = pypsa_installed.reset_index()
pypsa_installed['ENTSOE Category'] = pypsa_installed['ENTSOE Category'].apply(lambda x: replace_substrings(x, {' (Turbine)': ''}))
pypsa_attach_open_loop = pypsa_installed[pypsa_installed['ENTSOE Category'] == 'Reservoir']
pypsa_attach_open_loop['ENTSOE Category'] = 'Open Loop'
pypsa_installed = pd.concat([pypsa_installed, pypsa_attach_open_loop])

cap_national = cap_national.reset_index()
cap_national['Technology'] = cap_national['Technology'].apply(lambda x: replace_substrings(x, {' (Turbine)': ''}))
cap_national['Technology'] = cap_national['Technology'].apply(lambda x: replace_substrings(x, {' (Pumping)': ''}))



cap_node = pypsa_installed.merge(cap_national, left_on=['Country', 'ENTSOE Category'], right_on=['Country', 'Technology'])

cap_node['Capacity our work'] = cap_node['Share'] * cap_node['Capacity TYNDP']


cap_node_export = cap_node[['Country', 'Node', 'Capacity PyPsa', 'Technology', 'Type', 'Capacity TYNDP', 'Capacity our work']]
cap_node_export = cap_node_export.set_index(['Country', 'Node', 'Technology', 'Type'])
cap_node_export.to_csv(c.savepath_cap_per_node)
to_latex(cap_node_export, 'Installed Capacities per Node (GW) and per source',
         c.savepath_cap_per_node_summary, rounding=2, columns=None)

cap_national_summary = cap_node.groupby(['Country', 'Technology', 'Type']).sum()
cap_national_summary = cap_national_summary[['Capacity our work', 'Capacity PyPsa']]
cap_national_summary = cap_national_summary.reset_index().merge(cap_national, left_on=['Country', 'Technology', 'Type'], right_on=['Country', 'Technology', 'Type'])
cap_national_summary = cap_national_summary.drop(columns=['index', 'Source'])

cap_national_summary = cap_national_summary.set_index(['Country', 'Technology', 'Type'])
cap_national_summary.rename(columns={'Capacity TYNDP': 'TYNDP/ERAA 2022'})

cap_national_summary.to_csv(c.savepath_cap_per_country)
to_latex(cap_national_summary/1000,
         'Installed Capacities per Country (GW) and per source',
         c.savepath_cap_per_country_summary, rounding=2, columns=None)


#
#
#
#
# pypsa_installed_national = pypsa_installed[['Country', 'Capacity', 'ENTSOE Category']]
# pypsa_installed_national = pypsa_installed_national.groupby(['Country', 'ENTSOE Category']).sum()
# pypsa_installed_national = pypsa_installed_national.pivot_table(values='Capacity', columns='ENTSOE Category', index='Country', fill_value=0)
#
# multiindex = pd.MultiIndex.from_product([pypsa_installed_national.index, ['Gotzens et al. (2019)']], names=['Country', 'Source'])
# pypsa_installed_national.index = multiindex
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# pypsa_installed_national_conventional = pypsa_installed_national.drop(columns=['Hydro - Pump Storage Closed Loop (Turbine)',
#        'Hydro - Reservoir (Turbine)', 'Hydro - Run of River (Turbine)'])
# for col in cap_national:
#     if not col in pypsa_installed_national_conventional.columns:
#         pypsa_installed_national_conventional[col] = float('nan')
# multiindex = pd.MultiIndex.from_product([pypsa_installed_national_conventional.index, ['Gotzens et al. (2019)']], names=['Country', 'Source'])
# pypsa_installed_national_conventional.index = multiindex
#
# cap_national = pd.concat([pypsa_installed_national_conventional, cap_national])
# cap_national = cap_national.sort_index(axis=0)
#
# cap_national = cap_national.drop(columns=['Others renewable', 'Other RES', 'Solar', 'Wind Offshore', 'Wind Onshore'])
#
# cap_national.to_csv(c.savepath_cap_per_country)
# to_latex(cap_national/1000,
#          'Installed Capacities per Country (GW) and per source',
#          c.savepath_cap_per_country_summary, rounding=2, columns=None)
#
#
# # Determine Keys from PyPsa Data
# pypsa_installed = pypsa_installed.groupby(['Country', 'ENTSOE Category', 'Node']).sum()
# pypsa_installed = pypsa_installed.rename({'GB': 'UK'})
# pypsa_installed['Share'] = pypsa_installed.groupby(['Country', 'ENTSOE Category'])['Capacity'].transform(lambda x: x / x.sum())
# pypsa_installed = pypsa_installed.rename(columns={'Capacity': 'Capacity PyPsa'})
#
# tyndp_caps = cap_national[cap_national.index.get_level_values('Source') == 'TYNDP 2022']
# tyndp_caps = tyndp_caps.pivot_table(index='Country', fill_value=0)
# tyndp_caps = tyndp_caps.reset_index()
#
# tyndp_caps = pd.melt(tyndp_caps, id_vars=['Country'], var_name='ENTSOE Category', value_name='Capacity TYNDP')
#
# pypsa_installed = pypsa_installed.reset_index(level='Node')
# cap_node = pypsa_installed.merge(tyndp_caps, left_on=['Country', 'ENTSOE Category'], right_on=['Country', 'ENTSOE Category'])
#
# categories_to_filter = ['Coal & Lignite', 'Gas', 'Nuclear', 'Oil']
# cap_node = cap_node[cap_node['ENTSOE Category'].isin(categories_to_filter)]
# cap_node['Capacity our work'] = cap_node['Share'] * cap_node['Capacity TYNDP']
# cap_node.sort_index(axis=0, inplace=True)
#
# cap_node.to_csv(c.savepath_cap_per_node)
#
# cap_node = cap_node.drop(columns=['Capacity TYNDP'])
# cap_node['Capacity our work'] = cap_node['Capacity our work']/1000
# cap_node['Capacity PyPsa'] = cap_node['Capacity PyPsa']/1000
# to_latex(cap_node, 'Installed Capacities per Node (GW) and per source',
#          c.savepath_cap_per_node_summary, rounding=2, columns=None)
#
# # National Capacity ERAA - Hydro
# eraa_caps = pd.read_excel(c.load_path_tyndp_cap_hydro, sheet_name = 'TY 2030', skiprows=3, index_col=0)
#
# names_energy = ['Hydro - Reservoir',
#  'Hydro - Pump Storage Open Loop',
#  'Hydro - Pump Storage Closed Loop'
# ]
# names_capacity = ['Hydro - Run of River (Turbine)',
#  'Hydro - Reservoir (Turbine)',
#  'Hydro - Pump Storage Open Loop (Turbine)',
#  'Hydro - Pump Storage Closed Loop (Turbine)',
#  'Hydro - Pump Storage Open Loop (Pumping)',
#  'Hydro - Pump Storage Closed Loop (Pumping)'
# ]
#
# eraa_caps_energy = eraa_caps.loc[names_energy]
# eraa_caps_capacity = eraa_caps.loc[names_capacity]
#
# def calculate_national_caps(names, c, eraa_caps):
#     cap = {}
#     for type in names:
#         cap[type] = 0
#     caps_national = pd.DataFrame(columns=cap.keys())
#     for country in c.countries:
#         cap = {key: 0 for key in cap}  # Reset cap for each country
#         for bidding_zone in c.countries[country]:
#             for type in names:
#                 cap[type] = cap[type] + eraa_caps.at[type, bidding_zone]
#         caps_national = caps_national.append(pd.Series(cap, name=country))
#
#     multiindex = pd.MultiIndex.from_product([caps_national.index, ['ERAA 2022']], names=['Country', 'Source'])
#     caps_national.index = multiindex
#
#     return caps_national
#
# # NATIONAL
# eraa_caps_energy_national = calculate_national_caps(names_energy, c, eraa_caps_energy)
# eraa_caps_capacity_national = calculate_national_caps(names_capacity, c, eraa_caps_capacity)
#
# pypsa_installed_national_hydro = pypsa_installed_national[['Hydro - Pump Storage Closed Loop (Turbine)',
#        'Hydro - Reservoir (Turbine)', 'Hydro - Run of River (Turbine)']]
# multiindex = pd.MultiIndex.from_product([pypsa_installed_national.index, ['Gotzens et al. (2019)']], names=['Country', 'Source'])
# pypsa_installed_national_hydro.index = multiindex
#
# cap_national_hydro = pd.concat([pypsa_installed_national_hydro, eraa_caps_capacity_national])
# cap_national_hydro = cap_national_hydro.sort_index(axis=0)
#
# cap_national_hydro = cap_national_hydro.rename(columns= {'Hydro - Pump Storage Closed Loop (Turbine)': 'Closed Loop',
#        'Hydro - Reservoir (Turbine)': 'Reservoir', 'Hydro - Run of River (Turbine)': 'Run of River',
#                                     'Hydro - Pump Storage Open Loop (Turbine)': 'Open Loop'})
#
# to_latex(cap_national_hydro[['Closed Loop', 'Reservoir', 'Open Loop', 'Run of River']]/1000,
#          'Installed Hydro Turbine Capacities per Node (GW) and per source',
#          c.savepath_cap_per_node_summary_hydro, rounding=2, columns=None)
#
# # PyPsa adaptions
# pypsa_installed_node_hydro = pypsa_installed
# pypsa_installed_node_hydro = pypsa_installed_node_hydro.reset_index()
# pypsa_installed_node_hydro = pypsa_installed_node_hydro.replace({'Hydro - Pump Storage Closed Loop (Turbine)': 'Closed Loop',
#        'Hydro - Reservoir (Turbine)': 'Reservoir', 'Hydro - Run of River (Turbine)': 'Run of River',
#                                     'Hydro - Pump Storage Open Loop (Turbine)': 'Open Loop'})
#
# #Per Pump/Turbine per Node
# eraa_caps_capacity_national = eraa_caps_capacity_national.rename(columns =
#                      {'Hydro - Pump Storage Closed Loop (Turbine)': 'Closed Loop (Turbine)',
#                       'Hydro - Pump Storage Closed Loop (Pumping)': 'Closed Loop (Pump)',
#                       'Hydro - Reservoir (Turbine)': 'Reservoir (Turbine)',
#                       'Hydro - Run of River (Turbine)': 'Run of River',
#                       'Hydro - Pump Storage Open Loop (Pump)': 'Open Loop (Pump)',
#                       'Hydro - Pump Storage Open Loop (Turbine)': 'Open Loop (Turbine)'})
#
# to_melt = eraa_caps_capacity_national.reset_index()
# to_melt = to_melt.drop(columns=['Source'])
#
# to_merge = pd.melt(to_melt, id_vars=['Country'], var_name='ENTSOE Category', value_name='Capacity ERAA')
# energy_node_hydro = pypsa_installed.merge(to_merge, left_on=['Country', 'ENTSOE Category'], right_on=['Country', 'ENTSOE Category'])
#
#
