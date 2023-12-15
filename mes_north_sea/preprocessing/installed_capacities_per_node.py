import pandas as pd
from mes_north_sea.preprocessing.utilities import Configuration, to_latex

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

# National Capacity TYNDP - no hydro
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

multiindex = pd.MultiIndex.from_product([cap_national.index, ['TYNDP 2022']], names=['Country', 'Source'])
cap_national.index = multiindex


# National Capacity PyPsa - no hydro
pypsa_installed = pd.read_csv(c.load_path_pypsa_cap_all)
pypsa_installed = pypsa_installed.rename(columns={'NODE_NAME': 'Node', 'Others renewable': 'Other RES'})
pypsa_installed = pypsa_installed.rename({'GB': 'UK'})
pypsa_installed = pypsa_installed[['Country', 'Node', 'Capacity', 'ENTSOE Category']]
pypsa_installed_national = pypsa_installed[['Country', 'Capacity', 'ENTSOE Category']]
pypsa_installed_national = pypsa_installed_national.groupby(['Country', 'ENTSOE Category']).sum()
pypsa_installed_national = pypsa_installed_national.pivot_table(values='Capacity', columns='ENTSOE Category', index='Country', fill_value=0)
pypsa_installed_national = pypsa_installed_national.drop(columns=['Hydro - Pump Storage Closed Loop (Turbine)',
       'Hydro - Reservoir (Turbine)', 'Hydro - Run of River (Turbine)'])
for col in cap_national:
    if not col in pypsa_installed_national.columns:
        pypsa_installed_national[col] = float('nan')
multiindex = pd.MultiIndex.from_product([pypsa_installed_national.index, ['Gotzens et al. (2019)']], names=['Country', 'Source'])
pypsa_installed_national.index = multiindex

cap_national = pd.concat([pypsa_installed_national, cap_national])
cap_national = cap_national.sort_index(axis=0)

cap_national.to_csv(c.savepath_cap_per_country)
to_latex(cap_national/1000,
         'Installed Capacities per Country (GW) and per source',
         c.savepath_cap_per_country_summary, rounding=2, columns=None)



# Determine Keys from PyPsa Data
pypsa_installed = pypsa_installed.groupby(['Country', 'ENTSOE Category', 'Node']).sum()
pypsa_installed = pypsa_installed.rename({'GB': 'UK'})
pypsa_installed['Share'] = pypsa_installed.groupby(['Country', 'ENTSOE Category'])['Capacity'].transform(lambda x: x / x.sum())
pypsa_installed = pypsa_installed.rename(columns={'Capacity': 'Capacity PyPsa'})

tyndp_caps = cap_national[cap_national.index.get_level_values('Source') == 'TYNDP 2022']
tyndp_caps = tyndp_caps.pivot_table(index='Country', fill_value=0)
tyndp_caps = tyndp_caps.reset_index()

tyndp_caps = pd.melt(tyndp_caps, id_vars=['Country'], var_name='ENTSOE Category', value_name='Capacity TYNDP')

pypsa_installed = pypsa_installed.reset_index(level='Node')
cap_node = pypsa_installed.merge(tyndp_caps, left_on=['Country', 'ENTSOE Category'], right_on=['Country', 'ENTSOE Category'])

categories_to_filter = ['Coal & Lignite', 'Gas', 'Nuclear']
cap_node = cap_node[cap_node['ENTSOE Category'].isin(categories_to_filter)]
cap_node['Capacity our work'] = cap_node['Share'] * cap_node['Capacity TYNDP']
cap_node.sort_index(axis=0, inplace=True)

cap_node.to_csv(c.savepath_cap_per_node)

cap_node['Capacity our work'] = cap_node['Capacity our work']/1000
cap_node['Capacity TYNDP'] = cap_node['Capacity TYNDP']/1000
cap_node['Capacity PyPsa'] = cap_node['Capacity PyPsa']/1000
to_latex(cap_node,
         'Installed Capacities per Node (GW) and per source',
         c.savepath_cap_per_node_summary, rounding=2, columns=None)

