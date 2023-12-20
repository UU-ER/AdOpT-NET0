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
cap_national_summary = cap_national_summary.drop(columns=['Source'])

cap_national_summary = cap_national_summary.set_index(['Country', 'Technology', 'Type'])
cap_national_summary.rename(columns={'Capacity TYNDP': 'TYNDP/ERAA 2022'})

cap_national_summary.to_csv(c.savepath_cap_per_country)
to_latex(cap_national_summary/1000,
         'Installed Capacities per Country (GW) and per source',
         c.savepath_cap_per_country_summary, rounding=2, columns=None)


# Solar, wind onshore
# Installed capacity 2023
cap_re_nuts_2023 = pd.read_csv(c.load_path_re_cap_2023)
cap_re_nuts_2023 = cap_re_nuts_2023.set_index('NUTS_ID')
cap_re_national_2023 = cap_re_nuts_2023.drop(columns=['LEVL_CODE']).groupby('CNTR_CODE').sum()
cap_re_national_2023 = cap_re_national_2023.reset_index()
cap_re_national_2023 = cap_re_national_2023.rename(columns={'CNTR_CODE': 'Country',
                                                           'Capacity_Wind_on': 'Wind_on',
                                                            'Capacity_PV': 'PV'})
cap_re_national_2023 = cap_re_national_2023.set_index('Country')


cap_re_national_2030 = cap_national[(cap_national['Technology']=='Solar') | (cap_national['Technology']=='Wind Onshore')]
cap_re_national_2030 = cap_re_national_2030[['Country', 'Technology', 'Capacity TYNDP']].pivot(columns=['Technology'], index=['Country'])
cap_re_national_2030 = cap_re_national_2030.rename(columns={'Capacity TYNDP': 'Capacity 2030 (TYNDP)'})
cap_re_national_2030 = cap_re_national_2030.rename(columns={'Wind Onshore': 'Wind_on',
                                    'Solar': 'PV'})

print(cap_re_national_2023.merge(cap_re_national_2030, left_index=True, right_index=True)/1000)


# Potential
re_potential = pd.read_csv(c.load_path_re_potential, delimiter=';')
re_potential = re_potential.set_index('nuts2_code')
re_potential = re_potential * 1000

potential_re_nuts = cap_re_nuts_2023.join(re_potential[['solar_capacity_gw_high_total', 'wind_onshore_capacity_gw_high']])
potential_re_nuts = potential_re_nuts.rename(
    columns={"solar_capacity_gw_high_total": "Potential_PV", "wind_onshore_capacity_gw_high": "Potential_Wind_on"})

# Remaining Potential
potential_re_nuts["RemainingPotential_PV"] = potential_re_nuts["Potential_PV"] - potential_re_nuts["Capacity_PV"]
potential_re_nuts["RemainingPotential_Wind_on"] = potential_re_nuts["Potential_Wind_on"] - potential_re_nuts["Capacity_Wind_on"]
potential_re_nuts.loc[potential_re_nuts["RemainingPotential_PV"] < 0, "RemainingPotential_PV"] = 0
potential_re_nuts.loc[potential_re_nuts["RemainingPotential_Wind_on"] < 0, "RemainingPotential_Wind_on"] = 0
potential_re_nuts = potential_re_nuts[['CNTR_CODE','NUTS_NAME',
                                      'Capacity_Wind_on',
                                      'Capacity_PV',
                                      'Potential_Wind_on',
                                      'Potential_PV',
                                      'RemainingPotential_Wind_on',
                                      'RemainingPotential_PV',]]
potential_re_national = potential_re_nuts.groupby('CNTR_CODE').sum()
potential_re_national = potential_re_national.rename(columns={'RemainingPotential_Wind_on': 'RemainingPotentialNational_Wind_on',
                                                      'RemainingPotential_PV': 'RemainingPotentialNational_PV'})
potential_re_national = potential_re_national.reset_index()

# Calculate intalled capacity in 2030
cap_national_solar = cap_national[cap_national['Technology']=='Solar']
cap_national_solar = cap_national_solar.rename(columns={'Capacity TYNDP': 'CapacityNational_2030_PV'})
cap_national_wind_on = cap_national[cap_national['Technology']=='Wind Onshore']
cap_national_wind_on = cap_national_wind_on.rename(columns={'Capacity TYNDP': 'CapacityNational_2030_Wind_on'})
cap_re_nuts_2030 = potential_re_nuts.reset_index().merge(cap_national_wind_on[['CapacityNational_2030_Wind_on', 'Country']], left_on=['CNTR_CODE'], right_on=['Country'])
cap_re_nuts_2030 = cap_re_nuts_2030.drop(columns=['Country'])
cap_re_nuts_2030 = cap_re_nuts_2030.merge(cap_national_solar[['CapacityNational_2030_PV', 'Country']], left_on=['CNTR_CODE'], right_on=['Country'])
cap_re_nuts_2030 = cap_re_nuts_2030.drop(columns=['Country'])
cap_re_nuts_2030 = cap_re_nuts_2030.merge(cap_re_national_2023.rename(columns = {'PV': 'CapacityNational_2023_PV',
                                                                                 'Wind_on': 'CapacityNational_2023_Wind_on'}),
                                          left_on=['CNTR_CODE'], right_on=['Country'])


cap_re_nuts_2030 = cap_re_nuts_2030.merge(potential_re_national[['CNTR_CODE', 'RemainingPotentialNational_PV', 'RemainingPotentialNational_Wind_on']], left_on=['CNTR_CODE'], right_on=['CNTR_CODE'])


cap_re_nuts_2030['Capacity_PV_2030'] = (cap_re_nuts_2030['RemainingPotential_PV'] / cap_re_nuts_2030['RemainingPotentialNational_PV']) * \
                             (cap_re_nuts_2030['CapacityNational_2030_PV'] - cap_re_nuts_2030['CapacityNational_2023_PV']) + cap_re_nuts_2030['Capacity_PV']

cap_re_nuts_2030['Capacity_Wind_on_2030'] = cap_re_nuts_2030['RemainingPotential_Wind_on'] / cap_re_nuts_2030['RemainingPotentialNational_Wind_on'] * \
                                  (cap_re_nuts_2030['CapacityNational_2030_Wind_on'] - cap_re_nuts_2030['CapacityNational_2023_Wind_on']) + cap_re_nuts_2030[
                                      'Capacity_Wind_on']

cap_re_nuts_2030['problem_PV'] = cap_re_nuts_2030['Capacity_PV_2030'] >= cap_re_nuts_2030['Potential_PV']
cap_re_nuts_2030['problem_Wind_on'] = cap_re_nuts_2030['Capacity_Wind_on_2030'] >= cap_re_nuts_2030['Potential_Wind_on']

# Report National Capacities
cap_re_national_2030_ours = cap_re_nuts_2030[['CNTR_CODE', 'Capacity_PV_2030', 'Capacity_Wind_on_2030']].groupby('CNTR_CODE').sum()
cap_re_national_2030_ours = cap_re_national_2030_ours.merge(cap_national_wind_on[['CapacityNational_2030_Wind_on', 'Country']], left_index=True, right_on=['Country'])
cap_re_national_2030_ours = cap_re_national_2030_ours.merge(cap_national_solar[['CapacityNational_2030_PV', 'Country']], left_on=['Country'], right_on=['Country'])
cap_re_national_2030_ours = cap_re_national_2030_ours.set_index('Country')
cap_re_national_2030_ours = cap_re_national_2030_ours.rename(columns={'Capacity_PV_2030': 'PV (Our work)', 'Capacity_Wind_on_2030': 'Wind, onshore (Our work)',
       'CapacityNational_2030_Wind_on': 'Wind, onshore (TYNDP)', 'CapacityNational_2030_PV': 'PV (TYNDP)'})

to_latex(cap_re_national_2030_ours[['PV (Our work)', 'PV (TYNDP)', 'Wind, onshore (Our work)', 'Wind, onshore (TYNDP)']]/1000,
         'Installed PV and onshore wind capacities per country (GW)', c.savepath_cap_re_per_country_summary, rounding=2, columns=None)


cap_re_nuts_2030[['NUTS_ID', 'Capacity_PV_2030', 'Capacity_Wind_on_2030']].to_csv(c.savepath_cap_re_per_nutsregion)