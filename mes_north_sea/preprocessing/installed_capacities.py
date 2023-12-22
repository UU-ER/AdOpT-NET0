import pandas as pd
from mes_north_sea.preprocessing.utilities import Configuration, to_latex


def preprocess_tyndp_data(c, scenario, year, climate_year):
    """
    - selectr scenario, year, climate year
    - replace node names
    - Aggregates data per bidding zone
    """
    tyndp_caps = pd.read_excel(c.load_path_tyndp_cap, sheet_name='Capacity & Dispatch')
    tyndp_caps = tyndp_caps[tyndp_caps['Scenario'] == scenario]
    tyndp_caps = tyndp_caps[tyndp_caps['Year'] == year]
    tyndp_caps = tyndp_caps[tyndp_caps['Climate Year'] == climate_year]
    tyndp_caps = tyndp_caps[tyndp_caps['Parameter'] == parameter]
    tyndp_caps['Node'] = tyndp_caps['Node'].replace('DKKF', 'DK00')
    tyndp_caps['Node'] = tyndp_caps['Node'].replace('UKNI', 'UK00')
    tyndp_caps['Country'] = tyndp_caps['Node'].str[0:2]
    tyndp_caps = tyndp_caps.rename(columns={'Value': 'Capacity TYNDP', 'Fuel': 'Technology'})

    tecs_to_rename = {'Other Non RES': 'Gas',
                      'Other RES': 'Biofuels'
                    }
    tecs_to_drop = ['Hydro']
    tyndp_caps = tyndp_caps.set_index('Technology')
    tyndp_caps = tyndp_caps.rename(index=tecs_to_rename)
    tyndp_caps = tyndp_caps.drop(tecs_to_drop)
    tyndp_caps = tyndp_caps.reset_index()

    tyndp_caps = tyndp_caps[['Country', 'Technology', 'Capacity TYNDP']].groupby(['Country', 'Technology']).sum()

    return tyndp_caps

def preprocess_eraa_data(c):

    tecs_to_drop = ['Demand Side Response capacity',
                    'Electrolyser',
                    'Batteries (Offtake)',
                    'Solar (Thermal)',
                    'Energy Storage (MWh)',
                    'Batteries',
                    'Batteries (Injection)'
                    ]

    tecs_to_rename = {'Lignite': 'Coal & Lignite',
                        'Hard Coal': 'Coal & Lignite',
                        'Gas ': 'Gas',
                        'Solar (Photovoltaic)': 'Solar',
                        'Biofuel': 'Biofuels',
                        'Others renewable': 'Biofuels',
                        'Hydro - Reservoir': 'Hydro - Reservoir (Energy)',
                        'Hydro - Pondage': 'Hydro - Pondage (Energy)',
                        'Hydro - Pump Storage Open Loop': 'Hydro - Pump Storage Open Loop (Energy)',
                        'Hydro - Pump Storage Closed Loop': 'Hydro - Pump Storage Closed Loop (Energy)',
                        'Others non-renewable': 'Gas'
                        }

    eraa_caps = pd.read_excel(c.load_path_tyndp_cap_hydro, sheet_name='TY 2030', skiprows=3, index_col=0)
    eraa_caps.index.name = 'Technology'
    eraa_caps = eraa_caps.drop(tecs_to_drop)
    eraa_caps = eraa_caps.rename(index=tecs_to_rename)

    eraa_caps = eraa_caps.reset_index().melt(id_vars=['Technology'] ,var_name='Node', value_name='Capacity ERAA').reset_index()
    eraa_caps = eraa_caps.dropna()

    eraa_caps['Country'] = eraa_caps['Node'].str[0:2]
    eraa_caps = eraa_caps[['Country', 'Technology', 'Capacity ERAA']].groupby(['Country', 'Technology']).sum()

    return eraa_caps

def preprocess_pypsa_data(c):
    cap_pypsa = pd.read_csv(c.load_path_pypsa_cap_all)
    cap_pypsa = cap_pypsa.rename(columns={'NODE_NAME': 'Node'})
    cap_pypsa['ENTSOE Category'] = cap_pypsa['ENTSOE Category'].replace('Others renewable', 'Biofuels')
    cap_pypsa['Country'] = cap_pypsa['Country'].replace('GB', 'UK')
    cap_pypsa = cap_pypsa[['Country', 'Node', 'Capacity', 'ENTSOE Category']]
    cap_pypsa = cap_pypsa.groupby(['Country', 'ENTSOE Category', 'Node']).sum()
    cap_pypsa['Share'] = cap_pypsa.groupby(['Country', 'ENTSOE Category'])['Capacity'].transform(lambda x: x / x.sum())
    cap_pypsa = cap_pypsa.reset_index()
    cap_pypsa = cap_pypsa.rename(columns={'Capacity': 'Capacity PyPsa', 'ENTSOE Category': 'Technology'})

    return cap_pypsa

def preprocess_pv_wind_per_nuts(c, cap_national):
    cap_re_nuts_2023 = pd.read_csv(c.load_path_re_cap_2023)
    cap_re_nuts_2023 = cap_re_nuts_2023.set_index('NUTS_ID')
    cap_re_national_2023 = cap_re_nuts_2023.drop(columns=['LEVL_CODE']).groupby('CNTR_CODE').sum()
    cap_re_national_2023 = cap_re_national_2023.reset_index()
    cap_re_national_2023 = cap_re_national_2023.rename(columns={'CNTR_CODE': 'Country',
                                                                'Capacity_Wind_on': 'Wind_on',
                                                                'Capacity_PV': 'PV'})
    cap_re_national_2023 = cap_re_national_2023.set_index('Country')

    # Potential
    re_potential = pd.read_csv(c.load_path_re_potential, delimiter=';')
    re_potential = re_potential.set_index('nuts2_code')
    re_potential = re_potential * 1000

    potential_re_nuts = cap_re_nuts_2023.join(
        re_potential[['solar_capacity_gw_high_total', 'wind_onshore_capacity_gw_high']])
    potential_re_nuts = potential_re_nuts.rename(
        columns={"solar_capacity_gw_high_total": "Potential_PV", "wind_onshore_capacity_gw_high": "Potential_Wind_on"})

    # Remaining Potential
    potential_re_nuts["RemainingPotential_PV"] = potential_re_nuts["Potential_PV"] - potential_re_nuts["Capacity_PV"]
    potential_re_nuts["RemainingPotential_Wind_on"] = potential_re_nuts["Potential_Wind_on"] - potential_re_nuts[
        "Capacity_Wind_on"]
    potential_re_nuts.loc[potential_re_nuts["RemainingPotential_PV"] < 0, "RemainingPotential_PV"] = 0
    potential_re_nuts.loc[potential_re_nuts["RemainingPotential_Wind_on"] < 0, "RemainingPotential_Wind_on"] = 0
    potential_re_nuts = potential_re_nuts[['CNTR_CODE', 'NUTS_NAME',
                                           'Capacity_Wind_on',
                                           'Capacity_PV',
                                           'Potential_Wind_on',
                                           'Potential_PV',
                                           'RemainingPotential_Wind_on',
                                           'RemainingPotential_PV', ]]
    potential_re_national = potential_re_nuts.groupby('CNTR_CODE').sum()
    potential_re_national = potential_re_national.rename(
        columns={'RemainingPotential_Wind_on': 'RemainingPotentialNational_Wind_on',
                 'RemainingPotential_PV': 'RemainingPotentialNational_PV'})
    potential_re_national = potential_re_national.reset_index()

    # Calculate intalled capacity in 2030
    cap_national = cap_national.reset_index()
    cap_national_solar = cap_national[cap_national['Technology'] == 'Solar']
    cap_national_solar = cap_national_solar.rename(columns={'Capacity ours': 'CapacityNational_2030_PV'})
    cap_national_wind_on = cap_national[cap_national['Technology'] == 'Wind Onshore']
    cap_national_wind_on = cap_national_wind_on.rename(columns={'Capacity ours': 'CapacityNational_2030_Wind_on'})
    cap_re_nuts_2030 = potential_re_nuts.reset_index().merge(
        cap_national_wind_on[['CapacityNational_2030_Wind_on', 'Country']], left_on=['CNTR_CODE'], right_on=['Country'])
    cap_re_nuts_2030 = cap_re_nuts_2030.drop(columns=['Country'])
    cap_re_nuts_2030 = cap_re_nuts_2030.merge(cap_national_solar[['CapacityNational_2030_PV', 'Country']],
                                              left_on=['CNTR_CODE'], right_on=['Country'])
    cap_re_nuts_2030 = cap_re_nuts_2030.drop(columns=['Country'])
    cap_re_nuts_2030 = cap_re_nuts_2030.merge(cap_re_national_2023.rename(columns={'PV': 'CapacityNational_2023_PV',
                                                                                   'Wind_on': 'CapacityNational_2023_Wind_on'}),
                                              left_on=['CNTR_CODE'], right_on=['Country'])

    cap_re_nuts_2030 = cap_re_nuts_2030.merge(
        potential_re_national[['CNTR_CODE', 'RemainingPotentialNational_PV', 'RemainingPotentialNational_Wind_on']],
        left_on=['CNTR_CODE'], right_on=['CNTR_CODE'])

    cap_re_nuts_2030['Capacity_PV_2030'] = (cap_re_nuts_2030['RemainingPotential_PV'] / cap_re_nuts_2030[
        'RemainingPotentialNational_PV']) * \
                                           (cap_re_nuts_2030['CapacityNational_2030_PV'] - cap_re_nuts_2030[
                                               'CapacityNational_2023_PV']) + cap_re_nuts_2030['Capacity_PV']

    cap_re_nuts_2030['Capacity_Wind_on_2030'] = cap_re_nuts_2030['RemainingPotential_Wind_on'] / cap_re_nuts_2030[
        'RemainingPotentialNational_Wind_on'] * \
                                                (cap_re_nuts_2030['CapacityNational_2030_Wind_on'] - cap_re_nuts_2030[
                                                    'CapacityNational_2023_Wind_on']) + cap_re_nuts_2030[
                                                    'Capacity_Wind_on']

    cap_re_nuts_2030['problem_PV'] = cap_re_nuts_2030['Capacity_PV_2030'] >= cap_re_nuts_2030['Potential_PV']
    cap_re_nuts_2030['problem_Wind_on'] = cap_re_nuts_2030['Capacity_Wind_on_2030'] >= cap_re_nuts_2030[
        'Potential_Wind_on']

    cap_re_nuts_2030 = cap_re_nuts_2030.merge(c.nodekeys_nuts[['Node', 'NUTS_ID', 'lon', 'lat']], right_on='NUTS_ID', left_on='NUTS_ID')

    return cap_re_nuts_2030

def replace_substrings(text, mapping):
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text

c = Configuration()

scenario = 'National Trends'
climate_year = 'CY 1995'
year = 2030
parameter = 'Capacity (MW)'

# ENTSOE Data
cap_tyndp = preprocess_tyndp_data(c, scenario, year, climate_year)
cap_eraa = preprocess_eraa_data(c)

cap_entsoe_raw = pd.merge(cap_tyndp.reset_index(), cap_eraa.reset_index(), on=['Country', 'Technology'], how='outer')
cap_entsoe_raw = cap_entsoe_raw.set_index('Technology').reset_index()
cap_entsoe_raw.to_csv(c.clean_data_path + 'reporting/installed_capacities/entsoe_national.csv')


# OUR DATA - National
use_eraa_for = ['Hydro - Pondage (Turbine)',
                'Hydro - Reservoir (Turbine)',
                'Hydro - Pump Storage Open Loop (Turbine)',
                'Hydro - Pump Storage Closed Loop (Turbine)',
                'Hydro - Pump Storage Open Loop (Pumping)',
                'Hydro - Pump Storage Closed Loop (Pumping)',
                'Hydro - Run of River (Turbine)',
                'Hydro - Pondage (Energy)',
                'Hydro - Pump Storage Open Loop (Energy)',
                'Hydro - Pump Storage Closed Loop (Energy)'
                ]
use_tyndp_for = ['Biofuels',
                'Coal & Lignite',
                'Gas',
                'Nuclear',
                'Oil',
                'Solar',
                'Wind Onshore',
                ]

cap_national = pd.concat([
    cap_entsoe_raw[cap_entsoe_raw['Technology'].isin(use_tyndp_for)].
        drop(columns='Capacity ERAA').
        rename(columns={'Capacity TYNDP': 'Capacity ours'}),
    cap_entsoe_raw[cap_entsoe_raw['Technology'].isin(use_eraa_for)].
            drop(columns='Capacity TYNDP').
            rename(columns={'Capacity ERAA': 'Capacity ours'})
    ])
cap_national = cap_national.set_index(['Country', 'Technology'])
cap_national.loc[('BE','Nuclear'), 'Capacity ours'] = 2077

# PYPSA DATA - Per Node
cap_pypsa = preprocess_pypsa_data(c)

cap_pypsa_national = cap_pypsa.groupby(['Country', 'Technology']).sum().drop(columns='Share')
cap_pypsa_national.reset_index().to_csv(c.clean_data_path + 'reporting/installed_capacities/pypsa_national.csv')

# Treat hydro
cap_pypsa['Technology'] = cap_pypsa['Technology'].apply(lambda x: replace_substrings(x, {' (Turbine)': ''}))

df_concat = cap_pypsa[cap_pypsa['Technology'] == 'Hydro - Reservoir']
df_concat['Technology'] = 'Hydro - Pump Storage Open Loop'
cap_pypsa = pd.concat([cap_pypsa, df_concat])

hydro_tecs = ['Hydro - Pump Storage Closed Loop', 'Hydro - Pump Storage Open Loop', 'Hydro - Reservoir']

for tec in hydro_tecs:
    df_concat1 = cap_pypsa[cap_pypsa['Technology'] == tec]
    df_concat1['Technology'] = tec + ' (Turbine)'
    df_concat2 = df_concat1.copy()
    df_concat1['Technology'] = tec + ' (Pumping)'
    cap_pypsa.loc[cap_pypsa['Technology'] == tec, 'Technology'] = tec + ' (Energy)'
    df_concat = pd.concat([df_concat1, df_concat2])
    cap_pypsa = pd.concat([cap_pypsa, df_concat])

cap_pypsa.loc[cap_pypsa['Technology'] == 'Hydro - Run of River','Technology'] = 'Hydro - Run of River (Turbine)'

# ALLOCATE CAPACITIES TO NODES
# All but Wind, PV, Biomass
cap_node = cap_pypsa.merge(cap_national.reset_index()[cap_national.reset_index()['Technology'] != 'Biofuels'],
                           left_on=['Country', 'Technology'], right_on=['Country', 'Technology'])
cap_node['Capacity our work'] = cap_node['Share'] * cap_node['Capacity ours']

# Wind onshore, PV
cap_per_nuts = preprocess_pv_wind_per_nuts(c, cap_national)
cap_per_node_pv_wind = cap_per_nuts.groupby('Node').agg({'CNTR_CODE': 'first', 'Capacity_PV_2030': 'sum', 'Capacity_Wind_on_2030': 'sum'})
cap_per_node_pv_wind = cap_per_node_pv_wind.rename(columns = {'CNTR_CODE': 'Country',
                                                              'Capacity_PV_2030': 'Solar',
                                                              'Capacity_Wind_on_2030': 'Wind Onshore'})
cap_per_node_pv_wind = cap_per_node_pv_wind.reset_index().melt(id_vars=['Node', 'Country'], var_name='Technology', value_name = 'Capacity our work')

cap_node_ours = pd.concat([cap_node[cap_per_node_pv_wind.columns], cap_per_node_pv_wind])

# Wind offshore
cap_offshore = pd.read_csv(c.load_path_offshore_farms)
cap_offshore = cap_offshore[['POWER_MW', 'NODE_TYPE', 'NODE_2']].rename(columns={'POWER_MW': 'Capacity our work', 'NODE_2': 'Node'})
cap_offshore['Country'] = cap_offshore['Node'].str[:2]
cap_offshore['Technology'] = 'Wind Offshore'
# cap_offshore_offshore_nodes = cap_offshore[cap_offshore['NODE_TYPE'] != 'Allocated_to_onshore'].drop(columns='NODE_TYPE')
# cap_offshore_allocated_onshore = cap_offshore[cap_offshore['NODE_TYPE'] == 'Allocated_to_onshore'].drop(columns='NODE_TYPE')
cap_node_ours = pd.concat([cap_node_ours, cap_offshore.drop(columns='NODE_TYPE')])

# Biomass
cap_biomass = pd.read_csv(c.load_path_biomass, sep=';').rename(columns={'Biomass': 'Capacity'})
cap_biomass['Share'] = cap_biomass.groupby(['Country'])['Capacity'].transform(lambda x: x / x.sum())
cap_biomass['Technology'] = 'Biofuels'
cap_biomass['Share'] = cap_biomass['Share'].fillna(1)
cap_biomass = cap_biomass.merge(cap_national.reset_index(), left_on=['Country', 'Technology'], right_on=['Country', 'Technology'])
cap_biomass['Capacity our work'] = cap_biomass['Share'] * cap_biomass['Capacity ours']

cap_node_ours = pd.concat([cap_node_ours, cap_biomass[cap_node_ours.columns]])

# Determine National Capacities
cap_national_ours = cap_node_ours.groupby(['Country', 'Technology']).sum()
cap_national_ours.reset_index().to_csv(c.clean_data_path + 'reporting/installed_capacities/ours_national.csv')
cap_node_ours.reset_index().to_csv(c.clean_data_path + 'clean_data/installed_capacities/capacities_node.csv')
cap_per_nuts.reset_index().to_csv(c.clean_data_path + 'clean_data/installed_capacities/capacities_nuts.csv')