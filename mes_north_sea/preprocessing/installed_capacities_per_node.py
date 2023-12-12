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

# National Capacity TYNDP
cap_nonre = {}
cap_nonre['Gas'] = 0
cap_nonre['Coal & Lignite'] = 0
cap_nonre['Nuclear'] = 0

cap_re = {}
cap_re['Biofuels'] = 0
cap_re['Solar'] = 0
cap_re['Wind Onshore'] = 0
cap_re['Wind Offshore'] = 0


# NON RE (Coal, Gas, Nuclear)
cap_national = pd.DataFrame(columns=cap_nonre.keys())
for country in c.countries:
    cap_nonre = {key: 0 for key in cap_nonre}  # Reset cap for each country
    for bidding_zone in c.countries[country]:
        tyndp_installed_at_bidding_zone = tyndp_installed[tyndp_installed['Node'] == bidding_zone]
        tyndp_installed_at_bidding_zone = tyndp_installed_at_bidding_zone.groupby('Fuel').sum()
        for fuel_type in cap_nonre.keys():
            cap_nonre[fuel_type] = cap_nonre[fuel_type] + tyndp_installed_at_bidding_zone.at[fuel_type, 'Value']

    cap_national = cap_national.append(pd.Series(cap_nonre, name=country))

# Determine Keys from PyPsa Data
pypsa_installed_capacities_per_node = pd.read_csv(c.load_path_pypsa_cap)
pypsa_installed_capacities_per_country = pypsa_installed_capacities_per_node.groupby('CNTR_CODE').sum()

cap_node = pd.DataFrame(columns=cap_nonre.keys())
for node in pypsa_installed_capacities_per_node.iterrows():
    cap_nonre = {key: 0 for key in cap_nonre}  # Reset cap for each country
    for tec in cap_nonre:
        country = node[1]['CNTR_CODE']
        national_cap_tyndp = cap_national.at[country, tec]
        national_cap_pypsa = pypsa_installed_capacities_per_country.at[country, tec]
        nodal_cap_pypsa = node[1][tec]
        cap_nonre[tec] = nodal_cap_pypsa/national_cap_pypsa * national_cap_tyndp

    cap_node = cap_node.append(pd.Series(cap_nonre, name=node[1]['NODE_NAME']))

cap_node = cap_node.fillna(0)
cap_node.at['BE2', 'Nuclear'] = 2077

cap_node.to_csv(c.savepath_cap_per_node)



