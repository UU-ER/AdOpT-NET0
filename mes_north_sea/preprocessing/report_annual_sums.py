import pandas as pd
from mes_north_sea.preprocessing.utilities import to_latex

climate_year = 2009

hydro_inflows = pd.read_csv('./mes_north_sea/clean_data/hydro_inflows/hydro_inflows.csv', header=[0,1], index_col=0)
production = pd.read_csv('./mes_north_sea/clean_data/production_profiles_re/production_profiles_re.csv', header=[0,1], index_col=0)
demand = pd.read_csv('./mes_north_sea/clean_data/demand/TotalDemand_NT_'+ str(climate_year)+ '.csv', header=[0], index_col=0)
capacities = pd.read_csv('./mes_north_sea/clean_data/installed_capacities/capacities_node.csv', header=[0], index_col=0)

# Per node
production_node = production.sum().unstack()
production_node.columns = pd.MultiIndex.from_product([['Generation'], production_node.columns], names=('Type', 'Technology'))
production_node = production_node.swaplevel(axis=1)

hydro_inflows_node = hydro_inflows.sum().unstack()
hydro_inflows_node.rename(columns={col: col.replace(' (Energy)', '') for col in hydro_inflows_node.columns}, inplace=True)
hydro_inflows_node.columns = pd.MultiIndex.from_product([['Inflow'], hydro_inflows_node.columns], names=('Type', 'Technology'))
hydro_inflows_node = hydro_inflows_node.swaplevel(axis=1)

demand_node = pd.DataFrame(demand.sum(), columns=['Demand'])
demand_node.columns = pd.MultiIndex.from_product([['Demand'], demand_node.columns], names=('Type', 'Demand'))

capacities_node = capacities.drop(columns='index').groupby(['Node', 'Technology']).sum()
capacities_node = capacities_node.rename(index={'Solar': 'PV', 'Biofuels': 'Biomass', "Hydro - Run of River (Turbine)": 'Run of River', "Wind Offshore": 'Wind offshore', "Wind Onshore": 'Wind onshore'}, level='Technology')
capacities_node = capacities_node.query('Technology in ["PV", "Wind onshore", "Wind offshore", "Biomass", "Run of River"]')
capacities_node = capacities_node.rename(columns={'Capacity our work': 'Capacity'})
capacities_node = capacities_node.unstack()
capacities_node = capacities_node.swaplevel(axis=1)

capacity_factors_node = capacities_node * 8760
capacity_factors_node = capacity_factors_node.sort_index(axis=0).sort_index(axis=1)
production_node = production_node.sort_index(axis=0).sort_index(axis=1)
if capacity_factors_node.index.equals(production_node.index):
    print('works')
    capacity_factors_node = production_node.div(capacity_factors_node)
    capacity_factors_node = capacity_factors_node.droplevel('Type', axis=1).rename(columns={'Capacity': 'CF'}, level=1)

summary_table_node = pd.concat([production_node/1000000, hydro_inflows_node/1000000, capacity_factors_node, demand_node/1000000], axis=1)
summary_table_node = summary_table_node.sort_index(axis=1)
summary_table_node.to_csv('./mes_north_sea/reporting/demand_supply_node.csv')
# to_latex(summary_table_node,
#          'Capacity factors, total annual generation from non-dispatchable sources without curtailment and total demand per node in TWh',
#          './mes_north_sea/reporting/02_DemandSupply_node.tex',
#          rounding=2)

# Per country
production_country = production.sum().unstack().reset_index()
production_country['Country'] = production_country['Node'].str[0:2]
production_country = production_country.groupby('Country').sum()
production_country.columns = pd.MultiIndex.from_product([['Generation'], production_country.columns], names=('Type', 'Technology'))
production_country = production_country.swaplevel(axis=1)

hydro_inflows_country = hydro_inflows.sum().unstack().reset_index()
hydro_inflows_country.rename(columns={'index': 'Node'}, inplace=True)
hydro_inflows_country.rename(columns={col: col.replace(' (Energy)', '') for col in hydro_inflows_country.columns}, inplace=True)
hydro_inflows_country['Country'] = hydro_inflows_country['Node'].str[0:2]
hydro_inflows_country = hydro_inflows_country.groupby('Country').sum()
hydro_inflows_country.columns = pd.MultiIndex.from_product([['Inflow'], hydro_inflows_country.columns], names=('Type', 'Technology'))
hydro_inflows_country = hydro_inflows_country.swaplevel(axis=1)

demand_country = pd.DataFrame(demand.sum(), columns=['Demand']).reset_index()
demand_country['Country'] = demand_country['index'].str[0:2]
demand_country = demand_country.groupby('Country').sum()
demand_country.columns = pd.MultiIndex.from_product([['Demand'], demand_country.columns], names=('Type', 'Demand'))

capacities_country = capacities.drop(columns='index').groupby(['Country', 'Technology']).sum()
capacities_country = capacities_country.rename(index={'Solar': 'PV', 'Biofuels': 'Biomass', "Hydro - Run of River (Turbine)": 'Run of River', "Wind Offshore": 'Wind offshore', "Wind Onshore": 'Wind onshore'}, level='Technology')
capacities_country = capacities_country.query('Technology in ["PV", "Wind onshore", "Wind offshore", "Biomass", "Run of River"]')
capacities_country = capacities_country.rename(columns={'Capacity our work': 'Capacity'})
capacities_country = capacities_country.unstack()
capacities_country = capacities_country.swaplevel(axis=1)

capacity_factors_country = capacities_country * 8760
capacity_factors_country = capacity_factors_country.sort_index(axis=0).sort_index(axis=1)
production_country = production_country.sort_index(axis=0).sort_index(axis=1)
if capacity_factors_country.index.equals(production_country.index):
    print('works')
    capacity_factors_country = production_country.div(capacity_factors_country)
    capacity_factors_country = capacity_factors_country.droplevel('Type', axis=1).rename(columns={'Capacity': 'CF'}, level=1)

summary_table_country = pd.concat([production_country/1000000, hydro_inflows_country/1000000, capacity_factors_country, demand_country/1000000], axis=1)
summary_table_country = summary_table_country.sort_index(axis=1)
summary_table_country.to_csv('./mes_north_sea/reporting/demand_supply_country.csv')
to_latex(summary_table_country.T,
         'Capacity factors, total annual generation from non-dispatchable sources without curtailment and total demand per country in TWh',
         './mes_north_sea/reporting/02_DemandSupply_country.tex',
         rounding=2)

