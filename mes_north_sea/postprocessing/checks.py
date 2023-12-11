from pathlib import Path
import os
import pandas as pd
import numpy as np

# Check min, max, average demand per node
climate_year = 2008
demand = pd.read_csv('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/clean_data/demand/TotalDemand_NT_' + str(climate_year) + '.csv', index_col=0)
demand_statistics = pd.DataFrame()
demand_statistics['sum demand (TWh)'] = demand.sum() * 10 ** -6
demand_statistics['max demand (MWh)'] = demand.max()
demand_statistics['min demand (MWh)'] = demand.min()


demand_statistics['sum res demand (TWh)'] = None
demand_statistics['max res demand (MWh)'] = None
demand_statistics['min res demand (MWh)'] = None
for node in demand:
    gen_production = pd.read_csv(
        'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/clean_data/production_profiles_re/' + node + '_' + str(
            climate_year) + '.csv', index_col=0)
    residual_demand_series = demand[node].values - gen_production['total'].values
    demand_statistics.at[node,'sum res demand (TWh)'] = residual_demand_series.sum() * 10 ** -6
    demand_statistics.at[node,'max res demand (MWh)'] = residual_demand_series.max()
    demand_statistics.at[node,'min res demand (MWh)'] = residual_demand_series.min()

demand_statistics.to_csv('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/reporting/demand_statistics.csv')

# Compute total demand profile/generic production
demand = pd.read_csv('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/clean_data/demand/TotalDemand_NT_' + str(climate_year) + '.csv', index_col=0)
demand['total_demand'] = demand.sum(axis=1)

supply = pd.DataFrame({'total': np.zeros(8760)})
nodes = pd.read_excel('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/clean_data/nodes/nodes.xlsx', sheet_name='Nodes_all')
for node in nodes['Node']:
    print(node)
    gen_production = pd.read_csv(
        'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/clean_data/production_profiles_re/' + node + '_' + str(
            climate_year) + '.csv', index_col=0)
    supply[node] = gen_production['total'].values
    supply['total'] = supply['total'].values + gen_production['total'].values

demand['RE_supply'] = supply['total']
demand.to_csv('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/reporting/aggregated_demand_supply.csv')
supply.to_csv('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/reporting/aggregated_supply.csv')


#
# result_path = Path('//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20231201/20231206111512_Baseline/')
# node_path = Path.joinpath(result_path, 'nodes')
# nodes = [f.name for f in os.scandir(node_path) if f.is_dir()]
# 
# metrics = {}
# 
# # Max Imports
# metrics['max_import'] = {}
# for node in nodes:
#     energybalance = pd.read_excel(Path.joinpath(node_path, node, 'Energybalance.xlsx'), sheet_name='electricity',
#                                   index_col=0)
#     metrics['max_import'][node] = max(energybalance['Import'])
# 
# 
# pd.DataFrame(metrics['max_import'])
