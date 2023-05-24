from cases.NorthSea.read_input_data import *
import numpy as np
import pandas as pd

from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import src.plotting as pl
import matplotlib.pyplot as plt
"""
DATA INPUTS
- Electricity Demand data (from ENTSOE data)
- Installed capacities (from ENTSOE data)
- RE production profiles (Capacity factors + Installed Capacities)
- Network Matrixes

TODO
- define el network lengths
- define el network types with losses
- define gas network
- define gas network costs (repurposing for H2)
- define open loop storage technology
- define technology costs
- define H2 demand
- define SMR
- define new technologies
"""
year = 2030
scenario = 'GA'
climate_year = 2009


# NODES
node_data = r'cases/NorthSea/Nodes/Nodes.xlsx'
nodes = pd.read_excel(node_data, sheet_name='Nodes_used')
onshore_nodes = nodes[nodes['Type'] == 'onshore']['Node'].values.tolist()
offshore_nodes = nodes[nodes['Type'] == 'offshore']['Node'].values.tolist()
nodes = nodes['Node'].values.tolist()

# Define Topology
topology = dm.SystemTopology()
topology.define_time_horizon(year=2030, start_date='01-01 00:00', end_date='01-01 00:00', resolution=1)

# Carriers
topology.define_carriers(['electricity', 'gas', 'hydrogen'])

# Nodes
topology.define_nodes(nodes)

# Existing Technologies for countries
# THERE IS 2 GW TOO MUCH GAS IN NL!
installed_capacities = {}
for node in onshore_nodes:
    installed_capacities[node] = read_installed_capacity_eraa(node)
    topology.define_existing_technologies(node, installed_capacities[node]['Conventional'])

# Networks
network_data = read_network_data(topology.nodes)
topology.define_existing_network('electricitySimple', size=network_data['size'], distance=network_data['distance'])

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# Climate Data
# TODO: so far dummy data, not sure if 'good' data is needed
for node in nodes:
    data.read_climate_data_from_file(node, r'.\data\climate_data_onshore.txt')

# Generic Production Profiles Onshore
profiles = pd.read_csv(r'.\cases\NorthSea\ProductionProfiles\Production_Profiles' + str(climate_year) + '.csv', index_col=0)
for node in onshore_nodes:
    if node + '_tot' in profiles:
        data.read_production_profile(node, 'electricity', profiles[node + '_tot'].to_numpy(),1)

# Hydro Inflow
reservoir_inflow = pd.read_csv(r'.\cases\NorthSea\Hydro_Inflows\HydroInflowReservoir' + str(climate_year) + '.csv', index_col=0)
opencycle_inflow = pd.read_csv(r'.\cases\NorthSea\Hydro_Inflows\HydroInflowPump storage - Open Loop' + str(climate_year) + '.csv', index_col=0)
for node in nodes:
    if node in reservoir_inflow:
        if not node == 'NL00':
            data.read_hydro_natural_inflow(node, 'Storage_PumpedHydro_Reservoir', reservoir_inflow[node].values.tolist())
            data.read_hydro_natural_inflow(node, 'Storage_PumpedHydro_Open', opencycle_inflow[node].values.tolist())

# Generic Production Profiles Offshore
offshore_profiles = calculate_production_profiles_offshore(offshore_nodes)
for node in offshore_nodes:
    data.read_production_profile(node, 'electricity', offshore_profiles[node].to_numpy(),1)

# Demand Onshore
demand_el = read_demand_data_eraa(scenario, year, climate_year, 'Demand_Electricity')
for node in nodes:
    if node in demand_el:
        data.read_demand_data(node, 'electricity', demand_el[node].to_numpy())

demand_h2 = read_demand_data_eraa(scenario, year, climate_year, 'Demand_Hydrogen')
for node in nodes:
    if node in demand_h2:
        data.read_demand_data(node, 'hydrogen', demand_h2[node].to_numpy()/100)


# Import/Export of conventional fuels
import_carriers = {'gas': 100}
import_limit = np.ones(len(topology.timesteps)) * 10000

for node in onshore_nodes:
    for car in import_carriers:
        data.read_import_limit_data(node, car, import_limit)
        data.read_import_price_data(node, car, np.ones(len(topology.timesteps)) * import_carriers[car])

# Import Electricity
# import_carrier_price = {'electricity': 1000}
# import_limit = pd.read_excel(r'.\cases\NorthSea\Networks\ImportLimits.xlsx', index_col=0, sheet_name='ToPython')

import_carrier_price = {'electricity': 1000}

for node in onshore_nodes:
    for car in import_carrier_price:
        # data.read_import_limit_data(node, car, np.ones(len(topology.timesteps)) * import_limit[car][node])
        data.read_import_limit_data(node, car, np.ones(len(topology.timesteps)) * 100000)
        data.read_import_price_data(node, car, np.ones(len(topology.timesteps)) * import_carrier_price[car])


# Read technology data
tec_data_path = r'cases/NorthSea/Technology_Data/'
write_to_technology_data(tec_data_path, year)
data.read_technology_data(path =tec_data_path)

# Change charging and discharging efficiencies of hydro technologies
for node in onshore_nodes:
    storage_at_node = installed_capacities[node]['HydroStorage_charging']
    for storage in storage_at_node:
        data.technology_data[node][storage + '_existing'].fitted_performance.coefficients['charge_max'] = installed_capacities[node]['HydroStorage_charging'][storage]['max_charge']
        data.technology_data[node][storage + '_existing'].fitted_performance.coefficients['discharge_max'] = installed_capacities[node]['HydroStorage_charging'][storage]['max_discharge']

data.read_network_data()

# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()
configuration.solveroptions.solver = 'gurobi_persistent'
configuration.optimization.objective = 'pareto'

# Read data
energyhub = EnergyHub(data, configuration)
energyhub.quick_solve()

# pl.plot_balance_at_node(results.detailed_results[0], 'electricity')

# New technologies
new_tecs = pd.read_excel(r'.\cases\NorthSea\NewTechnologies\NewTechnologies.xlsx', index_col=0)
for node in nodes:
    # try:
    if not isinstance(new_tecs['Stage 1'][node], float):
        new_technologies = new_tecs['Stage 1'][node].split(', ')
        for tec in new_technologies:
            energyhub.add_technology_to_node(node, [tec], path =tec_data_path)
    # except:
    #     pass

energyhub.construct_balances()
results = energyhub.solve()
results.write_excel(r'user_Data/MultiCountry')
