from cases.NorthSea_helpers.read_input_data import *
import numpy as np
import pandas as pd

from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub

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

aggregated_nodes = {'onNL_C': ['onNL_SE', 'onNL_CE', 'onNL_E']}

# NODES
node_data = r'cases/NorthSea_v2/Nodes/Nodes.xlsx'
nodes = pd.read_excel(node_data, sheet_name='Nodes_used')
onshore_nodes = nodes[nodes['Type'] == 'onshore']['Node'].values.tolist()
offshore_nodes = nodes[nodes['Type'] == 'offshore']['Node'].values.tolist()
nodes = nodes['Node'].values.tolist()

# Define Topology
topology = dm.SystemTopology()
topology.define_time_horizon(year=2030, start_date='01-01 00:00', end_date='01-01 23:00', resolution=1)

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

# New technologies
new_tecs = pd.read_excel(r'.\cases\NorthSea_v2\NewTechnologies\NewTechnologies.xlsx', index_col=0)
stage = 'All'
for node in nodes:
    if not isinstance(new_tecs[stage][node], float):
        new_technologies = new_tecs[stage][node].split(', ')
        topology.define_new_technologies(node,new_technologies)

# YOU NEED TO REDO THE NETWORKS FOR THIS CASE!
# Networks - Electricity
network_data = read_network_data(topology.nodes, 'cases/NorthSea_v2/Networks/NetworkDataElectricity_existing.xlsx', 1)
topology.define_existing_network('electricityAC', size=network_data['size'], distance=network_data['distance'])

# Networks - New Electricity
network_data = read_network_data(topology.nodes, './cases/NorthSea_v2/Networks/NetworkDataElectricity_AC.xlsx', 0)
topology.define_new_network('electricityAC_int', connections=network_data['connection'], distance=network_data['distance'])

network_data = read_network_data(topology.nodes, './cases/NorthSea_v2/Networks/NetworkDataElectricity_DC.xlsx', 0)
topology.define_new_network('electricityDC_int', connections=network_data['connection'], distance=network_data['distance'])

# Networks - Hydrogen
network_data = read_network_data(topology.nodes, './cases/NorthSea_v2/Networks/NetworkDataHydrogen.xlsx', 0)
topology.define_new_network('hydrogenPipeline_int', connections=network_data['connection'], distance=network_data['distance'])

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# Climate Data
# TODO: so far dummy data, not sure if 'good' data is needed
for node in nodes:
    data.read_climate_data_from_file(node, r'.\data\climate_data_onshore.txt')

# Generic Production Profiles Onshore
profiles = pd.read_csv(r'.\cases\NorthSea_v2\ProductionProfiles\Production_Profiles' + str(climate_year) + '.csv', index_col=0)
for node in aggregated_nodes:
    list_of_nodes = [s + '_tot' for s in aggregated_nodes[node]]
    profiles[node] = profiles[list_of_nodes].sum(axis = 1)
for node in onshore_nodes:
    if node + '_tot' in profiles:
        data.read_production_profile(node, 'electricity', profiles[node + '_tot'].to_numpy(),1)

# Hydro Inflow
reservoir_inflow = pd.read_csv(r'.\cases\NorthSea_v2\Hydro_Inflows\HydroInflowReservoir' + str(climate_year) + '.csv', index_col=0)
opencycle_inflow = pd.read_csv(r'.\cases\NorthSea_v2\Hydro_Inflows\HydroInflowPump storage - Open Loop' + str(climate_year) + '.csv', index_col=0)
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
for node in aggregated_nodes:
    demand_el[node] = demand_el[aggregated_nodes[node]].sum(axis = 1)
for node in nodes:
    if node in demand_el:
        data.read_demand_data(node, 'electricity', demand_el[node].to_numpy())

demand_h2 = read_demand_data_eraa(scenario, year, climate_year, 'Demand_Hydrogen')
for node in aggregated_nodes:
    demand_h2[node] = demand_h2[aggregated_nodes[node]].sum(axis = 1)
for node in nodes:
    if node in demand_h2:
        data.read_demand_data(node, 'hydrogen', demand_h2[node].to_numpy()/100)


# Import/Export of conventional fuels
import_carriers = {'gas': 100}
import_limit = np.ones(len(topology.timesteps)) * 100000

for node in onshore_nodes:
    for car in import_carriers:
        data.read_import_limit_data(node, car, import_limit)
        data.read_import_price_data(node, car, np.ones(len(topology.timesteps)) * import_carriers[car])

# Import Electricity
import_carrier_price = {'electricity': 1000}
import_limit = pd.read_excel(r'.\cases\NorthSea_v2\Networks\ImportLimits.xlsx', index_col=0, sheet_name='ToPython')
factor = 100

for node in onshore_nodes:
    for car in import_carrier_price:
        data.read_import_limit_data(node, car, np.ones(len(topology.timesteps)) * import_limit[car][node] * factor)
        # data.read_import_limit_data(node, car, np.ones(len(topology.timesteps)) * 100000)
        data.read_import_price_data(node, car, np.ones(len(topology.timesteps)) * import_carrier_price[car])
        data.read_import_emissionfactor_data(node, car, np.ones(len(topology.timesteps)) * 0.3)


# Read technology data
tec_data_path = r'../NorthSea_v2/Technology_Data/'
write_to_technology_data(tec_data_path, year)
data.read_technology_data(path =tec_data_path)

# max onshore electrolysis
for node in onshore_nodes:
    data.technology_data[node]['Electrolyser_PEM'].size_max = 2000
    data.technology_data[node]['FuelCell'].size_max = 2000

# Change charging and discharging efficiencies of hydro technologies
for node in onshore_nodes:
    storage_at_node = installed_capacities[node]['HydroStorage_charging']
    for storage in storage_at_node:
        data.technology_data[node][storage + '_existing'].fitted_performance.coefficients['charge_max'] = installed_capacities[node]['HydroStorage_charging'][storage]['max_charge']
        data.technology_data[node][storage + '_existing'].fitted_performance.coefficients['discharge_max'] = installed_capacities[node]['HydroStorage_charging'][storage]['max_discharge']

data.read_network_data()

# Have different emission limits
# CONFIGURATION COSTS
configuration = ModelConfiguration()
configuration.solveroptions.solver = 'gurobi_persistent'
configuration.optimization.objective = 'costs'
configuration.solveroptions.mipgap = 0.01
configuration.solveroptions.lpwarmstart = 1
configuration.solveroptions.numericfocus = 3
configuration.optimization.save_log_files = 1
configuration.optimization.monte_carlo.on = 0
configuration.optimization.monte_carlo.N = 5
configuration.optimization.typicaldays = 0

emissions = []

# Read data
energyhub = EnergyHub(data, configuration)
results = energyhub.quick_solve()

emissions.append(energyhub.model.var_emissions_net.value)

# CONFIGURATION EMISSIONS
energyhub.configuration.optimization.objective = 'emissions_minC'
energyhub.solve()
results.write_excel('//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20230614/MES_NS_LimitedOnshoreElectrolysis')