from cases.NorthSea.read_input_data import *
import numpy as np
import pandas as pd

from src.model_configuration import ModelConfiguration
import src.data_management as dm
from src.energyhub import EnergyHub
import src.plotting as pl


# Get data for node
nodes = ['BE']
climate_year = 2015

electricity_demand = {}
cap_factors = {}
installed_capacities = {}

for node in nodes:
    electricity_demand[node] = read_demand_data_eraa(2015, node)
    cap_factors[node] = read_capacity_factors_eraa(climate_year, node)
    installed_capacities[node] = read_installed_capacity_eraa(node)

topology = dm.SystemTopology()
topology.define_time_horizon(year=2030, start_date='01-01 00:00', end_date='12-31 23:00', resolution=1)

# Carriers
topology.define_carriers(['electricity', 'gas', 'hydrogen'])

# Nodes
topology.define_nodes(nodes)

# Existing Technologies
for node in nodes:
    for tec in installed_capacities[node]['Conventional']:
        if installed_capacities[node]['Conventional'][tec] > 0:
            topology.define_existing_technologies(node, installed_capacities[node]['Conventional'])

# New Technologies
# TODO

# Networks
# TODO

# Initialize instance of DataHandle
data = dm.DataHandle(topology)

# Climate Data
# TODO: so far dummy data
for node in nodes:
    data.read_climate_data_from_file(node, r'.\data\climate_data_onshore.txt')

# Generic Production Profiles
node = nodes[0]
profile = pd.DataFrame(index=cap_factors[node]['PV'].index)
for node in nodes:
    profile[node] = 0
    for series in cap_factors[node]:
        profile[node] = profile[node] + cap_factors[node][series] * installed_capacities[node]['RE'][series]
        data.read_production_profile(node, 'electricity', profile[node].to_numpy(),1)

# Demand
for node in nodes:
    data.read_demand_data(node, 'electricity', electricity_demand[node].to_numpy())

# Import/Export of conventional fuels
import_carriers = {'gas': 100}
import_limit = np.ones(len(topology.timesteps)) * 500000

for node in nodes:
    for car in import_carriers:
        data.read_import_limit_data(node, car, import_limit)
        data.read_import_price_data(node, car, np.ones(len(topology.timesteps)) * import_carriers[car])

# Import Electricity
import_carriers = {'electricity': 1000}
import_limit = np.ones(len(topology.timesteps)) * 200000
for node in nodes:
    for car in import_carriers:
        data.read_import_limit_data(node, car, import_limit)
        data.read_import_price_data(node, car, np.ones(len(topology.timesteps)) * import_carriers[car])

# Read data
data.read_technology_data()
data.read_network_data()

# SAVING/LOADING DATA FILE
configuration = ModelConfiguration()

# # Read data
energyhub = EnergyHub(data, configuration)
results = energyhub.quick_solve()
# results.write_excel(r'user_Data/' + nodes[0])

pl.plot_balance_at_node(results.detailed_results[0], 'electricity')