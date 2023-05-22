import src.data_management as dm
import numpy as np
import pandas as pd
import pickle
from src.data_management.components.fit_technology_performance import perform_fitting_WT
from cases.NorthSea.read_input_data import *
import matplotlib.pyplot as plt
#
other_regions = ['DK1', 'DE', 'BE', 'NOS', 'UK']
climate_year = 2015

# DEMANDS
save_path = r'cases/NorthSea/Demand_Electricity/Scaled_Demand.csv'

# Demand Data NL
region = 'NL'

scaling_factors = {'Res': 2, 'Ind': 0.3}
# scaling_factors = {'Res': 2.5, 'Ind': 0}
NL_national_profile = read_demand_data_eraa(climate_year, region)
demand_corrected = scale_demand_data(NL_national_profile, scaling_factors)
demand_at_nodes = aggregate_provinces_to_node(demand_corrected)
NL_regions = demand_at_nodes.keys()
demand_at_nodes['NL_total'] = demand_at_nodes.sum(axis=1)

# Other regions
for region in other_regions:
    demand_at_nodes[region] = read_demand_data_eraa(climate_year, region)

demand_at_nodes.to_csv(save_path)
#
# # GENERIC PRODUCTION PROFILES

#
#
#
#
# def create_data():
#     topology = dm.SystemTopology()
#     topology.define_time_horizon(year=2022, start_date='01-01 00:00', end_date='01-31 23:00', resolution=1)
#
#     # Carriers
#     topology.define_carriers(['electricity', 'gas', 'nuclear', 'coal'])
#
#     # Get output from wind parks
#     load_path = '/data/climate_data/'
#     data_path = 'E:/00_Data/Windpark Data/00_Netherlands/WindParkData.csv'
#     park_data = pd.read_csv(data_path)
#     park_data_2030 = park_data[park_data.YEAR <= 2030]
#     offshore_nodes = park_data_2030['Transformer Platform'].unique().tolist()
#
#     # Output per transformer station (easy)
#     output_per_substation = pd.DataFrame()
#     for station in offshore_nodes:
#         parks_at_station = park_data_2030[park_data_2030['Transformer Platform'] == station]
#         output = np.zeros(8760)
#         for idx in parks_at_station.index:
#             name = parks_at_station['naam'][idx]
#             name = name.replace('(', '')
#             name = name.replace(')', '')
#             name = name.replace(' ', '_')
#             load_file = load_path + name
#
#             with open(load_file, 'rb') as handle:
#                 data = pickle.load(handle)
#
#             cap_factors = perform_fitting_WT(data['dataframe'],'WindTurbine_Offshore_11000',0)
#             cap_factors = cap_factors.coefficients['capfactor']
#             output = output + cap_factors * parks_at_station['POWER_MW'][idx]
#         output_per_substation[station] = output
#
#     # Nodes 2030
#     onshore_nodes = ['NL_on_Borssele',
#                      'NL_on_Rilland',
#                      'NL_on_Brabant',
#                      'NL_on_South',
#                      'NL_on_Holland_S',
#                      'NL_on_Holland_N',
#                      'NL_on_East',
#                      'NL_on_North',
#                      ]
#     nodes = onshore_nodes + offshore_nodes
#
#     topology.define_nodes(nodes)
#
#     # Technologies at onshore nodes
#     # https://transparency.entsoe.eu/generation/r2/installedGenerationCapacityAggregation/show?name=&defaultValue=true&viewType=TABLE&areaType=BZN&atch=false&dateTime.dateTime=01.01.2023+00:00|UTC|YEAR&dateTime.endDateTime=01.01.2024+00:00|UTC|YEAR&area.values=CTY|10YNL----------L!BZN|10YNL----------L&productionType.values=B01&productionType.values=B02&productionType.values=B03&productionType.values=B04&productionType.values=B05&productionType.values=B06&productionType.values=B07&productionType.values=B08&productionType.values=B09&productionType.values=B10&productionType.values=B11&productionType.values=B12&productionType.values=B13&productionType.values=B14&productionType.values=B20&productionType.values=B15&productionType.values=B16&productionType.values=B17&productionType.values=B18&productionType.values=B19
#     eta_nuc = 0.33
#     eta_coal = 0.55
#     eta_gas = 0.55
#
#     topology.define_existing_technologies('NL_on_Borssele', {'PowerPlant_Nuclear': 485/eta_nuc,
#                                                  'PowerPlant_Gas': (455+2*435)/eta_gas})
#     topology.define_existing_technologies('NL_on_Brabant', {'PowerPlant_Gas': 766/eta_gas})
#     topology.define_existing_technologies('NL_on_South', {'PowerPlant_Gas': (1304+209)/eta_gas})
#     topology.define_existing_technologies('NL_on_Holland_S', {'PowerPlant_Coal': (1070+731)/eta_coal,
#                                                  'PowerPlant_Gas': 2602/eta_gas})
#     topology.define_existing_technologies('NL_on_Holland_N', {'PowerPlant_Gas': 3338/eta_gas})
#     topology.define_existing_technologies('NL_on_North', {'PowerPlant_Coal': 1580/eta_coal,
#                                                           'PowerPlant_Gas': 3965/eta_gas})
#
#
#
#
#     # for node in onshore_nodes:
#     #     topology.define_existing_technologies(node, {'Photovoltaic': 19000,
#     #                                                  'WindTurbine_Onshore_4000': 6200/4,
#     #                                                  'PowerPlant_Nuclear': 486/0.5,
#     #                                                  'PowerPlant_Coal': 4000/0.5,
#     #                                                  'PowerPlant_Gas': 19000/0.5})
#     #     new_technologies = ['Storage_Battery', 'Electrolyser']
#     #     topology.define_new_technologies(node, new_technologies)
#
#
#     # Networks
#     distance = dm.create_empty_network_matrix(topology.nodes)
#     size = dm.create_empty_network_matrix(topology.nodes)
#
#     data_path = './cases/NorthSea/Networks_Connections.xlsx'
#     size_matrix = pd.read_excel(data_path, index_col=0)
#
#     for node1 in nodes:
#         for node2 in nodes:
#             if pd.isna(size_matrix[node1][node2]) and not pd.isna(size_matrix[node2][node1]):
#                 size_matrix[node1][node2] = size_matrix[node2][node1]
#             if not pd.isna(size_matrix[node1][node2]) and pd.isna(size_matrix[node2][node1]):
#                 size_matrix[node2][node1] = size_matrix[node1][node2]
#
#     for node1 in nodes:
#         for node2 in nodes:
#             distance.at[node1, node2] = 50
#
#             if pd.isna(size_matrix[node1][node2]):
#                 size.at[node1, node2] = 0
#                 size.at[node2, node1] = 0
#             else:
#                 size.at[node1, node2] = size_matrix[node1][node2]
#                 size.at[node2, node1] = size_matrix[node1][node2]
#     topology.define_existing_network('electricitySimple', size= size,distance=distance)
#
#     # Initialize instance of DataHandle
#     data = dm.DataHandle(topology)
#
#     # Climate data
#
#     # Production at offshore nodes
#     for node in offshore_nodes:
#         genericoutput = output_per_substation[node][0:len(topology.timesteps)].tolist()
#         data.read_production_profile(node, 'electricity', genericoutput,1)
#
#     # DEMAND
#     electricity_demand = pd.read_csv("./data/demand_data/Electricity Load NL2018.csv")
#     electricity_demand = electricity_demand.iloc[:, 2].to_numpy()
#     electricity_demand = np.nanmean(np.reshape(electricity_demand, (-1, 4)), axis=1)
#     electricity_demand = electricity_demand[0:len(topology.timesteps)]
#     # electricity_demand = np.ones(len(topology.timesteps))
#
#     for node in nodes:
#         lat = 52
#         lon = 5.16
#         data.read_climate_data_from_api(node, lon, lat, year=2022, save_path='.\data\climate_data_onshore.txt')
#
#     # Replacing nan values
#     ok = ~np.isnan(electricity_demand)
#     xp = ok.ravel().nonzero()[0]
#     fp = electricity_demand[~np.isnan(electricity_demand)]
#     x = np.isnan(electricity_demand).ravel().nonzero()[0]
#     electricity_demand[np.isnan(electricity_demand)] = np.interp(x, xp, fp)
#     h2_demand = np.ones(len(topology.timesteps)) * 1000
#
#     import_limit = np.ones(len(topology.timesteps)) * 50000
#     data.read_import_limit_data('NL_on_Borssele', 'gas', import_limit)
#     data.read_import_limit_data('NL_on_Brabant', 'gas', import_limit)
#     data.read_import_limit_data('NL_on_South', 'gas', import_limit)
#     data.read_import_limit_data('NL_on_Holland_S', 'gas', import_limit)
#     data.read_import_limit_data('NL_on_Holland_N', 'gas', import_limit)
#     data.read_import_limit_data('NL_on_North', 'gas', import_limit)
#
#     data.read_import_limit_data('NL_on_Borssele', 'nuclear', import_limit)
#
#     data.read_import_limit_data('NL_on_Holland_S', 'coal', import_limit)
#     data.read_import_limit_data('NL_on_North', 'coal', import_limit)
#
#     for node in onshore_nodes:
#         data.read_demand_data(node, 'electricity', electricity_demand/len(onshore_nodes))
#         # data.read_demand_data(node, 'hydrogen', h2_demand)
#
#         gas_price = np.ones(len(topology.timesteps)) * 100
#         coal_price = np.ones(len(topology.timesteps)) * 80
#         nuclear_price = np.ones(len(topology.timesteps)) * 10
#         data.read_import_price_data(node, 'gas', gas_price)
#         data.read_import_price_data(node, 'coal', coal_price)
#         data.read_import_price_data(node, 'nuclear', nuclear_price)
#
#     return data