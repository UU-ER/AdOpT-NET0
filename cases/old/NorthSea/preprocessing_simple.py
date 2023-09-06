import src.data_management as dm
import numpy as np
import pandas as pd
import pickle
from src.data_management.components.fit_technology_performance import perform_fitting_WT


def create_data():
    topology = dm.SystemTopology()
    topology.define_time_horizon(year=2022, start_date='01-01 00:00', end_date='01-31 23:00', resolution=1)

    # Carriers
    topology.define_carriers(['electricity', 'gas', 'nuclear', 'hydrogen'])

    # Get output from wind parks
    load_path = 'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/data/climate_data/'
    data_path = 'E:/00_Data/Windpark Data/00_Netherlands/WindParkData.csv'
    park_data = pd.read_csv(data_path)
    park_data_2030 = park_data[park_data.YEAR <= 2030]
    offshore_nodes = park_data_2030['Transformer Platform'].unique().tolist()
    offshore_nodes = [offshore_nodes[2]]

    # Output per transformer station (easy)
    output_per_substation = pd.DataFrame()
    for station in offshore_nodes:
        parks_at_station = park_data_2030[park_data_2030['Transformer Platform'] == station]
        output = np.zeros(8760)
        for idx in parks_at_station.index:
            name = parks_at_station['naam'][idx]
            name = name.replace('(', '')
            name = name.replace(')', '')
            name = name.replace(' ', '_')
            load_file = load_path + name

            with open(load_file, 'rb') as handle:
                data = pickle.load(handle)

            cap_factors = perform_fitting_WT(data['dataframe'],'WindTurbine_Offshore_11000',0)
            cap_factors = cap_factors.coefficients['capfactor']
            output = output + cap_factors * parks_at_station['POWER_MW'][idx]
        output_per_substation[station] = output

    # Nodes 2030
    onshore_nodes = ['onNL_SW',
                     'onNL_CW',
                     'onNL_CE',
                     'onNL_SE',
                     'onNL_SW',
                     'onNL_NW',
                     'onNL_E',
                     'onNL_NE',
                     ]
    onshore_nodes = [onshore_nodes[0]]
    nodes = onshore_nodes + offshore_nodes

    topology.define_nodes(nodes)

    # Technologies at onshore nodes
    # https://transparency.entsoe.eu/generation/r2/installedGenerationCapacityAggregation/show?name=&defaultValue=true&viewType=TABLE&areaType=BZN&atch=false&dateTime.dateTime=01.01.2023+00:00|UTC|YEAR&dateTime.endDateTime=01.01.2024+00:00|UTC|YEAR&area.values=CTY|10YNL----------L!BZN|10YNL----------L&productionType.values=B01&productionType.values=B02&productionType.values=B03&productionType.values=B04&productionType.values=B05&productionType.values=B06&productionType.values=B07&productionType.values=B08&productionType.values=B09&productionType.values=B10&productionType.values=B11&productionType.values=B12&productionType.values=B13&productionType.values=B14&productionType.values=B20&productionType.values=B15&productionType.values=B16&productionType.values=B17&productionType.values=B18&productionType.values=B19
    eta_nuc = 0.33
    eta_coal = 0.55
    eta_gas = 0.55

    topology.define_existing_technologies('onNL_SW', {
                                                'Photovoltaic': 12000,
                                                'WindTurbine_Onshore_4000': 6200 / 4,
                                                'PowerPlant_Nuclear': 485/eta_nuc,
                                                'PowerPlant_Gas': (455+2*435)/eta_gas})

    for node in onshore_nodes:
        new_technologies = ['Storage_Battery', 'Electrolyser', 'Storage_H2', 'FuelCell', 'SteamReformer']
        topology.define_new_technologies(node, new_technologies)


    # Networks
    distance = dm.create_empty_network_matrix(topology.nodes)
    size = dm.create_empty_network_matrix(topology.nodes)

    data_path = './cases/NorthSea/Networks_Connections.xlsx'
    size_matrix = pd.read_excel(data_path, index_col=0)

    for node1 in nodes:
        for node2 in nodes:
            if pd.isna(size_matrix[node1][node2]) and not pd.isna(size_matrix[node2][node1]):
                size_matrix[node1][node2] = size_matrix[node2][node1]
            if not pd.isna(size_matrix[node1][node2]) and pd.isna(size_matrix[node2][node1]):
                size_matrix[node2][node1] = size_matrix[node1][node2]

    for node1 in nodes:
        for node2 in nodes:
            distance.at[node1, node2] = 50

            if pd.isna(size_matrix[node1][node2]):
                size.at[node1, node2] = 0
                size.at[node2, node1] = 0
            else:
                size.at[node1, node2] = size_matrix[node1][node2]
                size.at[node2, node1] = size_matrix[node1][node2]
    topology.define_existing_network('electricitySimple', size= size,distance=distance)
    topology.define_existing_network('hydrogenPipelineOffshore', size= size,distance=distance)

    # Initialize instance of DataHandle
    data = dm.DataHandle(topology)

    # Climate data

    # Production at offshore nodes
    for node in offshore_nodes:
        genericoutput = output_per_substation[node][0:len(topology.timesteps)].tolist()
        data.read_production_profile(node, 'electricity', genericoutput, 1)

    # DEMAND
    electricity_demand = pd.read_csv("./data/demand_data/Electricity Load NL2018.csv")
    electricity_demand = electricity_demand.iloc[:, 2].to_numpy()
    electricity_demand = np.nanmean(np.reshape(electricity_demand, (-1, 4)), axis=1)
    electricity_demand = electricity_demand[0:len(topology.timesteps)]
    # electricity_demand = np.ones(len(topology.timesteps))

    for node in nodes:
        lat = 52
        lon = 5.16
        # data.read_climate_data_from_api(node, lon, lat, year=2022, save_path='.\data\climate_data_'+ node +'.txt')
        # else:
        data.read_climate_data_from_file(node, '.\data\climate_data_'+ node +'.txt')

    # Replacing nan values
    ok = ~np.isnan(electricity_demand)
    xp = ok.ravel().nonzero()[0]
    fp = electricity_demand[~np.isnan(electricity_demand)]
    x = np.isnan(electricity_demand).ravel().nonzero()[0]
    electricity_demand[np.isnan(electricity_demand)] = np.interp(x, xp, fp)
    h2_demand = np.ones(len(topology.timesteps)) * 200

    import_limit = np.ones(len(topology.timesteps)) * 50000
    data.read_import_limit_data('onNL_SW', 'gas', import_limit)

    data.read_import_limit_data('onNL_SW', 'nuclear', import_limit)


    for node in onshore_nodes:
        data.read_demand_data(node, 'electricity', electricity_demand/3)
        data.read_demand_data(node, 'hydrogen', h2_demand)

        gas_price = np.ones(len(topology.timesteps)) * 100
        coal_price = np.ones(len(topology.timesteps)) * 80
        nuclear_price = np.ones(len(topology.timesteps)) * 10
        data.read_import_price_data(node, 'gas', gas_price)
        # data.read_import_price_data(node, 'coal', coal_price)
        data.read_import_price_data(node, 'nuclear', nuclear_price)

    return data