import pandas as pd
import copy
import numpy as np
from pathlib import Path

from .utilities import *
from .import_data import import_jrc_climate_data
from ..utilities import ModelInformation
from ..components.networks import *


class DataHandle:
    """
    Data Handle for loading and performing operations on input data.

    The Data Handle class allows data import and modifications of input data to an instance of the energyhub class.
    The constructor of the class takes an instance of the class
    :func:`~src.data_management.handle_topology.SystemTopology` as an input. The DataHandle class is structured
    as follows:
    - node_data contains (mainly time-dependent) data on all nodes, e.g. demand, prices, import/export limit,...
    - technology_data contains data on all technologies. The data is read for all technologies in the topology
      with the function :func:`~src.data_management.handle_input_data.read_technology_data()`
    - network_data contains data on the networks. Similar to technolog_data, this is read with the respective function
      :func:`~src.data_management.handle_input_data.read_network_data()`
    - topology: contains data on the systems topology (see class
      :func:`~src.data_management.handle_topology.SystemTopology`)
    """
    def __init__(self, topology):
        """
        Constructor

        Initializes a data handle class and completes demand data for each carrier used (i.e. sets it to zero for all \
        time steps)

        :param SystemTopology topology: SystemTopology Class :func:`~src.data_management.handle_topology.SystemTopology`
        """
        self.topology = topology
        self.global_data = {}
        self.node_data = {}
        self.technology_data = {}
        self.network_data = {}
        self.model_information = ModelInformation()

        self.global_data = GlobalData(topology)


        # Initialize Node data
        for node in self.topology.nodes:
            self.node_data[node] = NodeData(topology)

    def read_climate_data_from_api(self, node, lon, lat, alt=10, dataset='JRC', year='typical_year', save_path=None):
        """
        Reads in climate data for a full year

        Reads in climate data for a full year from the specified source \
        (currently only `JRC PVGIS <https://re.jrc.ec.europa.eu/pvg_tools/en/>`_

        :param str node: node as specified in the topology
        :param float lon: longitude of node - the api will read data for this location
        :param float lat: latitude of node - the api will read data for this location
        :param str dataset: dataset to import from, can be JRC (only onshore) or ERA5 (global)
        :param int year: optional, needs to be in range of data available. If nothing is specified, a typical year \
        will be loaded
        :param str save_path: Can save climate data for later use to the specified path
        :return: self at ``self.node_data[node]['climate_data']``
        """
        if dataset == 'JRC':
            data = import_jrc_climate_data(lon, lat, year, alt)
        else:
            raise Exception('Other APIs are not available')

        # Save
        if save_path is not None:
            save_object(data, Path(save_path))

        # Match with timesteps
        data['dataframe'] = data['dataframe'].loc[self.topology.timesteps]

        # Write to DataHandle
        self.node_data[node].data['climate_data'] = data['dataframe']
        self.node_data[node].location.lon = data['longitude']
        self.node_data[node].location.lat = data['latitude']
        self.node_data[node].location.altitude = data['altitude']

    def read_climate_data_from_file(self, node, load_path):
        """
        Reads climate data from file

        Reads previously saved climate data (imported and saved with :func:`~read_climate_data_from_api`) from a file to \
        the respective node. This can save time, if api imports take too long

        :param str node: node as specified in the topology
        :param str load_path: path of climate data file
        :return: self at ``self.node_data[node]['climate_data']``
        """
        data = load_object(Path(load_path))

        self.node_data[node].data['climate_data'] = shorten_input_data(data['dataframe'], len(self.topology.timesteps))
        self.node_data[node].location.lon = data['longitude']
        self.node_data[node].location.lat = data['latitude']
        self.node_data[node].location.altitude = data['altitude']

    def read_climate_data_from_csv(self, node, load_path, lon, lat, alt=10):
        """
        Reads climate data from file

        Reads previously saved climate data (imported and saved with :func:`~read_climate_data_from_api`) from a file to \
        the respective node. This can save time, if api imports take too long

        :param str node: node as specified in the topology
        :param str load_path: path of csv data file. The csv needs to contain the following column headers:
                'ghi', 'dni', 'dhi', 'temp_air', 'rh', 'ws10'
        :param float lon: longitude of node
        :param float lat: latitude of node
        :param float alt: altitude of node
        :return: self at ``self.node_data[node]['climate_data']``
        """
        data = pd.read_csv(Path(load_path), index_col=0)

        # Create Datatime Index
        data.index = pd.to_datetime(data.index)

        # Calculate dni from ghi and dhi if not there
        if 'dni' not in data:
            data['dni'] = calculate_dni(data, lon, lat)

        self.node_data[node].data['climate_data'] = shorten_input_data(data,
                                                                          len(self.topology.timesteps))
        self.node_data[node].location.lon = lon
        self.node_data[node].location.lat = lat
        self.node_data[node].location.altitude = alt

    def read_hydro_natural_inflow(self, node:str, technology_name:str, hydro_natural_inflow:list):
        """
        Reads natural inflow for pumped hydro open cycle

        :param str node: node as specified in the topology
        :param str technology_name: technology that this inflow applies to
        :param list hydro_natural_inflow: hydro inflows in MWh
        :return: self at ``self.node_data[node]['climate_data'][technology_name + '_inflow']``
        """
        self.node_data[node].data['climate_data'][technology_name + '_inflow'] = shorten_input_data(hydro_natural_inflow,
                                                                                             len(self.topology.timesteps))

    def read_hydro_maximum_discharge(self, node:str, technology_name:str, maximum_discharge:list):
        """
        Reads maximum discharge of pumped hydro open cycles

        :param str node: node as specified in the topology
        :param str technology_name:  hydro technology for which maximum discharged is applied
        :param list maximum_discharge: max discharge in MWh for each timestep
        :return: self at ``self.node_data[node]['climate_data']['maximum_discharge']``
        """
        self.node_data[node].data['climate_data'][technology_name + '_maximum_discharge'] = shorten_input_data(maximum_discharge,
                                                                                             len(self.topology.timesteps))

    def read_demand_data(self, node:str, carrier:str, demand_data:list):
        """
        Reads demand data for one carrier to node.

        Note that demand for all carriers not specified is zero.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list demand_data: list of demand data. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[node]['demand'][carrier]``
        """
        self.node_data[node].data['demand'][carrier] = shorten_input_data(demand_data,
                                                                             len(self.topology.timesteps))

    def read_production_profile(self, node:str, carrier:str, production_data:list, curtailment:int):
        """
        Reads a production profile for one carrier to a node.

        If curtailment is 1, the production profile can be curtailed, if 0, then not.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list production_data: list of production data. Needs to have the same length as number of \
        time steps.
        :param int curtailment: 1 or 0, if 1 curtailment of production profile is allowed
        :return: self at ``self.node_data[node]['demand'][carrier]``
        """
        self.node_data[node].data['production_profile'][carrier] = shorten_input_data(production_data,
                                                                                         len(self.topology.timesteps))
        self.node_data[node].options.production_profile_curtailment[carrier] = curtailment

    def read_import_price_data(self, node:str, carrier:str, price_data:list):
        """
        Reads import price data of carrier to node

        Note that prices for all carriers not specified is zero.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list price_data: list of price data for respective carrier. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[node]['import_prices'][carrier]``
        """
        self.node_data[node].data['import_prices'][carrier] = shorten_input_data(price_data,
                                                                                    len(self.topology.timesteps))

    def read_export_price_data(self, node:str, carrier:str, price_data:list):
        """
        Reads export price data of carrier to node

        Note that prices for all carriers not specified is zero.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list price_data: list of price data for respective carrier. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[node]['export_prices'][carrier]``
        """
        self.node_data[node].data['export_prices'][carrier] = shorten_input_data(price_data,
                                                                                    len(self.topology.timesteps))

    def read_export_limit_data(self, node:str, carrier:str, export_limit_data:list):
        """
        Reads export limit data of carrier to node

        Note that limits for all carriers not specified is zero.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list export_limit_data: list of export limit data for respective carrier. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[node]['export_limit'][carrier]``
        """

        self.node_data[node].data['export_limit'][carrier] = export_limit_data

    def read_import_limit_data(self, node:str, carrier:str, import_limit_data:list):
        """
        Reads import limit data of carrier to node

        Note that limits for all carriers not specified is zero.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list import_limit_data: list of import limit data for respective carrier. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[node]['import_limit'][carrier]``
        """

        self.node_data[node].data['import_limit'][carrier] = shorten_input_data(import_limit_data,
                                                                                   len(self.topology.timesteps))

    def read_carbon_price_data(self, carbon_price_data:list, type:str):
        """
        Reads carbon price data. The price is the same for all nodes. Depending on the type, it can represent a carbon
        tax or a subsidy for negative emissions

        Note that carbon price is zero if not specified otherwise.

        :param list carbon_price_data: list of cost data for the carbon tax/subsidy. Needs to have the same length as number of \
        time steps.
        :param str type: 'tax' or 'subsidy' depending on what the carbon cost represents in the model
        :return: self at ``node_data[node]['carbon_tax'/'carbon_subsidy']``
        """
        if type == 'tax':
            self.global_data.data['carbon_prices']['tax'] = carbon_price_data

        elif type == 'subsidy':
            self.global_data.data['carbon_prices']['subsidy'] = carbon_price_data


    def read_export_emissionfactor_data(self, node:str, carrier:str, export_emissionfactor_data:list):
        """
        Reads export emission factor data of carrier to node

        Note that emission factors for all carriers not specified is zero.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list export_emissionfactor_data: list of emission data for respective carrier. \
        Needs to have the same length as number of time steps.
        :return: self at ``self.node_data[node]['export_emissionfactors'][carrier]``
        """

        self.node_data[node].data['export_emissionfactors'][carrier] = shorten_input_data(export_emissionfactor_data,
                                                                                             len(self.topology.timesteps))

    def read_import_emissionfactor_data(self, node:str, carrier:str, import_emissionfactor_data:list):
        """
        Reads import emission factor data of carrier to node

        Note that emission factors for all carriers not specified is zero.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list import_emissionfactor_data: list of emission data for respective carrier. \
        Needs to have the same length as number of time steps.
        :return: self at ``self.node_data[node]['import_emissionfactors'][carrier]``
        """

        self.node_data[node].data['import_emissionfactors'][carrier] = shorten_input_data(import_emissionfactor_data,
                                                                                             len(self.topology.timesteps))

    def read_technology_data(self, load_path='./data/Technology_Data/'):
        """
        Writes new and existing technologies to self and fits performance functions

        For the default settings, it reads in technology data from JSON files located at ``./data/Technology_Data`` for \
        all technologies specified in the topology. When technology data is stored at a different location, the path \
        should be specified as a string.

        :param str path: path to read technology data from
        :return: self at ``self.Technology_Data[node][tec]``
        """
        load_path = Path(load_path)
        self.model_information.tec_data_path = load_path

        for node in self.topology.nodes:
            self.technology_data[node] = {}
            # New technologies
            for technology in self.topology.technologies_new[node]:
                tec_data = open_json(technology, load_path)
                tec_data['name'] = technology

                self.technology_data[node][technology] = select_technology(tec_data)
                self.technology_data[node][technology].fit_technology_performance(self.node_data[node])

            # Existing technologies
            for technology in self.topology.technologies_existing[node].keys():
                tec_data = open_json(technology, load_path)
                tec_data['name'] = technology

                self.technology_data[node][technology + '_existing'] = select_technology(tec_data)
                self.technology_data[node][technology + '_existing'].existing = 1
                self.technology_data[node][technology + '_existing'].size_initial = \
                    self.topology.technologies_existing[node][technology]
                self.technology_data[node][technology + '_existing'].fit_technology_performance(self.node_data[node])

    def read_single_technology_data(self, node, technologies):
        """
        Reads technologies to DataHandle after it has been initialized.

        :param str node: node name as specified in the topology
        :param list technologies: technologies to add to node
        :param str path: path to read technology data from
        This function is only required if technologies are added to the model after the DataHandle has been initialized.
        """
        load_path = self.model_information.tec_data_path

        for technology in technologies:
            tec_data = open_json(technology, load_path)
            tec_data['name'] = technology

            self.technology_data[node][technology] = select_technology(tec_data)
            self.technology_data[node][technology].fit_technology_performance(self.node_data[node])

    def read_network_data(self, load_path:str='./data/network_data/'):
        """
        Writes new and existing network to self and calculates energy consumption

        Reads in network data from JSON files located at ``./data/network_data`` for all technologies specified in \
        the topology.

        :param str path: path to read network data from
        :return: self at ``self.Technology_Data[node][tec]``
        """
        load_path = Path(load_path)
        self.model_information.netw_data_path = load_path

        # New Networks
        for netw in self.topology.networks_new:
            netw_data = open_json(netw, load_path)
            netw_data['name'] = netw

            self.network_data[netw] = Network(netw_data)
            self.network_data[netw].connection = self.topology.networks_new[netw]['connection']
            self.network_data[netw].distance = self.topology.networks_new[netw]['distance']
            self.network_data[netw].size_max_arcs = self.topology.networks_new[netw]['size_max_arcs']
            self.network_data[netw].calculate_max_size_arc()

        # Existing Networks
        for netw in self.topology.networks_existing:
            netw_data = open_json(netw, load_path)
            netw_data['name'] = netw

            self.network_data[netw + '_existing'] = Network(netw_data)
            self.network_data[netw + '_existing'].existing = 1
            self.network_data[netw + '_existing'].connection = self.topology.networks_existing[netw]['connection']
            self.network_data[netw + '_existing'].distance = self.topology.networks_existing[netw]['distance']
            self.network_data[netw + '_existing'].size_initial = self.topology.networks_existing[netw]['size']
            self.network_data[netw + '_existing'].calculate_max_size_arc()

    def pprint(self):
        """
        Prints a summary of the input data (excluding climate data)

        :return: None
        """

        print('----- SET OF CARRIERS -----')
        for car in self.topology.carriers:
            print('- ' + car)
        print('----- NODE DATA -----')
        for node in self.node_data:
            print('\t -----------------------------------------------------')
            print('\t Nodename: ' + node)
            print('\t\tNew technologies:')
            for tec in self.topology.technologies_new[node]:
                print('\t\t - ' + tec)
            print('\t\tExisting technologies:')
            for tec in self.topology.technologies_existing[node]:
                print('\t\t - ' + tec)
            print('\t\tOther Node data:')
            for var in self.node_data[node].data:
                print('\t\t\tAverage of ' + var + ':')
                for ser in self.node_data[node].data[var]:
                    avg = round(self.node_data[node].data[var][ser].mean(), 2)
                    print('\t\t\t - ' + ser + ': ' + str(avg))
        print('----- NETWORK DATA -----')
        for netw in self.topology.networks_new:
            print('\t -----------------------------------------------------')
            print('\t' + netw)
            connection = self.topology.networks_new[netw]['connection']
            for from_node in connection:
                for to_node in connection[from_node].index:
                    if connection.at[from_node, to_node] == 1:
                        print('\t\t\t' + from_node + ' - ' + to_node)
        for netw in self.topology.networks_existing:
            print('\t -----------------------------------------------------')
            print('\t' + netw)
            connection = self.topology.networks_existing[netw]['connection']
            for from_node in connection:
                for to_node in connection[from_node].index:
                    if connection.at[from_node, to_node] == 1:
                        print('\t\t\t' + from_node + ' - ' + to_node)

    def save(self, save_path):
        """
        Saves instance of DataHandle to path.
        The instance can later be loaded

        :param str save_path: path to save to
        :return: None
        """
        save_object(self, Path(save_path))


class ClusteredDataHandle(DataHandle):
    """
    Performs the clustering process

    This function performs the k-means algorithm on the data resulting in a new DataHandle object that can be
    passed to the energhub class for optimization.

    :param DataHandle data: DataHandle containing data of the full resolution
    :param int nr_clusters: nr of clusters (tyical days) the data contains after the algorithm
    :param int nr_time_intervals_per_day: nr of time intervals per day in data (full resolution)
    """

    def __init__(self, data:DataHandle, nr_clusters:int, nr_time_intervals_per_day:int=24):
        """
        Constructor

        :param DataHandle data: DataHandle containing data of the full resolution
        :param int nr_clusters: nr of clusters (tyical days) the data contains after the algorithm
        :param int nr_time_intervals_per_day: nr of time intervals per day in data (full resolution)
        """
        # Copy over data from old object
        self.topology = data.topology
        self.node_data = data.node_data
        self.technology_data = {}
        self.network_data = data.network_data
        self.global_data = data.global_data
        self.model_information = data.model_information

        # k-means specs
        self.k_means_specs = simplification_specs(data.topology.timesteps)

        # perform clustering
        nr_days_full_resolution = (max(data.topology.timesteps) - min(data.topology.timesteps)).days + 1
        self.__cluster_data(nr_clusters, nr_days_full_resolution, nr_time_intervals_per_day)

    def __cluster_data(self, nr_clusters:int, nr_days_full_resolution:int, nr_time_intervals_per_day:int):
        """
        Performs the clustering process

        This function performs the k-means algorithm on the data resulting in a new DataHandle object that can be passed
        to the energhub class for optimization.

        :param DataHandle data: DataHandle containing data of the full resolution
        :param int nr_clusters: nr of clusters (tyical days) the data contains after the algorithm
        :param int nr_days_full_resolution: nr of days in data (full resolution)
        :param int nr_time_intervals_per_day: nr of time intervalls per day in data (full resolution)
        :return: instance of :class:`~ClusteredDataHandle`
        """
        # adjust timesteps
        self.topology.timesteps_clustered = range(0, nr_clusters * nr_time_intervals_per_day)
        # compile full matrix to cluster
        full_resolution = self.__compile_full_resolution_matrix(nr_time_intervals_per_day)
        # Perform clustering
        clustered_data, day_labels = perform_k_means(full_resolution,
                                                        nr_clusters)
        # Get order of typical days
        self.k_means_specs.full_resolution['sequence'] = compile_sequence(day_labels,
                                                                             nr_clusters,
                                                                             nr_days_full_resolution,
                                                                             nr_time_intervals_per_day)
        # Match typical day to actual day
        self.k_means_specs.full_resolution['typical_day'] = np.repeat(day_labels, nr_time_intervals_per_day)
        # Create factors, indicating how many times an hour occurs
        self.k_means_specs.reduced_resolution = get_day_factors(self.k_means_specs.full_resolution['sequence'])
        # Read data back in
        self.__read_clustered_data(clustered_data)

        self.read_technology_data()

    def __read_clustered_data(self, clustered_data:pd.DataFrame):
        """
        Reads clustered data back to self

        :param pd.DataFrame clustered_data: Clustered data
        """
        node_data = self.node_data
        for node in node_data:
            for series1 in node_data[node].data:
                self.node_data[node].data_clustered[series1] = pd.DataFrame(index=self.topology.timesteps_clustered)
                for series2 in node_data[node].data[series1]:
                    self.node_data[node].data_clustered[series1][series2] = \
                        reshape_df(clustered_data[node][series1][series2], None, 1)

        carbon_prices = self.global_data.data['carbon_prices']
        self.global_data.data_clustered['carbon_prices'] = pd.DataFrame(
            index=self.topology.timesteps_clustered)
        for series3 in carbon_prices:
            self.global_data.data_clustered['carbon_prices'][series3] = reshape_df(clustered_data['global_data']['carbon_prices'][series3], None, 1)

    def __compile_full_resolution_matrix(self, nr_time_intervals_per_day:int):
        """
        Compiles full resolution matrix to be clustered

        Contains, prices, emission factors, capacity factors,...
        """
        full_resolution = pd.DataFrame()
        node_data = self.node_data
        for node in node_data:
            for series1 in node_data[node].data:
                for series2 in node_data[node].data[series1]:
                    series_names = define_multiindex([
                        [node] * nr_time_intervals_per_day,
                        [series1] * nr_time_intervals_per_day,
                        [series2] * nr_time_intervals_per_day,
                        list(range(1, nr_time_intervals_per_day + 1))
                    ])
                    to_add = reshape_df(node_data[node].data[series1][series2],
                                           series_names, nr_time_intervals_per_day)
                    full_resolution = pd.concat([full_resolution, to_add], axis=1)

        carbon_prices = self.global_data.data['carbon_prices']
        for series3 in carbon_prices:
            series_names = define_multiindex([
                ['global_data'] * nr_time_intervals_per_day,
                ['carbon_prices'] * nr_time_intervals_per_day,
                [series3] * nr_time_intervals_per_day,
                list(range(1, nr_time_intervals_per_day + 1))
            ])
            to_add = reshape_df(carbon_prices[series3],
                                   series_names, nr_time_intervals_per_day)
            full_resolution = pd.concat([full_resolution, to_add], axis=1)
        return full_resolution


class DataHandle_AveragedData(DataHandle):
    """
    DataHandle sub-class for handling averaged data

    This class is used to generate time series of averaged data based on a full resolution
    or clustered input data.
    """

    def __init__(self, data:DataHandle, nr_timesteps_averaged:int):
        """
        Constructor
        """
        # Copy over data from old object
        self.topology = copy.deepcopy(data.topology)
        self.node_data = {}
        self.technology_data = {}
        self.network_data = data.network_data
        self.global_data = data.global_data
        self.model_information = data.model_information


        if hasattr(data, 'k_means_specs'):
            self.k_means_specs = data.k_means_specs

        # averaging specs
        self.averaged_specs = simplification_specs(data.topology.timesteps)

        # perform averaging for all nodal and global data
        self.__average_data(data, nr_timesteps_averaged)

        # read technology data
        self.__read_technology_data(data, nr_timesteps_averaged)

        # Write averaged specs
        self.averaged_specs.reduced_resolution = pd.DataFrame(
            data=np.ones(len(self.topology.timesteps)) * nr_timesteps_averaged,
            index=self.topology.timesteps,
            columns=['factor'])

        self.model_information.averaged_data_specs.nr_timesteps_averaged = nr_timesteps_averaged

    def __average_data(self, data_full_resolution:DataHandle, nr_timesteps_averaged:int):
        """
        Averages all nodal and global data

        :param data_full_resolution: Data full resolution
        :param nr_timesteps_averaged: How many time-steps should be averaged?
        """

        node_data = data_full_resolution.node_data
        global_data = data_full_resolution.global_data


        # Average data for full resolution
        # adjust timesteps
        end_interval = max(self.topology.timesteps)
        start_interval = min(self.topology.timesteps)
        time_resolution = str(nr_timesteps_averaged) + 'h'
        self.topology.timestep_length_h = nr_timesteps_averaged
        self.topology.timesteps = pd.date_range(start=start_interval, end=end_interval, freq=time_resolution)

        for node in node_data:
            self.node_data[node] = NodeData(self.topology)
            self.node_data[node].options = node_data[node].options
            self.node_data[node].location = node_data[node].location
            for series1 in node_data[node].data:
                self.node_data[node].data[series1] = pd.DataFrame(index=self.topology.timesteps)
                for series2 in node_data[node].data[series1]:
                    self.node_data[node].data[series1][series2] = \
                        average_series(node_data[node].data[series1][series2], nr_timesteps_averaged)

        self.global_data = GlobalData(self.topology)
        for series1 in global_data.data:
            self.global_data.data[series1] = pd.DataFrame(index=self.topology.timesteps)
            for series2 in global_data.data[series1]:
                self.global_data.data[series1][series2] = \
                    average_series(global_data.data[series1][series2], nr_timesteps_averaged)

        # Average data for clustered resolution
        if self.model_information.clustered_data == 1:
            # adjust timesteps
            end_interval = max(self.topology.timesteps_clustered)
            start_interval = min(self.topology.timesteps_clustered)
            self.topology.timesteps_clustered = range(start_interval, int((end_interval + 1) / nr_timesteps_averaged))


            for node in node_data:
                for series1 in node_data[node].data:
                    self.node_data[node].data_clustered[series1] = pd.DataFrame(self.topology.timesteps_clustered)
                    for series2 in node_data[node].data[series1]:
                        self.node_data[node].data_clustered[series1][series2] = \
                            average_series(node_data[node].data_clustered[series1][series2], nr_timesteps_averaged)


            for series1 in global_data.data:
                self.global_data.data_clustered[series1] = pd.DataFrame(self.topology.timesteps_clustered)
                for series2 in global_data.data[series1]:
                    self.global_data.data_clustered[series1][series2] = \
                        average_series(global_data.data_clustered[series1][series2], nr_timesteps_averaged)


    def __read_technology_data(self, data_full_resolution:DataHandle, nr_timesteps_averaged:int):
        """
        Reads technology data for time-averaging algorithm

        :param data_full_resolution: Data full resolution
        :param nr_timesteps_averaged: How many time-steps should be averaged?
        """
        load_path = self.model_information.tec_data_path
        for node in self.topology.nodes:
            self.technology_data[node] = {}
            # New technologies
            for technology in self.topology.technologies_new[node]:
                tec_data = open_json(technology, load_path)
                tec_data['name'] = technology
                self.technology_data[node][technology] = select_technology(tec_data)

                if self.technology_data[node][technology].technology_model == 'RES':
                    # Fit performance based on full resolution and average capacity factor
                    self.technology_data[node][technology].fit_technology_performance(
                        data_full_resolution.node_data[node])
                    cap_factor = self.technology_data[node][technology].fitted_performance.coefficients['capfactor']
                    new_cap_factor = average_series(cap_factor, nr_timesteps_averaged)
                    self.technology_data[node][technology].fitted_performance.coefficients['capfactor'] = \
                        new_cap_factor

                    lower_output_bound = np.zeros(shape=(len(new_cap_factor)))
                    upper_output_bound = new_cap_factor
                    output_bounds = np.column_stack((lower_output_bound, upper_output_bound))

                    self.technology_data[node][technology].fitted_performance.bounds['output']['electricity'] = \
                        output_bounds
                else:
                    # Fit performance based on averaged data
                    self.technology_data[node][technology].fit_technology_performance(self.node_data[node])

            # Existing technologies
            for technology in self.topology.technologies_existing[node].keys():
                tec_data = open_json(technology, load_path)
                tec_data['name'] = technology

                self.technology_data[node][technology + '_existing'] = select_technology(tec_data)
                self.technology_data[node][technology + '_existing'].existing = 1
                self.technology_data[node][technology + '_existing'].size_initial = \
                    self.topology.technologies_existing[node][technology]
                if self.technology_data[node][technology + '_existing'].technology_model == 'RES':
                    # Fit performance based on full resolution and average capacity factor
                    self.technology_data[node][technology + '_existing'].fit_technology_performance(
                        data_full_resolution.node_data[node])
                    cap_factor = self.technology_data[node][technology + '_existing'].fitted_performance.coefficients[
                        'capfactor']
                    new_cap_factor = average_series(cap_factor, nr_timesteps_averaged)

                    self.technology_data[node][technology + '_existing'].fitted_performance.coefficients['capfactor'] = \
                        new_cap_factor

                    lower_output_bound = np.zeros(shape=(len(new_cap_factor)))
                    upper_output_bound = new_cap_factor
                    output_bounds = np.column_stack((lower_output_bound, upper_output_bound))

                    self.technology_data[node][technology + '_existing'].fitted_performance.bounds['output'][
                        'electricity'] = \
                        output_bounds
                else:
                    # Fit performance based on averaged data
                    self.technology_data[node][technology + '_existing'].fit_technology_performance(
                        self.node_data[node])
