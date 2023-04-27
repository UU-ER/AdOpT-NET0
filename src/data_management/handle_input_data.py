import src.data_management as dm
import src.data_management.components as comp
import src.global_variables as global_variables

import pandas as pd
import copy
import numpy as np



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
        self.node_data = {}
        self.technology_data = {}
        self.network_data = {}

        self.topology = topology

        # Initialize Node data
        for node in self.topology.nodes:
            self.node_data[node] = dm.NodeData(topology)

    def read_climate_data_from_api(self, node, lon, lat, alt=10, dataset='JRC', year='typical_year', save_path=0):
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
            data = dm.import_jrc_climate_data(lon, lat, year, alt)
        else:
            raise Exception('Other APIs are not available')

        # Save
        if not save_path==0:
            dm.save_object(data, save_path)

        # Match with timesteps
        data['dataframe'] = data['dataframe'].loc[self.topology.timesteps]

        # Write to DataHandle
        self.node_data[node].data['climate_data'] = data['dataframe']
        self.node_data[node].location.lon = data['longitude']
        self.node_data[node].location.lat = data['latitude']
        self.node_data[node].location.altitude = data['altitude']

    def read_climate_data_from_file(self, node, file):
        """
        Reads climate data from file

        Reads previously saved climate data (imported and saved with :func:`~read_climate_data_from_api`) from a file to \
        the respective node. This can save time, if api imports take too long

        :param str node: node as specified in the topology
        :param str file: path of climate data file
        :return: self at ``self.node_data[node]['climate_data']``
        """
        data = dm.load_object(file)

        # Match with timesteps
        data['dataframe'] = data['dataframe'][0:len(self.topology.timesteps)]

        self.node_data[node].data['climate_data'] = data['dataframe']
        self.node_data[node].location.lon = data['longitude']
        self.node_data[node].location.lat = data['latitude']
        self.node_data[node].location.altitude = data['altitude']

    def read_climate_data_from_csv(self, node, file, lon, lat, alt=10):
        """
        Reads climate data from file

        Reads previously saved climate data (imported and saved with :func:`~read_climate_data_from_api`) from a file to \
        the respective node. This can save time, if api imports take too long

        :param str node: node as specified in the topology
        :param str file: path of csv data file. The csv needs to contain the following column headers:
                'ghi', 'dni', 'dhi', 'temp_air', 'rh', 'ws10'
        :param float lon: longitude of node
        :param float lat: latitude of node
        :param float alt: altitude of node
        :return: self at ``self.node_data[node]['climate_data']``
        """
        data = pd.read_csv(file, index_col=0)

        # Create Datatime Index
        data.index = pd.to_datetime(data.index)

        # Calculate dni from ghi and dhi if not there
        if 'dni' not in data:
            data['dni'] = dm.calculate_dni(data, lon, lat)

        # Match with timesteps
        data = data[0:len(self.topology.timesteps)]

        self.node_data[node].data['climate_data'] = data
        self.node_data[node].location.lon = lon
        self.node_data[node].location.lat = lat
        self.node_data[node].location.altitude = alt

    def read_demand_data(self, node, carrier, demand_data):
        """
        Reads demand data for one carrier to node.

        Note that demand for all carriers not specified is zero.
        
        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list demand_data: list of demand data. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[node]['demand'][carrier]``
        """

        self.node_data[node].data['demand'][carrier] = demand_data

    def read_production_profile(self, node, carrier, production_data, curtailment):
        """
        Reads a production profile for one carrier to a node.

        If curtailment is 1, the production profile can be curtailed, if 0, then not.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list demand_data: list of demand data. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[node]['demand'][carrier]``
        """
        self.node_data[node].data['production_profile'][carrier] = production_data
        self.node_data[node].options.production_profile_curtailment[carrier] = curtailment

    def read_import_price_data(self, node, carrier, price_data):
        """
        Reads import price data of carrier to node

        Note that prices for all carriers not specified is zero.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list price_data: list of price data for respective carrier. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[node]['import_prices'][carrier]``
        """

        self.node_data[node].data['import_prices'][carrier] = price_data

    def read_export_price_data(self, node, carrier, price_data):
        """
        Reads export price data of carrier to node

        Note that prices for all carriers not specified is zero.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list price_data: list of price data for respective carrier. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[node]['export_prices'][carrier]``
        """

        self.node_data[node].data['export_prices'][carrier] = price_data

    def read_export_limit_data(self, node, carrier, export_limit_data):
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

    def read_import_limit_data(self, node, carrier, import_limit_data):
        """
        Reads import limit data of carrier to node

        Note that limits for all carriers not specified is zero.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list import_limit_data: list of import limit data for respective carrier. Needs to have the same length as number of \
        time steps.
        :return: self at ``self.node_data[node]['import_limit'][carrier]``
        """

        self.node_data[node].data['import_limit'][carrier] = import_limit_data

    def read_export_emissionfactor_data(self, node, carrier, export_emissionfactor_data):
        """
        Reads export emission factor data of carrier to node

        Note that emission factors for all carriers not specified is zero.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list export_emissionfactor_data: list of emission data for respective carrier. \
        Needs to have the same length as number of time steps.
        :return: self at ``self.node_data[node]['export_emissionfactors'][carrier]``
        """

        self.node_data[node].data['export_emissionfactors'][carrier] = export_emissionfactor_data

    def read_import_emissionfactor_data(self, node, carrier, import_emissionfactor_data):
        """
        Reads import emission factor data of carrier to node

        Note that emission factors for all carriers not specified is zero.

        :param str node: node name as specified in the topology
        :param str carrier: carrier name as specified in the topology
        :param list import_emissionfactor_data: list of emission data for respective carrier. \
        Needs to have the same length as number of time steps.
        :return: self at ``self.node_data[node]['import_emissionfactors'][carrier]``
        """

        self.node_data[node].data['import_emissionfactors'][carrier] = import_emissionfactor_data

    def read_technology_data(self):
        """
        Writes new and existing technologies to self and fits performance functions

        Reads in technology data from JSON files located at ``./data/technology_data`` for all technologies specified in \
        the topology.

        :return: self at ``self.technology_data[node][tec]``
        """
        for node in self.topology.nodes:
            self.technology_data[node] = {}
            # New technologies
            for technology in self.topology.technologies_new[node]:
                self.technology_data[node][technology] = comp.Technology(technology)
                self.technology_data[node][technology].fit_technology_performance(self.node_data[node])
            # Existing technologies
            for technology in self.topology.technologies_existing[node].keys():
                self.technology_data[node][technology + '_existing'] = comp.Technology(technology)
                self.technology_data[node][technology + '_existing'].existing = 1
                self.technology_data[node][technology + '_existing'].size_initial = self.topology.technologies_existing[node][technology]
                self.technology_data[node][technology + '_existing'].fit_technology_performance(self.node_data[node])

    def read_single_technology_data(self, node, technologies):
        """
        Reads technologies to DataHandle after it has been initialized.

        This function is only required if technologies are added to the model after the DataHandle has been initialized.
        """

        for technology in technologies:
            self.technology_data[node][technology] = comp.Technology(technology)
            self.technology_data[node][technology].fit_technology_performance(self.node_data[node])

    def read_network_data(self):
        """
        Writes newand existing network to self and calculates energy consumption

        Reads in network data from JSON files located at ``./data/network_data`` for all technologies specified in \
        the topology.

        :return: self at ``self.technology_data[node][tec]``
        """

        # New Networks
        for network in self.topology.networks_new:
            self.network_data[network] = comp.Network(network)
            self.network_data[network].connection = self.topology.networks_new[network]['connection']
            self.network_data[network].distance = self.topology.networks_new[network]['distance']

        # Existing Networks
        for network in self.topology.networks_existing:
            self.network_data[network + '_existing'] = comp.Network(network)
            self.network_data[network + '_existing'].existing = 1
            self.network_data[network + '_existing'].connection = self.topology.networks_existing[network]['connection']
            self.network_data[network + '_existing'].distance = self.topology.networks_existing[network]['distance']
            self.network_data[network + '_existing'].size_initial = self.topology.networks_existing[network]['size']
            

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
            print('\t Nodename: '+ node)
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
            print('\t'+ netw)
            connection = self.topology.networks_new[netw]['connection']
            for from_node in connection:
                for to_node in connection[from_node].index:
                    if connection.at[from_node, to_node] == 1:
                        print('\t\t\t' + from_node  + ' - ' +  to_node)
        for netw in self.topology.networks_existing:
            print('\t -----------------------------------------------------')
            print('\t'+ netw)
            connection = self.topology.networks_existing[netw]['connection']
            for from_node in connection:
                for to_node in connection[from_node].index:
                    if connection.at[from_node, to_node] == 1:
                        print('\t\t\t' + from_node  + ' - ' +  to_node)


    def save(self, path):
        """
        Saves instance of DataHandle to path.

        The instance can later be loaded with

        :param str path: path to save to
        :return: None
        """
        dm.save_object(self, path)



class ClusteredDataHandle(DataHandle):
    """
    Performs the clustering process

    This function performs the k-means algorithm on the data resulting in a new DataHandle object that can be
    passed to the energhub class for optimization.

    :param DataHandle data: DataHandle containing data of the full resolution
    :param int nr_clusters: nr of clusters (tyical days) the data contains after the algorithm
    :param int nr_time_intervals_per_day: nr of time intervals per day in data (full resolution)
    """
    def __init__(self, data, nr_clusters, nr_time_intervals_per_day=24):
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

        # k-means specs
        self.k_means_specs = dm.simplification_specs(data.topology.timesteps)

        # perform clustering
        nr_days_full_resolution = (max(data.topology.timesteps) -  min(data.topology.timesteps)).days + 1
        self.__cluster_data(nr_clusters, nr_days_full_resolution, nr_time_intervals_per_day)

    def __cluster_data(self, nr_clusters, nr_days_full_resolution, nr_time_intervals_per_day):
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
        clustered_data, day_labels = dm.perform_k_means(full_resolution,
                                                        nr_clusters)
        # Get order of typical days
        self.k_means_specs.full_resolution['sequence'] = dm.compile_sequence(day_labels,
                                         nr_clusters,
                                         nr_days_full_resolution,
                                         nr_time_intervals_per_day)
        # Match typical day to actual day
        self.k_means_specs.full_resolution['typical_day'] = np.repeat(day_labels, nr_time_intervals_per_day)
        # Create factors, indicating how many times an hour occurs
        self.k_means_specs.reduced_resolution = dm.get_day_factors(self.k_means_specs.full_resolution['sequence'])
        # Read data back in
        self.__read_clustered_data(clustered_data)

        self.read_technology_data()

    def __read_clustered_data(self, clustered_data):
        """
        Reads clustered data back to self

        :param clustered_data: Clustered data
        """
        node_data = self.node_data
        for node in node_data:
            for series1 in node_data[node].data:
                self.node_data[node].data_clustered[series1] = pd.DataFrame(index=self.topology.timesteps_clustered)
                for series2 in node_data[node].data[series1]:
                    self.node_data[node].data_clustered[series1][series2] = \
                        dm.reshape_df(clustered_data[node][series1][series2], None, 1)

    def __compile_full_resolution_matrix(self, nr_time_intervals_per_day):
        """
        Compiles full resolution matrix to be clustered

        Contains, prices, emission factors, capacity factors,...
        """
        full_resolution = pd.DataFrame()
        node_data = self.node_data
        for node in node_data:
            for series1 in node_data[node].data:
                    for series2 in node_data[node].data[series1]:
                        series_names = dm.define_multiindex([
                            [node] * nr_time_intervals_per_day,
                            [series1] * nr_time_intervals_per_day,
                            [series2] * nr_time_intervals_per_day,
                            list(range(1, nr_time_intervals_per_day + 1))
                        ])
                        to_add = dm.reshape_df(node_data[node].data[series1][series2],
                                            series_names, nr_time_intervals_per_day)
                        full_resolution = pd.concat([full_resolution, to_add], axis=1)
        return full_resolution


class DataHandle_AveragedData(DataHandle):
    """
    DataHandle sub-class for handling averaged data

    This class is used to generate time series of averaged data based on a full resolution
    or clustered input data.
    """
    def __init__(self, data, nr_timesteps_averaged):
        """
        Constructor
        """
        # Copy over data from old object
        self.topology = copy.deepcopy(data.topology)
        self.node_data = {}
        self.technology_data = {}
        self.network_data = data.network_data

        if hasattr(data, 'k_means_specs'):
            self.k_means_specs = data.k_means_specs

        # averaging specs
        self.averaged_specs = dm.simplification_specs(data.topology.timesteps)

        # perform averaging for all nodal data
        self.__average_node_data(data, nr_timesteps_averaged)

        # read technology data
        self.__read_technology_data(data, nr_timesteps_averaged)

        # Write averaged specs
        self.averaged_specs.reduced_resolution = pd.DataFrame(
            data=np.ones(len(self.topology.timesteps)) * nr_timesteps_averaged,
            index=self.topology.timesteps,
            columns=['factor'])

        global_variables.averaged_data_specs.nr_timesteps_averaged = nr_timesteps_averaged


    def __average_node_data(self, data_full_resolution, nr_timesteps_averaged):
        """
        Averages all nodal data
        
        :param data_full_resolution: Data full resolution
        :param nr_timesteps_averaged: How many time-steps should be averaged?
        """

        node_data = data_full_resolution.node_data

        # Average data for full resolution
        # adjust timesteps
        end_interval = max(self.topology.timesteps)
        start_interval = min(self.topology.timesteps)
        time_resolution = str(nr_timesteps_averaged) + 'h'
        self.topology.timestep_length_h = nr_timesteps_averaged
        self.topology.timesteps = pd.date_range(start=start_interval, end=end_interval, freq=time_resolution)

        for node in node_data:
            self.node_data[node] = dm.NodeData(self.topology)
            self.node_data[node].options = node_data[node].options
            self.node_data[node].location = node_data[node].location
            for series1 in node_data[node].data:
                self.node_data[node].data[series1] = pd.DataFrame(index=self.topology.timesteps)
                for series2 in node_data[node].data[series1]:
                    self.node_data[node].data[series1][series2] = \
                        dm.average_series(node_data[node].data[series1][series2], nr_timesteps_averaged)

        # Average data for clustered resolution
        if global_variables.clustered_data == 1:
            # adjust timesteps
            end_interval = max(self.topology.timesteps_clustered)
            start_interval = min(self.topology.timesteps_clustered)
            self.topology.timesteps_clustered = range(start_interval, int((end_interval+1) / nr_timesteps_averaged))


            for node in node_data:
                for series1 in node_data[node].data:
                    self.node_data[node].data_clustered[series1] = pd.DataFrame(self.topology.timesteps_clustered)
                    for series2 in node_data[node].data[series1]:
                        self.node_data[node].data_clustered[series1][series2] = \
                            dm.average_series(node_data[node].data_clustered[series1][series2], nr_timesteps_averaged)


    def __read_technology_data(self, data_full_resolution, nr_timesteps_averaged):
        """
        Reads technology data for time-averaging algorithm

        :param data_full_resolution: Data full resolution
        :param nr_timesteps_averaged: How many time-steps should be averaged?
        """
        for node in self.topology.nodes:
            self.technology_data[node] = {}
            # New technologies
            for technology in self.topology.technologies_new[node]:
                self.technology_data[node][technology] = comp.Technology(technology)
                if self.technology_data[node][technology].technology_model == 'RES':
                    # Fit performance based on full resolution and average capacity factor
                    self.technology_data[node][technology].fit_technology_performance(data_full_resolution.node_data[node])
                    cap_factor = self.technology_data[node][technology].fitted_performance.coefficients['capfactor']
                    new_cap_factor = dm.average_series(cap_factor, nr_timesteps_averaged)
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
                self.technology_data[node][technology + '_existing'] = comp.Technology(technology)
                self.technology_data[node][technology + '_existing'].existing = 1
                self.technology_data[node][technology + '_existing'].size_initial = self.topology.technologies_existing[node][technology]
                if self.technology_data[node][technology].technology_model == 'RES':
                    # Fit performance based on full resolution and average capacity factor
                    self.technology_data[node][technology + '_existing'].fit_technology_performance(data_full_resolution.node_data[node])
                    cap_factor = self.technology_data[node][technology + '_existing'].fitted_performance.coefficients['capfactor']
                    new_cap_factor = dm.average_series(cap_factor, nr_timesteps_averaged)

                    self.technology_data[node][technology + '_existing'].fitted_performance.coefficients['capfactor'] = \
                        new_cap_factor

                    lower_output_bound = np.zeros(shape=(len(new_cap_factor)))
                    upper_output_bound = new_cap_factor
                    output_bounds = np.column_stack((lower_output_bound, upper_output_bound))

                    self.technology_data[node][technology + '_existing'].fitted_performance.bounds['output']['electricity'] = \
                        output_bounds
                else:
                    # Fit performance based on averaged data
                    self.technology_data[node][technology + '_existing'].fit_technology_performance(self.node_data[node])