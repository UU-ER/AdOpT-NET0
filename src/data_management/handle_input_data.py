import src.data_management as dm
import src.data_management.components as comp

import pandas as pd
from sklearn.cluster import KMeans
import copy
import numpy as np



class DataHandle:
    """
    Data Handle for loading and performing operations on input data.

    The Data Handle class allows data import and modifications of input data to an instance of the energyhub class.
    The constructor of the class takes an instance of the class
    :func:`~src.data_management.handle_topology.SystemTopology` as an input.
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
        # Initialize demand, prices, emission factors = 0 for all timesteps, carriers and nodes
        variables = ['demand',
                     'import_prices',
                     'import_limit',
                     'import_emissionfactors',
                     'export_prices',
                     'export_limit',
                     'export_emissionfactors']

        for node in self.topology.nodes:
            self.node_data[node] = {}
            for var in variables:
                self.node_data[node][var] = pd.DataFrame(index=self.topology.timesteps)

            for carrier in self.topology.carriers:
                for var in variables:
                    self.node_data[node][var][carrier] = 0

            self.node_data[node]['production_profile'] = {}
            for carrier in self.topology.carriers:
                self.node_data[node]['production_profile'][carrier] = {}
                self.node_data[node]['production_profile'][carrier]['production_data'] \
                    = pd.DataFrame(index=self.topology.timesteps)
                self.node_data[node]['production_profile'][carrier]['production_data'] = 0
                self.node_data[node]['production_profile'][carrier]['curtailment'] = 0

    def read_climate_data_from_api(self, node, lon, lat, alt=10, dataset='JRC', year='typical_year', save_path=0):
        """
        Reads in climate data for a full year

        Reads in climate data for a full year from the specified source \
        (`JRC PVGIS <https://re.jrc.ec.europa.eu/pvg_tools/en/>`_ or \
        `ERA5 <https://cds.climate.copernicus.eu/cdsapp#!/home>`_). For access to the ERA5 api, \
        an api key is required. Refer to `<https://cds.climate.copernicus.eu/api-how-to>`_

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
        elif dataset == 'ERA5':
            data = dm.import_era5_climate_data(lon, lat, year)

        # Match with timesteps
        data['dataframe'] = data['dataframe'].loc[self.topology.timesteps]

        # Save
        if not save_path==0:
            dm.save_object(data, save_path)

        self.node_data[node]['climate_data'] = data


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
        self.node_data[node]['climate_data'] = data

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

        self.node_data[node]['demand'][carrier] = demand_data

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
        data = {}
        data['production_data'] = production_data
        data['curtailment'] = curtailment
        self.node_data[node]['production_profile'][carrier] = data

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

        self.node_data[node]['import_prices'][carrier] = price_data

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

        self.node_data[node]['export_prices'][carrier] = price_data

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

        self.node_data[node]['export_limit'][carrier] = export_limit_data

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

        self.node_data[node]['import_limit'][carrier] = import_limit_data

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

        self.node_data[node]['export_emissionfactors'][carrier] = export_emissionfactor_data

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

        self.node_data[node]['import_emissionfactors'][carrier] = import_emissionfactor_data

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
                self.technology_data[node][technology].fit_technology_performance(self.node_data[node]['climate_data'])
            # Existing technologies
            for technology in self.topology.technologies_existing[node].keys():
                self.technology_data[node][technology + '_existing'] = comp.Technology(technology)
                self.technology_data[node][technology + '_existing'].existing = 1
                self.technology_data[node][technology + '_existing'].size_initial = self.topology.technologies_existing[node][technology]
                self.technology_data[node][technology + '_existing'].fit_technology_performance(self.node_data[node]['climate_data'])

    def read_single_technology_data(self, node, technologies):
        """
        Reads technologies to DataHandle after it has been initialized.

        This function is only required if technologies are added to the model after the DataHandle has been initialized.
        """

        for technology in technologies:
            self.technology_data[node][technology] = comp.Technology(technology)
            self.technology_data[node][technology].fit_technology_performance(self.node_data[node]['climate_data'])

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
        for node in self.topology.nodes:
            print('----- NODE '+ node +' -----')
            for inst in self.node_data[node]:
                if not inst == 'climate_data':
                    print('\t ' + inst)
                    print('\t\t' + f"{'':<15}{'Mean':>10}{'Min':>10}{'Max':>10}")
                    for carrier in self.topology.carriers:
                        print('\t\t' + f"{carrier:<15}"
                                       f"{str(round(self.node_data[node][inst][carrier].mean(), 2)):>10}"
                                       f"{str(round(self.node_data[node][inst][carrier].min(), 2)):>10}"
                                       f"{str(round(self.node_data[node][inst][carrier].max(), 2)):>10}")

    def save(self, path):
        """
        Saves instance of DataHandle to path.

        The instance can later be loaded with

        :param str path: path to save to
        :return: None
        """
        dm.save_object(self, path)

    def flag_tecs_for_clustering(self):
        """
        Creates a dictonary with flags for RES technologies

        These technologies contain time-dependent input data, i.e. capacity factors.
        :return dict tecs_flagged_for_clustering: flags for technologies and nodes

        """
        tecs_flagged_for_clustering = {}
        for node in self.topology.nodes:
            tecs_flagged_for_clustering[node] = {}
            for technology in self.technology_data[node]:
                if self.technology_data[node][technology].technology_model == 'RES':
                    tecs_flagged_for_clustering[node][technology] = 'capacity_factor'
                elif self.technology_data[node][technology].technology_model == 'STOR':
                    tecs_flagged_for_clustering[node][technology] = 'ambient_loss_factor'
        return tecs_flagged_for_clustering


class ClusteredDataHandle(DataHandle):
    """
    Performs the clustering process

    This function performs the k-means algorithm on the data resulting in a new DataHandle object that can be
    passed to the energhub class for optimization.

    :param DataHandle data_in: DataHandle containing data of the full resolution
    :param int nr_clusters: nr of clusters (tyical days) the data contains after the algorithm
    :param int nr_time_intervals_per_day: nr of time intervals per day in data (full resolution)
    """
    def __init__(self, data_in, nr_clusters, nr_time_intervals_per_day=24):
        """
        Constructor

        :param DataHandle data_in: DataHandle containing data of the full resolution
        :param int nr_clusters: nr of clusters (tyical days) the data contains after the algorithm
        :param int nr_time_intervals_per_day: nr of time intervals per day in data (full resolution)
        """
        data = copy.deepcopy(data_in)

        # Copy over data from old object
        self.node_data = {}
        self.node_data_full_resolution = data.node_data
        self.technology_data = data.technology_data
        self.network_data = data.network_data
        self.topology = data.topology

        # k-means specs
        self.k_means_specs = dm.simplification_specs(data.topology.timesteps)

        # perform clustering
        nr_days_full_resolution = (max(data.topology.timesteps) -  min(data.topology.timesteps)).days + 1
        self.cluster_data(nr_clusters, nr_days_full_resolution, nr_time_intervals_per_day)

    def cluster_data(self, nr_clusters, nr_days_full_resolution, nr_time_intervals_per_day):
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
        self.topology.timesteps = range(0, nr_clusters * nr_time_intervals_per_day)
        # flag tecs that contain time-dependent data
        tecs_flagged_for_clustering = self.flag_tecs_for_clustering()
        # compile full matrix to cluster
        full_resolution = self.compile_full_resolution_matrix(nr_time_intervals_per_day,
                                                              tecs_flagged_for_clustering)
        # Perform clustering
        clustered_data, day_labels = dm.perform_k_means(full_resolution,
                                                        nr_clusters)
        # Get order of typical days
        self.k_means_specs.full_resolution['hourly_order'] = dm.compile_hourly_order(day_labels,
                                         nr_clusters,
                                         nr_days_full_resolution,
                                         nr_time_intervals_per_day)
        # Match typical day to actual day
        self.k_means_specs.full_resolution['typical_day'] = np.repeat(day_labels, nr_time_intervals_per_day)
        # Create factors, indicating how many times an hour occurs
        self.k_means_specs.reduced_resolution = dm.get_day_factors(self.k_means_specs.full_resolution['hourly_order'])
        # Read data back in
        self.read_clustered_data(clustered_data, tecs_flagged_for_clustering)

    def read_clustered_data(self, clustered_data, tecs_flagged_for_clustering):
        """
        Reads clustered data back to self

        :param clustered_data: Clustered data
        :param tecs_flagged_for_clustering: technologies that have time-dependent data
        """
        node_data = self.node_data_full_resolution
        for node in node_data:
            self.node_data[node] = {}
            for series in node_data[node]:
                if not series == 'climate_data':
                    self.node_data[node][series] = pd.DataFrame()
                    for carrier in node_data[node][series]:
                        self.node_data[node][series][carrier] = \
                            reshape_df(clustered_data[node][series][carrier],
                                       None, 1)
            for tec in tecs_flagged_for_clustering[node]:
                series_data = reshape_df(clustered_data[node][tec][tecs_flagged_for_clustering[node][tec]], None, 1)
                series_data = series_data.to_numpy()
                self.technology_data[node][tec].fitted_performance[tecs_flagged_for_clustering[node][tec]] = \
                    series_data

    def compile_full_resolution_matrix(self, nr_time_intervals_per_day, tecs_flagged_for_clustering):
        """
        Compiles full resolution matrix to be clustered

        Contains, prices, emission factors, capacity factors,...
        """
        full_resolution = pd.DataFrame()
        node_data = self.node_data_full_resolution
        for node in node_data:
            for series in node_data[node]:
                if not series == 'climate_data':
                    for carrier in node_data[node][series]:
                        series_names = define_multiindex([
                            [node] * nr_time_intervals_per_day,
                            [series] * nr_time_intervals_per_day,
                            [carrier] * nr_time_intervals_per_day,
                            list(range(1, nr_time_intervals_per_day + 1))
                        ])
                        to_add = reshape_df(node_data[node][series][carrier],
                                            series_names, nr_time_intervals_per_day)
                        full_resolution = pd.concat([full_resolution, to_add], axis=1)
            for tec in tecs_flagged_for_clustering[node]:
                series_names = define_multiindex([
                    [node] * nr_time_intervals_per_day,
                    [tec] * nr_time_intervals_per_day,
                    [tecs_flagged_for_clustering[node][tec]] * nr_time_intervals_per_day,
                    list(range(1, nr_time_intervals_per_day + 1))
                ])
                to_add = reshape_df(self.technology_data[node][tec].fitted_performance[tecs_flagged_for_clustering[node][tec]],
                                    series_names, nr_time_intervals_per_day)
                full_resolution = pd.concat([full_resolution, to_add], axis=1)
        return full_resolution


class DataHandle_AveragedData(DataHandle):
    """
    DataHandle sub-class for handling averaged data

    This class is used to generate time series of averaged data based on a full resolution
    or clustered input data.
    """
    def __init__(self, data_in, nr_timesteps_averaged):
        """
        Constructor
        """
        data = copy.deepcopy(data_in)
        # Copy over data from old object
        self.node_data_full_resolution = data.node_data
        self.node_data = {}
        self.technology_data = data.technology_data
        self.network_data = data.network_data
        self.topology = data.topology

        if hasattr(data, 'k_means_specs'):
            self.k_means_specs = data.k_means_specs

        # averaging specs
        self.averaged_specs = dm.simplification_specs(data.topology.timesteps)

        # perform averaging
        self.average_data(nr_timesteps_averaged)

    def average_data(self, nr_timesteps_averaged):
        # adjust timesteps
        end_interval = max(self.topology.timesteps)
        start_interval = min(self.topology.timesteps)
        time_resolution = str(nr_timesteps_averaged) + 'h'
        self.topology.timestep_length_h = nr_timesteps_averaged
        self.topology.timesteps = pd.date_range(start=start_interval, end=end_interval, freq=time_resolution)
        # flag tecs that contain time-dependent data
        tecs_flagged_for_clustering = self.flag_tecs_for_clustering()
        # Average all time-dependent data and write to self
        self.perform_averaging(nr_timesteps_averaged, tecs_flagged_for_clustering)
        # Write averaged specs
        self.averaged_specs.reduced_resolution = pd.DataFrame(
            data=np.ones(len(self.topology.timesteps)) * nr_timesteps_averaged,
            index=self.topology.timesteps,
            columns=['factor'])

    def perform_averaging(self, nr_timesteps_averaged, tecs_flagged_for_clustering):
        """
        Average all time-dependent data

        :param nr_timesteps_averaged: How many time-steps should be averaged?
        :param tecs_flagged_for_clustering: technologies that have time-dependent data
        """
        node_data = self.node_data_full_resolution
        for node in node_data:
            self.node_data[node] = {}
            for series in node_data[node]:
                self.node_data[node][series] = pd.DataFrame()
                if not series == 'climate_data':
                    for carrier in node_data[node][series]:
                        series_data = dm.reshape_df(node_data[node][series][carrier],
                                                    None, nr_timesteps_averaged)
                        self.node_data[node][series][carrier] = series_data.mean(axis=1)
            for tec in tecs_flagged_for_clustering[node]:
                series_data = dm.reshape_df(
                    self.technology_data[node][tec].fitted_performance[tecs_flagged_for_clustering[node][tec]],
                    None, nr_timesteps_averaged)
                self.technology_data[node][tec].fitted_performance[
                    tecs_flagged_for_clustering[node][tec]] = series_data.mean(axis=1)


def reshape_df(series_to_add, column_names, nr_cols):
    """
    Transform all data to large dataframe with each row being one day
    """
    if not type(series_to_add).__module__ == np.__name__:
        transformed_series = series_to_add.to_numpy()
    else:
        transformed_series = series_to_add
    transformed_series = transformed_series.reshape((-1, nr_cols))
    transformed_series = pd.DataFrame(transformed_series, columns=column_names)
    return transformed_series

def define_multiindex(ls):
    """
    Create a multi index from a list
    """
    multi_index = list(zip(*ls))
    multi_index = pd.MultiIndex.from_tuples(multi_index)
    return multi_index

