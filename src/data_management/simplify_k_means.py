import src.data_management as dm

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

class DataHandle_KMeans(dm.DataHandle):
    """
    DataHandle sub-class for handling k-means clustered data

    This class is used to generate time series of typical days based on a full resolution of input data.
    """
    def __init__(self, data):
        """
        Constructor
        """
        self.topology = data.topology
        self.node_data = {}
        self.node_data_full_resolution = data.node_data
        self.technology_data = data.technology_data
        self.network_data = data.network_data
        self.specifications_time_resolution = {}

    def cluster_data(self, nr_clusters, nr_days_full_resolution= 365, nr_time_intervals_per_day=24):
        """
        Performs the clustering process

        This function performsthe k-means algorithm on the data resulting in a new DataHandle object that can be passed
        to the energhub class for optimization.

        :param DataHandle data: DataHandle containing data of the full resolution
        :param int nr_clusters: nr of clusters (tyical days) the data contains after the algorithm
        :param int nr_days_full_resolution: nr of days in data (full resolution)
        :param int nr_time_intervals_per_day: nr of time intervalls per day in data (full resolution)
        :return: instance of :class:`~ClusteredDataHandle`
        """
        def perform_kmeans_clustering(data_full_resolution, nr_clusters):
            """
            Function performing k-means clustering on full data

            :param dataframe data_full_resolution: data with resolution (number_of_days, number_hours_per_day)
            :param int nr_clusters: number of typical days
            :return: k-means cluster class
            """
            kmeans = KMeans(
                init="random",
                n_clusters=nr_clusters,
                n_init=10,
                max_iter=300,
                random_state=42
            )
            kmeans.fit(data_full_resolution.to_numpy())
            series_names = pd.MultiIndex.from_tuples(data_full_resolution.columns.to_list())
            return kmeans

        def get_keys_from_kmeans(kmeans):
            """
            Gets typical day for each day of the full resolution.

            This function assigns a typical day and a typical hour to each time-step of the full resolution data.
            """
            time_slices_cluster = np.arange(1, nr_time_intervals_per_day * nr_clusters + 1)
            time_slices_cluster = time_slices_cluster.reshape((-1, nr_time_intervals_per_day))
            day_labels = kmeans.labels_
            keys = pd.DataFrame(index=self.topology['timesteps'])
            keys['typical_day'] = np.repeat(kmeans.labels_, nr_time_intervals_per_day)
            keys_temp = np.zeros((nr_days_full_resolution, nr_time_intervals_per_day), dtype=np.int16)
            for day in range(0, nr_days_full_resolution):
                keys_temp[day] = time_slices_cluster[day_labels[day]]
            keys_temp = keys_temp.reshape((-1, 1))
            keys['hourly_order'] = keys_temp
            return keys

        # Reshape data and perform k-means clustering
        full_resolution = self.compile_and_reshape_full_resolution(nr_time_intervals_per_day)
        kmeans = perform_kmeans_clustering(full_resolution, nr_clusters)

        # Write keys and factors to self
        self.specifications_time_resolution['keys'] = get_keys_from_kmeans(kmeans)
        self.specifications_time_resolution['factors'] = dm.get_day_factors(self.specifications_time_resolution['keys'])

        # Reformulate number of timesteps
        self.topology['timesteps'] = range(0, nr_clusters * nr_time_intervals_per_day)

        # Read data back in
        series_names = pd.MultiIndex.from_tuples(full_resolution.columns.to_list())
        clustered_data = pd.DataFrame(kmeans.cluster_centers_, columns=series_names)
        self.read_clustered_data(clustered_data)

    def compile_and_reshape_full_resolution(self, nr_time_intervals_per_day):
        """
        Reshape time-series data to format with each row containing one day of data.
        """

        # Create a multi index from a list
        def define_multiindex(ls):
            multi_index = list(zip(*ls))
            multi_index = pd.MultiIndex.from_tuples(multi_index)
            return multi_index

        RES_tecs = dm.get_RES_technologies(self.topology['technologies'])
        full_resolution = pd.DataFrame()
        for nodename in self.node_data_full_resolution:
            for series in self.node_data_full_resolution[nodename]:
                if not series == 'climate_data':
                    for carrier in self.node_data_full_resolution[nodename][series]:
                        series_names = define_multiindex([
                            [nodename] * nr_time_intervals_per_day,
                            [series] * nr_time_intervals_per_day,
                            [carrier] * nr_time_intervals_per_day,
                            list(range(1, nr_time_intervals_per_day + 1))
                        ])
                        to_add = dm.reshape_df(self.node_data_full_resolution[nodename][series][carrier],
                                            series_names, nr_time_intervals_per_day)
                        full_resolution = pd.concat([full_resolution, to_add], axis=1)
            for tec in RES_tecs[nodename]:
                series_names = define_multiindex([
                    [nodename] * nr_time_intervals_per_day,
                    [tec] * nr_time_intervals_per_day,
                    ['capacity_factor'] * nr_time_intervals_per_day,
                    list(range(1, nr_time_intervals_per_day + 1))
                ])
                to_add = dm.reshape_df(self.technology_data[nodename][tec]['fit']['capacity_factor'],
                                    series_names, nr_time_intervals_per_day)
                full_resolution = pd.concat([full_resolution, to_add], axis=1)
        return full_resolution

    def read_clustered_data(self, clustered_data):
        """
        Reads clustered data in required format to DataHandle
        """
        RES_tecs = dm.get_RES_technologies(self.topology['technologies'])
        for nodename in self.node_data_full_resolution:
            self.node_data[nodename] = {}
            for series in self.node_data_full_resolution[nodename]:
                if not series == 'climate_data':
                    self.node_data[nodename][series] = pd.DataFrame()
                    for carrier in self.node_data_full_resolution[nodename][series]:
                        self.node_data[nodename][series][carrier]= \
                            dm.reshape_df(clustered_data[nodename][series][carrier],
                                       None, 1)
            for tec in RES_tecs[nodename]:
                series_data = dm.reshape_df(clustered_data[nodename][tec]['capacity_factor'], None, 1)
                series_data = series_data.to_numpy()
                self.technology_data[nodename][tec]['fit']['capacity_factor'] = \
                    series_data