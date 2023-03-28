import pickle
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

def save_object(data, save_path):
    """
    Save object to path

    :param data: object to save
    :param str save_path: path to save object to
    """
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(load_path):
    """
    Loads a previously saved object

    :param load_path: Path to load object from
    :return object: object loaded
    """
    with open(load_path, 'rb') as handle:
        data = pickle.load(handle)
    return data


class simplification_specs:
    """
    Two dataframes with (1) full resolution specifications and (2) reduces resolution specifications
    Dataframe with full resolution:
    - full resolution as index
    - hourly order
    - typical day
    Dataframe with reduced resolution
    - factors (how many times does each day occur)
    """
    def __init__(self, full_resolution_index):
        self.full_resolution = pd.DataFrame(index=full_resolution_index)
        self.reduced_resolution = []


def perform_k_means(full_resolution, nr_clusters):
    """
    Performs k-means clustering on a matrix

    Each row of the matrix corresponds to one observation (i.e. a day in this context)

    :param full_resolution: matrix of full resolution matrix
    :param nr_clusters: how many clusters
    :return __clustered_data: matrix with clustered data
    :return labels: labels for each clustered day
    """
    kmeans = KMeans(
        init="random",
        n_clusters=nr_clusters,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(full_resolution.to_numpy())
    series_names = pd.MultiIndex.from_tuples(full_resolution.columns.to_list())
    clustered_data = pd.DataFrame(kmeans.cluster_centers_, columns=series_names)
    return clustered_data, kmeans.labels_


def compile_hourly_order(day_labels, nr_clusters, nr_days_full_resolution, nr_time_intervals_per_day):
    """

    :param day_labels: labels for each typical day
    :param nr_clusters: how many clusters (i.e. typical days)
    :param nr_days_full_resolution: how many days in full resolution
    :param nr_time_intervals_per_day: how many time-intervals per day
    :return hourly_order: Hourly order of typical days/hours in full resolution
    """
    time_slices_cluster = np.arange(1, nr_time_intervals_per_day * nr_clusters + 1)
    time_slices_cluster = time_slices_cluster.reshape((-1, nr_time_intervals_per_day))
    hourly_order = np.zeros((nr_days_full_resolution, nr_time_intervals_per_day), dtype=np.int16)
    for day in range(0, nr_days_full_resolution):
        hourly_order[day] = time_slices_cluster[day_labels[day]]
    hourly_order = hourly_order.reshape((-1, 1))
    return hourly_order

def get_day_factors(keys):
    """
    Get factors for each hour

    This function assigns an integer to each hour in the full resolution, specifying how many times
    this hour occurs in the clustered data-set.
    """
    factors = pd.DataFrame(np.unique(keys, return_counts=True))
    factors = factors.transpose()
    factors.columns = ['timestep', 'factor']
    return factors

