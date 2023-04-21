import pickle
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from types import SimpleNamespace


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
    :return clustered_data: matrix with clustered data
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


def compile_sequence(day_labels, nr_clusters, nr_days_full_resolution, nr_time_intervals_per_day):
    """

    :param day_labels: labels for each typical day
    :param nr_clusters: how many clusters (i.e. typical days)
    :param nr_days_full_resolution: how many days in full resolution
    :param nr_time_intervals_per_day: how many time-intervals per day
    :return sequence: Hourly order of typical days/hours in full resolution
    """
    time_slices_cluster = np.arange(1, nr_time_intervals_per_day * nr_clusters + 1)
    time_slices_cluster = time_slices_cluster.reshape((-1, nr_time_intervals_per_day))
    sequence = np.zeros((nr_days_full_resolution, nr_time_intervals_per_day), dtype=np.int16)
    for day in range(0, nr_days_full_resolution):
        sequence[day] = time_slices_cluster[day_labels[day]]
    sequence = sequence.reshape((-1, 1))
    return sequence

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

def flag_tecs_for_clustering(data):
    """
    Creates a dictonary with flags for RES technologies

    These technologies contain time-dependent input data, i.e. capacity factors.
    :return dict tecs_flagged_for_clustering: flags for technologies and nodes

    """
    tecs_flagged_for_clustering = {}
    for node in data.topology.nodes:
        tecs_flagged_for_clustering[node] = {}
        for technology in data.technology_data[node]:
            if data.technology_data[node][technology].technology_model == 'RES':
                tecs_flagged_for_clustering[node][technology] = 'capacity_factor'
            elif data.technology_data[node][technology].technology_model == 'STOR':
                tecs_flagged_for_clustering[node][technology] = 'ambient_loss_factor'
            elif data.technology_data[node][technology].technology_model == 'DAC_Adsorption':
                tecs_flagged_for_clustering[node][technology] = ['alpha','beta','b','gamma','delta','a']
            elif data.technology_data[node][technology].technology_model.startswith('HeatPump_'):
                if data.technology_data[node][technology].performance_data['performance_function_type'] == 1:
                    tecs_flagged_for_clustering[node][technology] = ['alpha1']
                else:
                    tecs_flagged_for_clustering[node][technology] = ['alpha1','alpha2']
            elif data.technology_data[node][technology].technology_model.startswith('GasTurbine_'):
                tecs_flagged_for_clustering[node][technology] = ['alpha','beta','epsilon','f']

    return tecs_flagged_for_clustering

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


class NodeData():
    """
    Class to handle node data
    """
    def __init__(self, topology):
        # Initialize Node Data (all time-dependent input data goes here)
        self.data = {}
        self.data_clustered = {}
        variables = ['demand',
                     'production_profile',
                     'import_prices',
                     'import_limit',
                     'import_emissionfactors',
                     'export_prices',
                     'export_limit',
                     'export_emissionfactors']

        for var in variables:
            self.data[var] = pd.DataFrame(index=topology.timesteps)
            for carrier in topology.carriers:
                self.data[var][carrier] = 0
        self.data['climate_data'] = pd.DataFrame(index=topology.timesteps)

        self.options = SimpleNamespace()
        self.options.production_profile_curtailment = {}
        for carrier in topology.carriers:
            self.options.production_profile_curtailment[carrier]= 0

        self.location = SimpleNamespace()
        self.location.lon = None
        self.location.lat = None
        self.location.altitude = None



