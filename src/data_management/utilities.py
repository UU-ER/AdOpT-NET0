import warnings
import dill as pickle
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from types import SimpleNamespace
import pvlib
import os
import json

from ..components.technologies import *


def save_object(data, save_path):
    """
    Save object to path

    :param data: object to save
    :param Path save_path: path to save object to
    """
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle)


def load_object(load_path):
    """
    Loads a previously saved object

    :param Path load_path: Path to load object from
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


def average_series(series, nr_timesteps_averaged):
    """
    Averages a number of timesteps
    """
    to_average = reshape_df(series, None, nr_timesteps_averaged)
    average =  np.array(to_average.mean(axis=1))

    return average


def calculate_dni(data, lon, lat):
    """
    Calculate direct normal irradiance from ghi and dhi
    :param DataFrame data: climate data
    :return: data: climate data including dni
    """
    zenith = pvlib.solarposition.get_solarposition(data.index, lat, lon)
    data['dni'] = pvlib.irradiance.dni(data['ghi'].to_numpy(), data['dhi'].to_numpy(), zenith['zenith'].to_numpy())
    data['dni'] = data['dni'].fillna(0)
    data['dni'] = data['dni'].where(data['dni'] > 0, 0)

    return data['dni']


def shorten_input_data(time_series, nr_time_steps):
    """
    Shortens time series to required length

    :param list time_series: time_series to shorten
    :param int nr_time_steps: nr of time steps to shorten to
    """
    if len(time_series) != nr_time_steps:
        warnings.warn('Time series is longer than chosen time horizon - taking only the first ' + \
                      'couple of time slices')
        time_series = time_series[0:nr_time_steps]

    return time_series


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


class GlobalData():
    """
    Class to handle global data. All global time-dependent input data goes here
    """
    def __init__(self, topology):
        self.data = {}
        self.data_clustered = {}

        variables = ['subsidy', 'tax']
        self.data['carbon_prices'] = pd.DataFrame(index=topology.timesteps)
        for var in variables:
            self.data['carbon_prices'][var] = np.zeros(len(topology.timesteps))
            
def select_technology(tec_data):
    """
    Returns the correct subclass for a technology
    
    :param str tec_name: Technology Name 
    :param int existing: if technology is existing 
    :return: Technology Class
    """
    # Generic tecs
    if tec_data['tec_type'] == 'RES':
        return Res(tec_data)
    elif tec_data['tec_type'] == 'CONV1':
        return Conv1(tec_data)
    elif tec_data['tec_type'] == 'CONV2':
        return Conv2(tec_data)
    elif tec_data['tec_type'] == 'CONV3':
        return Conv3(tec_data)
    elif tec_data['tec_type'] == 'CONV4':
        return Conv4(tec_data)
    elif tec_data['tec_type'] == 'STOR':
        return Stor(tec_data)
    # Specific tecs
    elif tec_data['tec_type'] == 'DAC_Adsorption':
        return DacAdsorption(tec_data)
    elif tec_data['tec_type'].startswith('GasTurbine'):
        return GasTurbine(tec_data)
    elif tec_data['tec_type'].startswith('HeatPump'):
        return HeatPump(tec_data)
    elif tec_data['tec_type'] == 'HydroOpen':
        return HydroOpen(tec_data)


def open_json(tec, load_path):
    # Read in JSON files
    for path, subdirs, files in os.walk(load_path):
        if 'data' in locals():
            break
        else:
            for name in files:
                if (tec + '.json') == name:
                    filepath = os.path.join(path, name)
                    with open(filepath) as json_file:
                        data = json.load(json_file)
                    break

    # Assign name
    if 'data' in locals():
        data['Name'] = tec
    else:
        raise Exception('There is no json data file for technology ' + tec)

    return data
    





