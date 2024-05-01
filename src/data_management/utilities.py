from pathlib import Path
import dill as pickle
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from types import SimpleNamespace
import pvlib
import os
import json
from ..logger import logger

from ..components.technologies import *


def save_object(data, save_path):
    """
    Save object to path

    :param data: object to save
    :param Path save_path: path to save object to
    """
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle)


def load_object(load_path):
    """
    Loads a previously saved object

    :param Path load_path: Path to load object from
    :return object: object loaded
    """
    with open(load_path, "rb") as handle:
        data = pd.read_pickle(handle)

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
        init="random", n_clusters=nr_clusters, n_init=10, max_iter=300, random_state=42
    )
    kmeans.fit(full_resolution.to_numpy())
    series_names = pd.MultiIndex.from_tuples(full_resolution.columns.to_list())
    clustered_data = pd.DataFrame(kmeans.cluster_centers_, columns=series_names)
    return clustered_data, kmeans.labels_


def compile_sequence(
    day_labels, nr_clusters, nr_days_full_resolution, nr_time_intervals_per_day
):
    """

    :param day_labels: labels for each typical day
    :param nr_clusters: how many clusters (i.e. typical days)
    :param nr_days_full_resolution: how many days in full resolution
    :param nr_time_intervals_per_day: how many time-intervals per day
    :return sequence: Hourly order of typical days/hours in full resolution
    """
    time_slices_cluster = np.arange(1, nr_time_intervals_per_day * nr_clusters + 1)
    time_slices_cluster = time_slices_cluster.reshape((-1, nr_time_intervals_per_day))
    sequence = np.zeros(
        (nr_days_full_resolution, nr_time_intervals_per_day), dtype=np.int16
    )
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
    factors.columns = ["timestep", "factor"]
    return factors


def compile_full_resolution_matrix(data_full_res, nr_time_intervals_per_day):
    """
    Compiles full resolution matrix to be clustered

    """
    time_intervals = range(1, nr_time_intervals_per_day + 1)
    nr_of_days_full_res = len(data_full_res) // nr_time_intervals_per_day

    # Reshape each column into a DataFrame with nr_of_days_full_res rows and nr_time_intervals_per_day columns
    reshaped_data = pd.DataFrame()
    for col in data_full_res.columns:
        col_data = data_full_res[col].values.reshape(
            nr_of_days_full_res, nr_time_intervals_per_day
        )
        col_df = pd.DataFrame(col_data, columns=time_intervals)
        reshaped_data = pd.concat([reshaped_data, col_df], axis=1)

    # Repeat each row of the index frame separately and add time_intervals as the last column
    index_frame = data_full_res.columns.to_frame()
    repeated_frames = []
    for _, row in index_frame.iterrows():
        repeated_index = pd.concat(
            [pd.DataFrame(row).T] * nr_time_intervals_per_day, ignore_index=True
        )
        repeated_index["Time Interval"] = sorted(list(time_intervals))
        repeated_frames.append(repeated_index)
    repeated_index = pd.concat(repeated_frames)

    # Set index with the modified frame
    reshaped_data.columns = pd.MultiIndex.from_frame(repeated_index)

    return reshaped_data


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


def average_timeseries_data(data_matrix, nr_timesteps_averaged, time_index):
    """
    Averages the nr_timesteps_averaged in the DataFrame.

    Parameters:
        series (pd.DataFrame): Input DataFrame.
        nr_timesteps_averaged (int): Number of consecutive rows to average.

    Returns:
        pd.DataFrame: New DataFrame with averaged rows.
    """
    averaged_df = pd.DataFrame(index=time_index, columns=data_matrix.columns)
    for col in data_matrix.columns:
        reshaped_data = data_matrix[col].values.reshape(-1, nr_timesteps_averaged)
        averages = np.mean(reshaped_data, axis=1)
        averaged_df[col] = averages

    return averaged_df


def average_timeseries_data_clustered(
    data_matrix, nr_timesteps_averaged, clustered_days
):
    """
    Averages the nr_timesteps_averaged in the DataFrame.

    Parameters:
        data_matrix (pd.DataFrame): Input DataFrame.
        nr_timesteps_averaged (int): Number of consecutive rows to average.
        clustered_days (int): Number of days for which the data is clustered.

    Returns:
        pd.DataFrame: New DataFrame with averaged rows.
    """
    averaged_dfs = []
    for i in range(0, data_matrix.shape[1], nr_timesteps_averaged):
        start_idx = i
        end_idx = min(i + nr_timesteps_averaged, data_matrix.shape[1])
        col_name = data_matrix.columns[start_idx]
        averaged_values = data_matrix.iloc[:, start_idx:end_idx].mean(axis=1)
        averaged_dfs.append(pd.DataFrame({col_name: averaged_values}))

    return pd.concat(averaged_dfs, axis=1)


def calculate_dni(data, lon, lat):
    """
    Calculate direct normal irradiance from ghi and dhi
    :param DataFrame data: climate data
    :return: data: climate data including dni
    """
    zenith = pvlib.solarposition.get_solarposition(data.index, lat, lon)
    data["dni"] = pvlib.irradiance.dni(
        data["ghi"].to_numpy(), data["dhi"].to_numpy(), zenith["zenith"].to_numpy()
    )
    data["dni"] = data["dni"].fillna(0)
    data["dni"] = data["dni"].where(data["dni"] > 0, 0)

    return data["dni"]


def shorten_input_data(time_series, nr_time_steps):
    """
    Shortens time series to required length

    :param list time_series: time_series to shorten
    :param int nr_time_steps: nr of time steps to shorten to
    """
    if len(time_series) != nr_time_steps:
        warnings.warn(
            "Time series is longer than chosen time horizon - taking only the first "
            + "couple of time slices"
        )
        time_series = time_series[0:nr_time_steps]

    return time_series


class NodeData:
    """
    Class to handle node data
    """

    def __init__(self, topology):
        # Initialize Node Data (all time-dependent input data goes here)
        self.data = {}
        self.data_clustered = {}
        variables = [
            "demand",
            "production_profile",
            "import_prices",
            "import_limit",
            "import_emissionfactors",
            "export_prices",
            "export_limit",
            "export_emissionfactors",
        ]

        for var in variables:
            self.data[var] = pd.DataFrame(index=topology.timesteps)
            for carrier in topology.carriers:
                self.data[var][carrier] = 0
        self.data["climate_data"] = pd.DataFrame(index=topology.timesteps)

        self.options = SimpleNamespace()
        self.options.production_profile_curtailment = {}
        for carrier in topology.carriers:
            self.options.production_profile_curtailment[carrier] = 0

        self.location = SimpleNamespace()
        self.location.lon = None
        self.location.lat = None
        self.location.altitude = None


def select_technology(tec_data):
    """
    Returns the correct subclass for a technology

    :param str tec_name: Technology Name
    :param int existing: if technology is existing
    :return: Technology Class
    """
    # Generic tecs
    if tec_data["tec_type"] == "RES":
        return Res(tec_data)
    elif tec_data["tec_type"] == "CONV1":
        return Conv1(tec_data)
    elif tec_data["tec_type"] == "CONV2":
        return Conv2(tec_data)
    elif tec_data["tec_type"] == "CONV3":
        return Conv3(tec_data)
    elif tec_data["tec_type"] == "CONV4":
        return Conv4(tec_data)
    elif tec_data["tec_type"] == "STOR":
        return Stor(tec_data)
    # Specific tecs
    elif tec_data["tec_type"] == "DAC_Adsorption":
        return DacAdsorption(tec_data)
    elif tec_data["tec_type"].startswith("GasTurbine"):
        return GasTurbine(tec_data)
    elif tec_data["tec_type"].startswith("HeatPump"):
        return HeatPump(tec_data)
    elif tec_data["tec_type"] == "HydroOpen":
        return HydroOpen(tec_data)


def open_json(tec, load_path):
    # Read in JSON files
    for path, subdirs, files in os.walk(load_path):
        if "data" in locals():
            break
        else:
            for name in files:
                if (tec + ".json") == name:
                    filepath = os.path.join(path, name)
                    with open(filepath) as json_file:
                        data = json.load(json_file)
                    break

    # Assign name
    if "data" in locals():
        data["Name"] = tec
    else:
        raise Exception("There is no json data file for technology " + tec)

    return data


def check_input_data_consistency(path: Path | str) -> None:
    """
    Checks if the topology is consistent with the input data.

    :param str/Path node: node as specified in the topology
    """

    def check_path_existance(path: Path, error_message: str) -> None:
        if not os.path.exists(path):
            raise Exception(error_message)

    # Convert to Path
    if isinstance(path, str):
        path = Path(path)

    # Read topology
    with open(path / "Topology.json") as json_file:
        topology = json.load(json_file)

    for investment_period in topology["investment_periods"]:

        # Check investment periods
        check_path = path / investment_period
        check_path_existance(
            check_path,
            f"The investment period {investment_period} is missing in {check_path}",
        )

        # Check networks
        check_path_existance(
            check_path / "Networks.json",
            f"A Network.json file is missing in {check_path}",
        )
        with open(check_path / "Networks.json") as json_file:
            all_networks = json.load(json_file)
        for type in all_networks.keys():
            networks = all_networks[type]
            for network in networks:
                check_path_existance(
                    check_path / "network_data" / (network + ".json"),
                    f"A json file for {network} is missing in {check_path / 'network_data'}",
                )
                check_path_existance(
                    check_path / "network_topology" / type,
                    f"A directory for {network} is missing in {check_path / 'network_topology'}",
                )
                check_path_existance(
                    check_path / "network_topology" / type / network / "connection.csv",
                    f"A connection.csv for {network} is missing in {check_path / 'network_topology' / type / network}",
                )
                check_path_existance(
                    check_path / "network_topology" / type / network / "distance.csv",
                    f"A distance.csv for {network} is missing in {check_path / 'network_topology' / type / network}",
                )
                check_path_existance(
                    check_path
                    / "network_topology"
                    / type
                    / network
                    / "size_max_arcs.csv",
                    f"A size_max_arcs.csv for {network} is missing in {check_path / 'network_topology' / type / network}",
                )
                if type == "existing":
                    check_path_existance(
                        check_path / "network_topology" / type / network / "size.csv",
                        f"A size.csv for {network} is missing in {check_path / 'network_topology' / type / network}",
                    )

        for node in topology["nodes"]:

            # Check nodes
            check_node_path = path / investment_period / "node_data" / node
            check_path_existance(
                check_node_path, f"The node {node} is missing in {check_node_path}"
            )

            # Check if all files are there
            check_path_existance(
                check_node_path / "ClimateData.csv",
                f"ClimateData.csv is missing in {check_node_path}",
            )
            check_path_existance(
                check_node_path / "CarbonCost.csv",
                f"CarbonCost.csv is missing in {check_node_path}",
            )
            check_path_existance(
                check_node_path / "Technologies.json",
                f"Technologies.json is missing in {check_node_path}",
            )

            # Check if all technologies have a json file
            with open(check_node_path / "Technologies.json") as json_file:
                technologies_at_node = json.load(json_file)
            technologies_at_node = set(
                list(technologies_at_node["existing"].keys())
                + technologies_at_node["new"]
            )
            for technology in technologies_at_node:
                check_path_existance(
                    check_node_path / "technology_data" / (technology + ".json"),
                    f"A json file for {technology} is missing in {check_node_path / 'technology_data'}",
                )
                # TODO: Check if carriers are in carrier set

            # Check if all carriers are there
            for carrier in topology["carriers"]:
                check_path_existance(
                    check_node_path / "carrier_data" / (carrier + ".csv"),
                    f"Data for carrier {carrier} is missing in {check_node_path}",
                )

    logger.info("Input data folder has been checked successfully - no errors occurred.")
