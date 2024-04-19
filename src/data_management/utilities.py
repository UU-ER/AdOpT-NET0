import warnings
from pathlib import Path
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
    average = np.array(to_average.mean(axis=1))

    return average


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


class GlobalData:
    """
    Class to handle global data. All global time-dependent input data goes here
    """

    def __init__(self, topology):
        self.data = {}
        self.data_clustered = {}

        variables = ["subsidy", "tax"]
        self.data["carbon_prices"] = pd.DataFrame(index=topology.timesteps)
        for var in variables:
            self.data["carbon_prices"][var] = np.zeros(len(topology.timesteps))


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


def create_input_data_folder_template(base_path: Path | str) -> None:
    """
    This function creates the input data folder structure required to organize the input data to the model.
    Note that the folder needs to already exist with a topology.json file in it that specifies the nodes, carriers,
    timesteps, investement periods and the length of the investment period.

    You can create an examplary json template with the function `create_topology_template`

    :param str/Path base_path: path to folder
    """
    # Convert to Path
    if isinstance(base_path, str):
        base_path = Path(base_path)

    # Read topology.json
    with open(base_path / "topology.json") as json_file:
        topology = json.load(json_file)

    timesteps = pd.date_range(
        start=topology["start_date"],
        end=topology["end_date"],
        freq=topology["resolution"],
    )

    # template_jsons:
    networks = {"existing": [], "new": []}
    technologies = {"existing": [], "new": []}
    carrier_data = pd.DataFrame(
        index=timesteps,
        columns=[
            "Demand",
            "Import limit",
            "Export limit",
            "Import price",
            "Export price",
            "Import emission factor",
            "Export emission factor",
            "Generic production",
        ],
    )
    climate_data = pd.DataFrame(
        index=timesteps,
        columns=["ghi", "dni", "dhi", "temp_air", "rh", "TECHNOLOGYNAME_hydro_inflow"],
    )
    carbon_cost = pd.DataFrame(index=timesteps, columns=["price", "subsidy"])
    node_locations = pd.DataFrame(index=topology["nodes"], columns=["lon", "lat", "alt"])
    node_locations.to_csv(base_path / "NodeLocations.csv", sep=";")

    # Make folder structure
    for investment_period in topology["investment_periods"]:
        (base_path / investment_period).mkdir(parents=True, exist_ok=True)

        # Networks
        with open(base_path / investment_period / "Networks.json", "w") as f:
            json.dump(networks, f, indent=4)
        (base_path / investment_period / "network_data").mkdir(
            parents=True, exist_ok=True
        )

        # Node data
        (base_path / investment_period / "node_data").mkdir(parents=True, exist_ok=True)
        for node in topology["nodes"]:
            (base_path / investment_period / "node_data" / node / "carrier_data").mkdir(
                parents=True, exist_ok=True
            )
            with open(
                base_path
                / investment_period
                / "node_data"
                / node
                / "Technologies.json",
                "w",
            ) as f:
                json.dump(technologies, f, indent=4)
            for carrier in topology["carriers"]:
                carrier_data.to_csv(
                    base_path
                    / investment_period
                    / "node_data"
                    / node
                    / "carrier_data"
                    / f"{carrier}.csv",
                    sep=";",
                )
            climate_data.to_csv(
                base_path / investment_period / "node_data" / node / "ClimateData.csv",
                sep=";",
            )
            carbon_cost.to_csv(
                base_path / investment_period / "node_data" / node / "CarbonCost.csv",
                sep=";",
            )
            (
                base_path / investment_period / "node_data" / node / "technology_data"
            ).mkdir(parents=True, exist_ok=True)


def create_optimization_templates(path: Path | str) -> None:
    """
    Creates an examplary topology json file in the specified path.

    :param str/Path path: path to folder to create topology.json
    """
    if isinstance(path, str):
        path = Path(path)

    topology_template = {
        "nodes": ["node1", "node2"],
        "carriers": ["electricity", "hydrogen"],
        "investment_periods": ["period1", "period2"],
        "start_date": "2022-01-01 00:00",
        "end_date": "2022-12-31 23:00",
        "resolution": "1h",
        "investment_period_length": 1,
    }

    configuration_template = {
        "optimization": {
            "objective": {
                "description": "String specifying the objective/type of optimization.",
                "options": [
                    "costs",
                    "emissions_pos",
                    "emissions_net",
                    "emissions_minC",
                    "costs_emissionlimit",
                    "pareto",
                ],
                "default": "costs",
            },
            "monte_carlo": {
                "on": {
                    "description": "Turn Monte Carlo simulation on.",
                    "options": [0, 1],
                    "default": 0,
                },
                "sd": {
                    "description": "Value defining the range in which variables are varied in Monte Carlo simulations (defined as the standard deviation of the original value).",
                    "default": 0.2,
                },
                "N": {
                    "description": "Number of Monte Carlo simulations.",
                    "default": 100,
                },
                "on_what": {
                    "description": "List: Defines component to vary.",
                    "options": ["Technologies", "ImportPrices", "ExportPrices"],
                    "default": "Technologies",
                },
            },
            "pareto_points": {"description": "Number of Pareto points.", "default": 5},
            "timestaging": {
                "description": "Defines number of timesteps that are averaged (0 = off).",
                "default": 0,
            },
            "typicaldays": {
                "N": {
                    "description": "Determines number of typical days (0 = off).",
                    "default": 0,
                },
                "method": {
                    "description": "Determine method used for modeling technologies with typical days.",
                    "options": [2],
                    "default": 2,
                },
            },
            "multiyear": {
                "description": "Enable multiyear analysis, if turned off max time horizon is 1 year.",
                "options": [0, 1],
                "default": 0,
            },
        },
        "solveroptions": {
            "solver": {
                "description": "String specifying the solver used.",
                "default": "gurobi",
            },
            "mipgap": {"description": "Value to define MIP gap.", "default": 0.001},
            "timelim": {
                "description": "Value to define time limit in hours.",
                "default": 10,
            },
            "threads": {
                "description": "Value to define number of threads (default is maximum available).",
                "default": 0,
            },
            "mipfocus": {
                "description": "Modifies high level solution strategy.",
                "options": [0, 1, 2, 3],
                "default": 0,
            },
            "nodefilestart": {
                "description": "Parameter to decide when nodes are compressed and written to disk.",
                "default": 60,
            },
            "method": {
                "description": "Defines algorithm used to solve continuous models.",
                "options": [-1, 0, 1, 2, 3, 4, 5],
                "default": -1,
            },
            "heuristics": {
                "description": "Parameter to determine amount of time spent in MIP heuristics.",
                "default": 0.05,
            },
            "presolve": {
                "description": "Controls the presolve level.",
                "options": [-1, 0, 1, 2],
                "default": -1,
            },
            "branchdir": {
                "description": "Determines which child node is explored first in the branch-and-cut.",
                "options": [-1, 0, 1],
                "default": 0,
            },
            "lpwarmstart": {
                "description": "Controls whether and how warm start information is used for LP.",
                "options": [0, 1, 2],
                "default": 0,
            },
            "intfeastol": {
                "description": "Value that determines the integer feasibility tolerance.",
                "default": 1e-05,
            },
            "feastol": {
                "description": "Value that determines feasibility for all constraints.",
                "default": 1e-06,
            },
            "numericfocus": {
                "description": "Degree of which Gurobi tries to detect and manage numeric issues.",
                "options": [0, 1, 2, 3],
                "default": 0,
            },
            "cuts": {
                "description": "Setting defining the aggressiveness of the global cut.",
                "options": [-1, 0, 1, 2, 3],
                "default": -1,
            },
        },
        "reporting": {
            "save_detailed": {
                "description": "Setting to select how the results are saved. When turned off only the summary is saved.",
                "options": [0, 1],
                "default": 1,
            },
            "save_path": {
                "description": "Option to define the save path.",
                "default": "./userData/",
            },
            "case_name": {
                "description": "Option to define a case study name that is added to the results folder name.",
                "default": -1,
            },
            "write_solution_diagnostics": {
                "description": "If 1, writes solution quality, if 2 also writes pyomo to Gurobi variable map and constraint map to file.",
                "options": [0, 1, 2],
                "default": 0,
            },
        },
        "energybalance": {
            "violation": {
                "description": "Determines the energy balance violation price (-1 is no violation allowed).",
                "default": -1,
            },
            "copperplate": {
                "description": "Determines if a copperplate approach is used.",
                "options": [0, 1],
                "default": 0,
            },
        },
        "economic": {
            "global_discountrate": {
                "description": "Determines if and which global discount rate is used. This holds for the CAPEX of all technologies and networks.",
                "default": -1,
            },
            "global_simple_capex_model": {
                "description": "Determines if the CAPEX model of technologies is set to 1 for all technologies.",
                "options": [0, 1],
                "default": 0,
            },
        },
        "performance": {
            "dynamics": {
                "description": "Determines if dynamics are used.",
                "options": [0, 1],
                "default": 0,
            }
        },
        "scaling": {
            "scaling": {
                "description": "Determines if the model is scaled. If 1, it uses global and component specific scaling factors.",
                "options": [0, 1],
                "default": 0,
            },
            "scaling_factors": {
                "energy_vars": {
                    "description": "Scaling factor used for all energy variables.",
                    "default": 0.001,
                },
                "cost_vars": {
                    "description": "Scaling factor used for all cost variables.",
                    "default": 0.001,
                },
                "objective": {
                    "description": "Scaling factor used for the objective function.",
                    "default": 1,
                },
            },
        },
    }

    with open(path / "topology.json", "w") as f:
        json.dump(topology_template, f, indent=4)
    with open(path / "config_optimization.json", "w") as f:
        json.dump(configuration_template, f, indent=4)


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
    with open(path / "topology.json") as json_file:
        topology = json.load(json_file)

    for investment_period in topology["investment_periods"]:

        # Check investment periods
        check_path = path / investment_period
        check_path_existance(
            check_path,
            f"The investment period {investment_period} is missing in {check_path}",
        )

        # Check if all networks have a json file
        check_path_existance(
            check_path / "Networks.json",
            f"A Network.json file is missing in {check_path}",
        )
        with open(check_path / "Networks.json") as json_file:
            networks = json.load(json_file)
        networks = set(networks["existing"] + networks["new"])
        for network in networks:
            check_path_existance(
                check_path / "network_data" / (network + ".json"),
                f"A json file for {network} is missing in {check_path / 'network_data'}",
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
                technologies_at_node["existing"] + technologies_at_node["new"]
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

    print("Input data folder has been checked successfully - no errors occurred.")
