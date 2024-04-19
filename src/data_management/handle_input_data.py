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

    The Data Handle class reads in the data previously specified in the respective input data folder. Pass the
    folder path when initializing the class:

    .. code-block:: python

       data = DataHandle(path)
    """

    def __init__(self, data_path: Path | str) -> None:
        """
        Constructor

        :param str/Path base_path: path to folder
        """
        # Convert to Path
        if isinstance(data_path, str):
            data_path = Path(data_path)

        # Attributes
        self.data_path = data_path
        self.time_series = {}
        self.energybalance_options = {}
        self.technology_data = {}
        self.network_data = {}
        self.node_locations = {}
        self.model_config = {}
        self.k_means_specs = {}

        # Check consistency
        check_input_data_consistency(data_path)

        # Read in data
        self._read_topology()
        self._read_model_config()
        self._read_time_series()
        self._read_node_locations()
        self._read_energybalance_options()
        self._read_technology_data()
        self._read_network_data()

        # Clustering/Averaging algorithms
        if self.model_config["optimization"]["typicaldays"]["N"]["default"] != 0:
            self._cluster_data()

    def _read_topology(self) -> None:
        """
        Reads topology from path
        """
        with open(self.data_path / "topology.json") as json_file:
            self.topology = json.load(json_file)

        self.topology["time_index"] = {}
        self.topology["time_index"]["full"] = pd.date_range(
            start=self.topology["start_date"],
            end=self.topology["end_date"],
            freq=self.topology["resolution"],
        )

    def _read_model_config(self) -> None:
        """
        Reads model configuration
        """
        with open(self.data_path / "config_optimization.json") as json_file:
            self.model_config = json.load(json_file)

    def _read_time_series(self) -> None:
        """
        Reads all time-series data
        """
        data = {}
        for investment_period in self.topology["investment_periods"]:
            for node in self.topology["nodes"]:
                # Carbon Costs
                carbon_cost = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "node_data"
                    / node
                    / "CarbonCost.csv",
                    sep=";",
                    index_col=0,
                ).to_dict(orient="list")
                for key in carbon_cost.keys():
                    data[(investment_period, node, "CarbonCost", None, key)] = (
                        carbon_cost[key]
                    )

                # Climate Data
                climate_data = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "node_data"
                    / node
                    / "ClimateData.csv",
                    sep=";",
                    index_col=0,
                ).to_dict(orient="list")
                for key in climate_data.keys():
                    data[(investment_period, node, "ClimateData", None, key)] = (
                        climate_data[key]
                    )

                # Carrier Data
                for carrier in self.topology["carriers"]:
                    carrier_data = pd.read_csv(
                        self.data_path
                        / investment_period
                        / "node_data"
                        / node
                        / "carrier_data"
                        / (carrier + ".csv"),
                        sep=";",
                        index_col=0,
                    ).to_dict(orient="list")
                    for key in carrier_data.keys():
                        data[(investment_period, node, "ClimateData", carrier, key)] = (
                            carrier_data[key]
                        )

        self.time_series["full"] = pd.DataFrame(
            data, index=self.topology["time_index"]["full"]
        )

    def _read_node_locations(self) -> None:
        """
        Reads node locations
        """
        self.node_locations = pd.read_csv(self.data_path / "NodeLocations.csv", sep=";")

    def _read_energybalance_options(self) -> None:
        """
        Reads energy balance options
        """
        for investment_period in self.topology["investment_periods"]:
            self.energybalance_options[investment_period] = {}
            for node in self.topology["nodes"]:
                with open(
                    self.data_path
                    / investment_period
                    / "node_data"
                    / node
                    / "carrier_data"
                    / "EnergybalanceOptions.json"
                ) as json_file:
                    energybalance_options = json.load(json_file)
                self.energybalance_options[investment_period][
                    node
                ] = energybalance_options

    def _read_technology_data(self, aggregation_type: str = "full") -> None:
        """
        Reads all technology data and fits it
        """
        technology_data = {}
        for investment_period in self.topology["investment_periods"]:
            technology_data[investment_period] = {}
            for node in self.topology["nodes"]:
                technology_data[investment_period][node] = {}
                with open(
                    self.data_path
                    / investment_period
                    / "node_data"
                    / node
                    / "Technologies.json"
                ) as json_file:
                    technologies_at_node = json.load(json_file)

                # New technologies
                for technology in technologies_at_node["new"]:
                    tec_data = open_json(
                        technology,
                        self.data_path
                        / investment_period
                        / "node_data"
                        / node
                        / "technology_data",
                    )
                    tec_data["name"] = technology
                    tec_data = select_technology(tec_data)
                    tec_data.fit_technology_performance(
                        self.time_series[aggregation_type][investment_period][node]
                    )
                    technology_data[investment_period][node][technology] = tec_data

                # Existing technologies
                for technology in technologies_at_node["existing"]:
                    tec_data = open_json(
                        technology,
                        self.data_path
                        / investment_period
                        / "node_data"
                        / node
                        / "technology_data",
                    )
                    tec_data["name"] = technology
                    tec_data = select_technology(tec_data)
                    tec_data.existing = 1
                    tec_data.size_initial = technologies_at_node["existing"][technology]
                    tec_data.fit_technology_performance(
                        self.time_series[aggregation_type][investment_period][node]
                    )
                    technology_data[investment_period][node][
                        technology + "_existing"
                    ] = tec_data

        self.technology_data[aggregation_type] = technology_data

    def _read_network_data(self) -> None:
        """
        Reads all network data
        """
        for investment_period in self.topology["investment_periods"]:
            self.network_data[investment_period] = {}
            with open(
                self.data_path / investment_period / "Networks.json"
            ) as json_file:
                networks = json.load(json_file)

            # New networks
            for network in networks["new"]:
                netw_data = open_json(
                    network, self.data_path / investment_period / "network_data"
                )

                netw_data["name"] = network
                netw_data = Network(netw_data)
                netw_data.connection = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "network_topology"
                    / "new"
                    / network
                    / "connection.csv",
                    sep=";",
                )
                netw_data.distance = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "network_topology"
                    / "new"
                    / network
                    / "distance.csv",
                    sep=";",
                )
                netw_data.size_max_arcs = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "network_topology"
                    / "new"
                    / network
                    / "size_max_arcs.csv",
                    sep=";",
                )
                netw_data.calculate_max_size_arc()
                self.network_data[investment_period][network] = netw_data

            # Existing networks
            for network in networks["existing"]:
                netw_data = open_json(
                    network, self.data_path / investment_period / "network_data"
                )

                netw_data["name"] = network + "_existing"
                netw_data = Network(netw_data)
                netw_data.existing = 1
                netw_data.connection = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "network_topology"
                    / "new"
                    / network
                    / "connection.csv",
                    sep=";",
                )
                netw_data.distance = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "network_topology"
                    / "new"
                    / network
                    / "distance.csv",
                    sep=";",
                )
                netw_data.size_initial = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "network_topology"
                    / "new"
                    / network
                    / "size.csv",
                    sep=";",
                )
                netw_data.size_max_arcs = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "network_topology"
                    / "new"
                    / network
                    / "size_max_arcs.csv",
                    sep=";",
                )
                netw_data.calculate_max_size_arc()
                self.network_data[investment_period][network + "_existing"] = netw_data

    def _cluster_data(self):
        nr_clusters = 20
        nr_time_intervals_per_day = 24
        nr_days_full_resolution = (
            max(self.topology["time_index"]["full"])
            - min(self.topology["time_index"]["full"])
        ).days + 1

        self.topology["time_index"]["clustered"] = range(
            0, nr_clusters * nr_time_intervals_per_day
        )

        # TODO: This needs to be changed
        # TODO: Do this for each investment period
        full_resolution = self.time_series["full"].fillna(3)

        # Perform clustering
        self.time_series["clustered"], day_labels = perform_k_means(
            full_resolution, nr_clusters
        )
        # Get order of typical days
        self.k_means_specs["sequence"] = compile_sequence(
            day_labels, nr_clusters, nr_days_full_resolution, nr_time_intervals_per_day
        )
        # Match typical day to actual day
        self.k_means_specs["typical_day"] = np.repeat(
            day_labels, nr_time_intervals_per_day
        )
        # Create factors, indicating how many times an hour occurs
        self.k_means_specs["factors"] = get_day_factors(self.k_means_specs["sequence"])

        self._read_technology_data(aggregation_type="clustered")


class DataHandle_AveragedData(DataHandle):
    """
    DataHandle sub-class for handling averaged data

    This class is used to generate time series of averaged data based on a full resolution
    or clustered input data.
    """

    def __init__(self, data: DataHandle, nr_timesteps_averaged: int):
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

        if hasattr(data, "k_means_specs"):
            self.k_means_specs = data.k_means_specs

        # averaging specs
        self.averaged_specs = simplification_specs(data.topology.timesteps)

        # perform averaging for all nodal and global data
        self._average_data(data, nr_timesteps_averaged)

        # read technology data
        self._read_technology_data(data, nr_timesteps_averaged)

        # Write averaged specs
        self.averaged_specs.reduced_resolution = pd.DataFrame(
            data=np.ones(len(self.topology.timesteps)) * nr_timesteps_averaged,
            index=self.topology.timesteps,
            columns=["factor"],
        )

        self.model_information.averaged_data_specs.nr_timesteps_averaged = (
            nr_timesteps_averaged
        )

    def _average_data(
        self, data_full_resolution: DataHandle, nr_timesteps_averaged: int
    ):
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
        time_resolution = str(nr_timesteps_averaged) + "h"
        self.topology.timestep_length_h = nr_timesteps_averaged
        self.topology.timesteps = pd.date_range(
            start=start_interval, end=end_interval, freq=time_resolution
        )

        for node in node_data:
            self.node_data[node] = NodeData(self.topology)
            self.node_data[node].options = node_data[node].options
            self.node_data[node].location = node_data[node].location
            for series1 in node_data[node].data:
                self.node_data[node].data[series1] = pd.DataFrame(
                    index=self.topology.timesteps
                )
                for series2 in node_data[node].data[series1]:
                    self.node_data[node].data[series1][series2] = average_series(
                        node_data[node].data[series1][series2], nr_timesteps_averaged
                    )

        self.global_data = GlobalData(self.topology)
        for series1 in global_data.data:
            self.global_data.data[series1] = pd.DataFrame(index=self.topology.timesteps)
            for series2 in global_data.data[series1]:
                self.global_data.data[series1][series2] = average_series(
                    global_data.data[series1][series2], nr_timesteps_averaged
                )

        # Average data for clustered resolution
        if self.model_information.clustered_data == 1:
            # adjust timesteps
            end_interval = max(self.topology.timesteps_clustered)
            start_interval = min(self.topology.timesteps_clustered)
            self.topology.timesteps_clustered = range(
                start_interval, int((end_interval + 1) / nr_timesteps_averaged)
            )

            for node in node_data:
                for series1 in node_data[node].data:
                    self.node_data[node].data_clustered[series1] = pd.DataFrame(
                        self.topology.timesteps_clustered
                    )
                    for series2 in node_data[node].data[series1]:
                        self.node_data[node].data_clustered[series1][series2] = (
                            average_series(
                                node_data[node].data_clustered[series1][series2],
                                nr_timesteps_averaged,
                            )
                        )

            for series1 in global_data.data:
                self.global_data.data_clustered[series1] = pd.DataFrame(
                    self.topology.timesteps_clustered
                )
                for series2 in global_data.data[series1]:
                    self.global_data.data_clustered[series1][series2] = average_series(
                        global_data.data_clustered[series1][series2],
                        nr_timesteps_averaged,
                    )

    def _read_technology_data(
        self, data_full_resolution: DataHandle, nr_timesteps_averaged: int
    ):
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
                tec_data["name"] = technology
                self.technology_data[node][technology] = select_technology(tec_data)

                if self.technology_data[node][technology].technology_model == "RES":
                    # Fit performance based on full resolution and average capacity factor
                    self.technology_data[node][technology].fit_technology_performance(
                        data_full_resolution.node_data[node]
                    )
                    cap_factor = self.technology_data[node][
                        technology
                    ].fitted_performance.coefficients["capfactor"]
                    new_cap_factor = average_series(cap_factor, nr_timesteps_averaged)
                    self.technology_data[node][
                        technology
                    ].fitted_performance.coefficients["capfactor"] = new_cap_factor

                    lower_output_bound = np.zeros(shape=(len(new_cap_factor)))
                    upper_output_bound = new_cap_factor
                    output_bounds = np.column_stack(
                        (lower_output_bound, upper_output_bound)
                    )

                    self.technology_data[node][technology].fitted_performance.bounds[
                        "output"
                    ]["electricity"] = output_bounds
                else:
                    # Fit performance based on averaged data
                    self.technology_data[node][technology].fit_technology_performance(
                        self.node_data[node]
                    )

            # Existing technologies
            for technology in self.topology.technologies_existing[node].keys():
                tec_data = open_json(technology, load_path)
                tec_data["name"] = technology

                self.technology_data[node][technology + "_existing"] = (
                    select_technology(tec_data)
                )
                self.technology_data[node][technology + "_existing"].existing = 1
                self.technology_data[node][technology + "_existing"].size_initial = (
                    self.topology.technologies_existing[node][technology]
                )
                if (
                    self.technology_data[node][
                        technology + "_existing"
                    ].technology_model
                    == "RES"
                ):
                    # Fit performance based on full resolution and average capacity factor
                    self.technology_data[node][
                        technology + "_existing"
                    ].fit_technology_performance(data_full_resolution.node_data[node])
                    cap_factor = self.technology_data[node][
                        technology + "_existing"
                    ].fitted_performance.coefficients["capfactor"]
                    new_cap_factor = average_series(cap_factor, nr_timesteps_averaged)

                    self.technology_data[node][
                        technology + "_existing"
                    ].fitted_performance.coefficients["capfactor"] = new_cap_factor

                    lower_output_bound = np.zeros(shape=(len(new_cap_factor)))
                    upper_output_bound = new_cap_factor
                    output_bounds = np.column_stack(
                        (lower_output_bound, upper_output_bound)
                    )

                    self.technology_data[node][
                        technology + "_existing"
                    ].fitted_performance.bounds["output"]["electricity"] = output_bounds
                else:
                    # Fit performance based on averaged data
                    self.technology_data[node][
                        technology + "_existing"
                    ].fit_technology_performance(self.node_data[node])
