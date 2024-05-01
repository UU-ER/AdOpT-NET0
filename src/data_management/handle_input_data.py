import pandas as pd
import copy
import numpy as np
from pathlib import Path

from .utilities import *
from ..components.networks import *
from ..logger import logger


class DataHandle:
    """
    Data Handle for loading and performing operations on input data.

    The Data Handle class reads in the data previously specified in the respective input data folder. Pass the
    folder path when initializing the class.
    """

    def __init__(self) -> None:
        """
        Constructor

        :param str/Path base_path: path to folder
        """
        # Get logger
        self.logger = logger

        # Attributes
        self.topology = {}
        self.data_path = None
        self.time_series = {}
        self.energybalance_options = {}
        self.technology_data = {}
        self.network_data = {}
        self.node_locations = {}
        self.model_config = {}
        self.k_means_specs = {}
        self.averaged_specs = {}
        self.start_period = None
        self.end_period = None

    def read_input_data(
        self, data_path: Path | str, start_period: int = None, end_period: int = None
    ) -> None:
        """
        Reads the data from folder structure
        """
        # Read in data
        # Convert to Path
        if isinstance(data_path, str):
            data_path = Path(data_path)

        self.data_path = data_path
        self.start_period = start_period
        self.end_period = end_period

        # Check consistency
        check_input_data_consistency(data_path)

        self._read_topology()
        self._read_model_config()
        self._read_time_series()
        self._read_node_locations()
        self._read_energybalance_options()
        self._read_technology_data()
        self._read_network_data()

        # Clustering/Averaging algorithms
        if self.model_config["optimization"]["typicaldays"]["N"]["value"] != 0:
            self._cluster_data()
        if self.model_config["optimization"]["timestaging"]["value"] != 0:
            self._average_data()

    def _read_topology(self) -> None:
        """
        Reads topology from path
        """
        with open(self.data_path / "Topology.json") as json_file:
            self.topology = json.load(json_file)

        self.topology["time_index"] = {}
        time_index = pd.date_range(
            start=self.topology["start_date"],
            end=self.topology["end_date"],
            freq=self.topology["resolution"],
        )
        original_number_timesteps = len(time_index)
        self.topology["time_index"]["full"] = time_index[
            self.start_period : self.end_period
        ]
        new_number_timesteps = len(self.topology["time_index"]["full"])
        self.topology["fraction_of_year_modelled"] = (
            new_number_timesteps / original_number_timesteps
        )

        self.logger.info("Topology read successfully")

    def _read_model_config(self) -> None:
        """
        Reads model configuration
        """
        with open(self.data_path / "ConfigModel.json") as json_file:
            self.model_config = json.load(json_file)

        self.logger.info("Model Configuration read successfully")
        self.logger.info("Model Configuration used: " + json.dumps(self.model_config))

    def _read_time_series(self) -> None:
        """
        Reads all time-series data
        """

        def replace_nan_in_list(ls: list) -> list:
            """
            Replaces nan with zeros and writes warning to logger
            """
            if any(np.isnan(x) for x in ls):
                ls = [0 if np.isnan(x) else x for x in ls]
                self.logger.warning(
                    f"Found NaN values in data for investment period {investment_period}, node {node}, key1 {var}, carrier {carrier}, key2 {key}. Replaced with zeros."
                )
                return ls
            else:
                return ls

        data = {}
        for investment_period in self.topology["investment_periods"]:
            for node in self.topology["nodes"]:
                # Carbon Costs
                var = "CarbonCost"
                carrier = "global"
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
                    data[(investment_period, node, var, carrier, key)] = (
                        replace_nan_in_list(carbon_cost[key])
                    )

                # Climate Data
                var = "ClimateData"
                carrier = "global"
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
                    data[(investment_period, node, var, carrier, key)] = (
                        replace_nan_in_list(climate_data[key])
                    )

                # Carrier Data
                var = "CarrierData"
                carrier = "global"
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
                        data[(investment_period, node, var, carrier, key)] = (
                            replace_nan_in_list(carrier_data[key])
                        )

        data = pd.DataFrame(data)
        data = data.iloc[self.start_period : self.end_period]
        data.index = self.topology["time_index"]["full"]
        data.columns.set_names(
            ["InvestmentPeriod", "Node", "Key1", "Carrier", "Key2"], inplace=True
        )
        self.time_series["full"] = data

        self.logger.info("Time series read successfully")

    def _read_node_locations(self) -> None:
        """
        Reads node locations
        """
        self.node_locations = pd.read_csv(
            self.data_path / "NodeLocations.csv", index_col=0, sep=";"
        )

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
                        self.time_series[aggregation_type][investment_period][node][
                            "ClimateData"
                        ]["global"],
                        self.node_locations.loc[node, :],
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
                        self.time_series[aggregation_type][investment_period][node][
                            "ClimateData"
                        ]["global"],
                        self.node_locations.loc[node, :],
                    )
                    technology_data[investment_period][node][
                        technology + "_existing"
                    ] = tec_data

        self.technology_data[aggregation_type] = technology_data

    def _read_network_data(self, aggregation_type: str = "full") -> None:
        """
        Reads all network data
        """
        self.network_data[aggregation_type] = {}
        for investment_period in self.topology["investment_periods"]:
            self.network_data[aggregation_type][investment_period] = {}
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
                    index_col=0,
                )
                netw_data.distance = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "network_topology"
                    / "new"
                    / network
                    / "distance.csv",
                    sep=";",
                    index_col=0,
                )
                netw_data.size_max_arcs = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "network_topology"
                    / "new"
                    / network
                    / "size_max_arcs.csv",
                    sep=";",
                    index_col=0,
                )
                netw_data.calculate_max_size_arc()
                self.network_data[aggregation_type][investment_period][
                    network
                ] = netw_data

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
                    / "existing"
                    / network
                    / "connection.csv",
                    sep=";",
                    index_col=0,
                )
                netw_data.distance = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "network_topology"
                    / "existing"
                    / network
                    / "distance.csv",
                    sep=";",
                    index_col=0,
                )
                netw_data.size_initial = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "network_topology"
                    / "existing"
                    / network
                    / "size.csv",
                    sep=";",
                    index_col=0,
                )
                netw_data.size_max_arcs = pd.read_csv(
                    self.data_path
                    / investment_period
                    / "network_topology"
                    / "existing"
                    / network
                    / "size_max_arcs.csv",
                    sep=";",
                    index_col=0,
                )
                netw_data.calculate_max_size_arc()
                self.network_data[aggregation_type][investment_period][
                    network + "_existing"
                ] = netw_data

    def _cluster_data(self):
        nr_clusters = self.model_config["optimization"]["typicaldays"]["N"]["value"]
        nr_time_intervals_per_day = 24
        nr_days_full_resolution = (
            max(self.topology["time_index"]["full"])
            - min(self.topology["time_index"]["full"])
        ).days + 1

        self.topology["time_index"]["clustered"] = range(
            0, nr_clusters * nr_time_intervals_per_day
        )

        full_resolution = self.time_series["full"]

        self.k_means_specs["sequence"] = {}
        self.k_means_specs["typical_day"] = {}
        self.k_means_specs["factors"] = {}
        clustered_data = {}
        for investment_period in self.topology["investment_periods"]:
            full_res_data_matrix = compile_full_resolution_matrix(
                full_resolution.loc[:, investment_period], nr_time_intervals_per_day
            )

            # Perform clustering
            clustered_data[investment_period], day_labels = perform_k_means(
                full_res_data_matrix, nr_clusters
            )
            # Get order of typical days
            self.k_means_specs["sequence"][investment_period] = compile_sequence(
                day_labels,
                nr_clusters,
                nr_days_full_resolution,
                nr_time_intervals_per_day,
            )
            # Match typical day to actual day
            self.k_means_specs["typical_day"][investment_period] = np.repeat(
                day_labels, nr_time_intervals_per_day
            )
            # Create factors, indicating how many times an hour occurs
            self.k_means_specs["factors"] = get_day_factors(
                self.k_means_specs["sequence"][investment_period]
            )
        clustered_data = pd.concat(
            clustered_data.values(), axis=1, keys=clustered_data.keys()
        )
        clustered_data.columns.set_names(
            ["InvestmentPeriod", "Node", "Key1", "Carrier", "Key2", "Timestep"],
            inplace=True,
        )
        self.time_series["clustered"] = clustered_data

        self._read_technology_data(aggregation_type="clustered")

    def _average_data(self):
        """
        Averages all nodal and global data

        :param data_full_resolution: Data full resolution
        :param nr_timesteps_averaged: How many time-steps should be averaged?
        """

        # Adjust time index
        nr_timesteps_averaged = self.model_config["optimization"]["timestaging"][
            "value"
        ]
        time_resolution_averaged = str(nr_timesteps_averaged) + "h"
        self.topology["time_index"]["averaged"] = pd.date_range(
            start=self.topology["start_date"],
            end=self.topology["end_date"],
            freq=time_resolution_averaged,
        )

        # Averaging data
        if self.model_config["optimization"]["typicaldays"]["N"]["value"] == 0:
            full_res_data_matrix = self.time_series["full"]
            self.time_series["averaged"] = average_timeseries_data(
                full_res_data_matrix,
                nr_timesteps_averaged,
                self.topology["time_index"]["averaged"],
            )

            # read technology data
            self._read_technology_data(aggregation_type="averaged")

        else:
            clustered_data_matrix = self.time_series["clustered"]
            clustered_days = self.model_config["optimization"]["typicaldays"]["N"][
                "value"
            ]
            self.time_series["clustered_averaged"] = average_timeseries_data_clustered(
                clustered_data_matrix, nr_timesteps_averaged, clustered_days
            )

            # read technology data
            self._read_technology_data(aggregation_type="clustered_averaged")
