import numpy as np
import pandas as pd
from pathlib import Path
import tsam.timeseriesaggregation as tsam
import copy

from .utilities import *
from ..components.networks import *
from ..logger import log_event


class DataHandle:
    """
    Data Handle for loading and performing operations on input data.

    The Data Handle class reads in the data previously specified in the respective input data folder.
    It can also perform input data manipulation for two solving algorithms:

    - clustering algorithm
    - averaging algorithm

    :param dict topology: Container for the topology
    :param Path data_path: Container data_path
    :param dict time_series: Container for all time series
    :param dict energybalance_options: Container for energy balance options
    :param dict technology_data: Container for technology data
    :param dict network_data: Container for network data
    :param pd.DataFrame node_locations: Container for node locations
    :param dict model_config: Container for the model configuration
    :param dict k_means_specs: Container for k-means clustering algorithm specifications
    :param dict averaged_specs: Container for averaging algorithm specifications
    :param int, None start_period: starting period to use, if None, the first available period is used
    :param int, None end_period: end period to use, if None, the last available period is used
    """

    def __init__(self):
        """
        Constructor
        """
        self.topology = {}
        self.data_path = Path()
        self.time_series = {}
        self.energybalance_options = {}
        self.technology_data = {}
        self.network_data = {}
        self.node_locations = pd.DataFrame()
        self.model_config = {}
        self.k_means_specs = {}
        self.averaged_specs = {}
        self.start_period = None
        self.end_period = None

    def set_settings(
        self, data_path: Path, start_period: int = None, end_period: int = None
    ):
        """
        Overarching function to read the data from folder structure contained in data_path

        Checks the consistency of the provided folder structure and reads all required data form it, in case the
        input data check succeeds. In case used, it also clusters/averages the data accordingly.

        The following items are read:

        - topology
        - model configuration
        - time series for all periods and nodes
        - node locations
        - energy balance options
        - technology data
        - network data

        :param Path data_path: Path to read input data from
        :param int | None start_period: Starting period of model if None, the first available period is used
        :param int | None end_period: End period of model if None, the last available period is used
        """
        # Convert to Path
        if isinstance(data_path, str):
            data_path = Path(data_path)

        self.data_path = data_path
        self.start_period = start_period
        self.end_period = end_period

        # Check consistency
        check_input_data_consistency(data_path)

    def read_data(self):
        """
        Reads all data from folder
        """
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

    def _read_topology(self):
        """
        Reads topology
        """
        # Open json
        with open(self.data_path / "Topology.json") as json_file:
            self.topology = json.load(json_file)

        # Process timesteps
        self.topology["time_index"] = {}
        time_index = pd.date_range(
            start=self.topology["start_date"],
            end=self.topology["end_date"],
            freq=self.topology["resolution"],
        )

        # Calculate fraction of year modelled
        original_number_timesteps = len(time_index)
        self.topology["time_index"]["full"] = time_index[
            self.start_period : self.end_period
        ]
        new_number_timesteps = len(self.topology["time_index"]["full"])
        self.topology["fraction_of_year_modelled"] = (
            new_number_timesteps / original_number_timesteps
        )

        # Resolution in hours
        self.topology["resolution_in_h"] = {}
        self.topology["resolution_in_h"]["full"] = (
            pd.Timedelta(self.topology["time_index"]["full"].freq).seconds / 3600
        )

        # Hours per day
        self.topology["hours_per_day"] = {}
        self.topology["hours_per_day"]["full"] = int(
            24 / self.topology["resolution_in_h"]["full"]
        )

        # Log success
        log_event("Topology read successfully")

    def _read_model_config(self):
        """
        Reads model configuration
        """
        # Open json
        with open(self.data_path / "ConfigModel.json") as json_file:
            self.model_config = json.load(json_file)

        # Log success
        log_event("Model Configuration read successfully")
        log_event(
            "Model Configuration used: " + json.dumps(self.model_config), print_it=0
        )

    def _read_time_series(self):
        """
        Reads all time-series data and shortens time series accordingly
        """

        def replace_nan_in_list(ls: list) -> list:
            """
            Replaces nan with zeros and writes warning to logger

            :param list ls: List
            :return list: returns list with nan replaces by zero
            :rtype: list
            """
            if any(np.isnan(x) for x in ls):
                ls = [0 if np.isnan(x) else x for x in ls]
                log_event(
                    f"Found NaN values in data for investment period {investment_period}, node {node}, key1 {var}, carrier {carrier}, key2 {key}. Replaced with zeros."
                )
                return ls
            else:
                return ls

        # Initialize data dict
        data = {}

        # Loop through all investment_periods and nodes
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

        # Post-process data dict to dataframe and shorten
        data = pd.DataFrame(data)
        data = data.iloc[self.start_period : self.end_period]
        data.index = self.topology["time_index"]["full"]
        data.columns.set_names(
            ["InvestmentPeriod", "Node", "Key1", "Carrier", "Key2"], inplace=True
        )
        self.time_series["full"] = data

        # Log success
        log_event("Time series read successfully")

    def _read_node_locations(self):
        """
        Reads node locations
        """
        self.node_locations = pd.read_csv(
            (self.data_path / "NodeLocations.csv"), index_col=0, sep=";"
        )

        # Log success
        log_event("Node Locations read successfully")

    def _read_energybalance_options(self):
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

        log_event("Energy balance options read successfully")

    def _read_technology_data(self):
        """
        Reads all technology data and fits it

        :param str aggregation_type: specifies the aggregation type and thus the dict key to write the data to
        """
        aggregation_type = "full"
        # Initialize technology_data dict
        technology_data = {}

        # Loop through all investment_periods and nodes
        for investment_period in self.topology["investment_periods"]:
            technology_data[investment_period] = {}
            for node in self.topology["nodes"]:
                technology_data[investment_period][node] = {}

                # Get technologies at node
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
                    tec_data = read_tec_data(
                        technology,
                        self.data_path
                        / investment_period
                        / "node_data"
                        / node
                        / "technology_data",
                    )
                    tec_data.fit_technology_performance(
                        self.time_series[aggregation_type][investment_period][node][
                            "ClimateData"
                        ]["global"],
                        self.node_locations.loc[node, :],
                    )
                    technology_data[investment_period][node][technology] = tec_data

                # Existing technologies
                for technology in technologies_at_node["existing"]:
                    tec_data = read_tec_data(
                        technology,
                        self.data_path
                        / investment_period
                        / "node_data"
                        / node
                        / "technology_data",
                    )
                    tec_data.existing = 1
                    tec_data.parameters.size_initial = technologies_at_node["existing"][
                        technology
                    ]
                    tec_data.fit_technology_performance(
                        self.time_series[aggregation_type][investment_period][node][
                            "ClimateData"
                        ]["global"],
                        self.node_locations.loc[node, :],
                    )
                    technology_data[investment_period][node][
                        technology + "_existing"
                    ] = tec_data

        self.technology_data = technology_data

        log_event("Technology data read successfully")

    def _read_network_data(self):
        """
        Reads all network data
        """

        # Loop through all investment_periods and nodes
        for investment_period in self.topology["investment_periods"]:
            self.network_data[investment_period] = {}

            # Get all networks in period
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
                netw_data.fit_network_performance()
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
                netw_data.fit_network_performance()

                self.network_data[investment_period][network + "_existing"] = netw_data

        log_event("Network data read successfully")

    def _collect_full_res_data(self, investment_period: str) -> pd.DataFrame:
        """
        Collects data from time_series and technology performances and writes it to a
        single dataframe

        time_series (demand, import, export, carbon prices,...) are stored in a
        different location then time dependent technology performances. to aggregate
        all time series, they need to be merged into a single dataframe.

        :param str investment_period: investment period to collect data for
        :return: single data frame with all time dependent data
        """
        time_series = self.time_series["full"].loc[:, investment_period]
        time_series = pd.concat(
            {"time_series": time_series}, names=["type_series"], axis=1
        )

        # Get time dependent technology parameters
        tec_series = {}
        for node in self.technology_data[investment_period]:
            for tec in self.technology_data[investment_period][node]:
                tec_data = self.technology_data[investment_period][node][tec]
                c_td = tec_data.coeff.time_dependent_full
                for series in c_td:
                    if c_td[series].ndim > 1:
                        count = 0
                        for c in c_td[series].T:
                            tec_series[(node, tec, series, count)] = c
                            count += 1
                    else:
                        tec_series[(node, tec, series, "")] = c_td[series]

        # Make sure dataframe is correctly formatted
        if tec_series:
            tec_series = pd.DataFrame(tec_series)
            tec_series.columns.set_names(
                ["Node", "Key1", "Carrier", "Key2"], inplace=True
            )
        else:
            tec_series = pd.DataFrame(columns=["Node", "Key1", "Carrier", "Key2"])

        tec_series = pd.concat(
            {"tec_series": tec_series}, names=["type_series"], axis=1
        )
        tec_series.index = time_series.index

        # full matrix
        return pd.concat([time_series, tec_series], axis=1)

    def _write_aggregated_data_to_technologies(
        self, investment_period: str, tec_series: pd.DataFrame, aggregation_type: str
    ):
        """
        Writes aggregated technology performances to technology classes.

        After time aggregation this function writes the aggregated technology data
        back to the time dependent coefficients of the technologies.

        :param str investment_period: investment period to use
        :param pd.DataFrame tec_series: aggregated technology series
        """
        for node in self.technology_data[investment_period]:
            for tec in self.technology_data[investment_period][node]:
                tec_data = self.technology_data[investment_period][node][tec]
                if tec_data.coeff.time_dependent_full:
                    time_dependent_coeff = tec_series[node][tec]
                    c_td = {}
                    for series in time_dependent_coeff.columns.get_level_values(0):
                        c_td[series] = time_dependent_coeff[series].values
                    if aggregation_type == "clustered":
                        tec_data.coeff.time_dependent_clustered = c_td
                    elif aggregation_type == "averaged":
                        tec_data.coeff.time_dependent_averaged = c_td

    def _cluster_data(self):
        """
        Cluster full resolution input data

        Uses the package tsam to cluster all time-dependent input data (time series
        and time dependent technology performance).
        """
        nr_clusters = self.model_config["optimization"]["typicaldays"]["N"]["value"]
        hours_per_day = self.topology["hours_per_day"]["full"]

        self.topology["time_index"]["clustered"] = range(0, nr_clusters * hours_per_day)

        clustered_resolution = {}
        for investment_period in self.topology["investment_periods"]:
            self.k_means_specs[investment_period] = {}
            self.k_means_specs[investment_period]["sequence"] = []
            self.k_means_specs[investment_period]["factors"] = []

            full_res_data_matrix = self._collect_full_res_data(investment_period)

            # Cluster to typical days
            aggregation = tsam.TimeSeriesAggregation(
                full_res_data_matrix,
                noTypicalPeriods=nr_clusters,
                hoursPerPeriod=hours_per_day,
                noSegments=hours_per_day,
                clusterMethod="k_means",
            )

            typPeriods = aggregation.createTypicalPeriods()

            # Determine help variables
            cluster_order = aggregation._clusterOrder
            cluster_no_occ = aggregation._clusterPeriodNoOccur
            clustered_index = typPeriods.index
            clustered_index = clustered_index.set_names(["Day", "Hour"])
            clustered_index = clustered_index.to_frame().reset_index(drop=True)
            clustered_index = clustered_index["Day"].reset_index()
            clustered_index["index"] = clustered_index["index"] + 1

            # Determine Sequence
            for d in cluster_order:
                self.k_means_specs[investment_period]["sequence"].extend(
                    (clustered_index[clustered_index["Day"] == d]["index"].to_list())
                )

            # Determine Factors (how many times does a clustered hour occur)
            self.k_means_specs[investment_period]["factors"] = (
                clustered_index["Day"].map(cluster_no_occ).to_list()
            )

            # Write time series
            typPeriods = typPeriods.reset_index()
            clustered_resolution[investment_period] = typPeriods["time_series"]

            # Write technology performance
            self._write_aggregated_data_to_technologies(
                investment_period, typPeriods["tec_series"], "clustered"
            )

        self.time_series["clustered"] = pd.concat(
            clustered_resolution, names=["InvestmentPeriod"], axis=1
        )

        log_event("Clustered data successfully")

    def _average_data(self):
        """
        Averages full resolution input data

        Uses the package tsam to average all time-dependent input data (time series
        and time dependent technology performance).
        """
        nr_timesteps_averaged = self.model_config["optimization"]["timestaging"][
            "value"
        ]
        nr_timesteps_full = len(self.topology["time_index"]["full"])
        resolution_full = self.topology["resolution_in_h"]["full"]
        hours_per_day = self.topology["hours_per_day"]["full"]

        self.topology["time_index"]["averaged"] = range(
            0, int(nr_timesteps_full / nr_timesteps_averaged)
        )

        averaged_resolution = {}
        for investment_period in self.topology["investment_periods"]:
            self.averaged_specs[investment_period] = {}
            self.averaged_specs[investment_period][
                "nr_timesteps_averaged"
            ] = nr_timesteps_averaged

            full_res_data_matrix = self._collect_full_res_data(investment_period)

            # Cluster to typical days
            aggregation = tsam.TimeSeriesAggregation(
                full_res_data_matrix,
                noTypicalPeriods=int(nr_timesteps_full / nr_timesteps_averaged),
                hoursPerPeriod=1,
                noSegments=1,
                resolution=resolution_full,
                clusterMethod="averaging",
            )

            typPeriods = aggregation.createTypicalPeriods()

            typPeriods.index = self.topology["time_index"]["averaged"]
            averaged_resolution[investment_period] = typPeriods["time_series"]

            # Write technology performance
            self._write_aggregated_data_to_technologies(
                investment_period, typPeriods["tec_series"], "averaged"
            )

        self.time_series["averaged"] = pd.concat(
            averaged_resolution, names=["InvestmentPeriod"], axis=1
        )

        log_event("Averaged data successfully")
