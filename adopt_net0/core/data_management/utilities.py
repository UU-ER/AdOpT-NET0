from pathlib import Path
import pandas as pd
import os
import json

import logging

log = logging.getLogger(__name__)


def open_json(component: str, load_path: Path) -> dict:
    """
    Loops through load_path and subdirectories and returns json with name tec + ".json"

    :param str component: name of component to read json for
    :param Path load_path: directory path to loop through all subdirectories and search for component + ".json"
    :return: Dictionary containing the json data
    :rtype: dict
    """
    # Read in JSON files
    for path, subdirs, files in os.walk(load_path):
        if "data" in locals():
            break
        else:
            for name in files:
                if (component + ".json") == name:
                    filepath = os.path.join(path, name)
                    with open(filepath) as json_file:
                        data = json.load(json_file)
                    break

    # Assign name
    if "data" in locals():
        data["name"] = component
    else:
        raise Exception("There is no json data file for component " + component)

    return data


def check_input_data_consistency(path: Path):
    """
    Checks if the topology is consistent with the input data.

    Checks for:
    - is there a folder for each investment period?
    - is there a network file for each network defined?
    - are there all required files for all networks in the directory?
    - are node directories there?
    - is TechnologyData, CarbonCost for each node there?
    - is Technologies.json there?
    - is there a json file for all technologies?
    - is there a carrier file for each defined carrier?

    :param Path path: path to check for consistency
    """

    def check_path_existance(path: Path, error_message: str):
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

    # Read config
    with open(path / "ConfigModel.json") as json_file:
        config = json.load(json_file)

    # Check that averaging and k-means is not used at same time
    if (config["optimization"]["typicaldays"]["N"]["value"] != 0) and (
        config["optimization"]["timestaging"]["value"] != 0
    ):
        raise Exception(
            "Using time step averaging and k-means clustering at the same"
            " time is not allowed"
        )

    log_msg = "Input data folder has been checked successfully - no errors occurred."
    print(log_msg)
    log.info(log_msg)


def merge_node_names_and_locations(node_names: list, node_locations: pd.DataFrame) -> dict:
    """
    Merges node names and locations into one dict

    :param list node_names:
    :param pd.DataFrame node_locations:
    :return dict: merged dict
    """
    merged = {}
    for node in node_names:
        if node not in node_locations.index:
            raise ValueError(f"Node {node} defined in Topology.json but not found in NodeLocations.csv")
        else:
            merged[node] = node_locations.loc[node, :].to_dict()
    return merged


def get_temporal_information(start_date: str, end_date: str, resolution: str, start_period: int, end_period: int, aggregation: str) -> dict:
    """
    Collects temporal information

    Makes:
    - time index as pd.DatetimeIndex
    - original number of timesteps as integer
    - new number of timesteps as integer
    - fraction of year modelled as float
    - resolution in hours as float
    - hours per day as integer

    :param str start_date:
    :param str end_date:
    :param str resolution:
    :param int start_period:
    :param int end_period:
    :param str aggregation: Type of temporal aggregation
    :return dict: dict with temporal information
    """
    time_index = pd.date_range(
        start=start_date,
        end=end_date,
        freq=resolution,
    )
    temporal_information = {
        "time_index": time_index[start_period:end_period],
        "original_number_timesteps": len(time_index),
        "new_number_timesteps": len(time_index[start_period:end_period]),
        "fraction_of_year_modelled": len(time_index[start_period:end_period]) / len(time_index),
        "resolution_in_h": pd.Timedelta(time_index.freq).seconds / 3600,
        "hours_per_day": int(24 / pd.Timedelta(time_index.freq).seconds / 3600),
        "aggregation": aggregation
    }
    return temporal_information