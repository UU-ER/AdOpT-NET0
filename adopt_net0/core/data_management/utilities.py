from pathlib import Path
import pandas as pd
import pvlib
import os
import json

import logging

log = logging.getLogger(__name__)


def calculate_dni(data: pd.DataFrame, lon: float, lat: float) -> pd.Series:
    """
    Calculate direct normal irradiance from ghi and dhi. The function assumes that
    the ghi and dhi are given as an average value for the timestep and dni is
    calculated using the position of the sun in the middle of the timestep.

    :param pd.DataFrame data: climate data with columns ghi and dhi
    :param float lon: longitude
    :param float lat: latitude
    :return data: climate data including dni
    :rtype: pd.Series
    """
    timesteps = pd.to_datetime(data.index)
    timestep_length = pd.to_datetime(data.index[1]) - pd.to_datetime(data.index[0])
    timesteps = timesteps + (timestep_length / 2)

    zenith = pvlib.solarposition.get_solarposition(timesteps, lat, lon)
    data["dni"] = pvlib.irradiance.dni(
        data["ghi"].to_numpy(), data["dhi"].to_numpy(), zenith["zenith"].to_numpy()
    )
    data["dni"] = data["dni"].fillna(0)
    data["dni"] = data["dni"].where(data["dni"] > 0, 0)

    return data["dni"]


# def network_factory(netw_data: dict):
#     """
#     Returns the correct subclass for a network
#
#     :param dict netw_data: dictonary derived from the network json files
#     :return: Network Class
#     """
#     # TODO: plugin architecture for networks
#     if "netw_type" in netw_data:
#         if netw_data["netw_type"] == "fluid":
#             return Fluid(netw_data)
#         elif netw_data["netw_type"] == "electricity":
#             return Electricity(netw_data)
#     else:
#         return Simple(netw_data)


# def technology_factory(tec_data: dict):
#     """
#     Returns the correct subclass for a technology
#
#     :param dict tec_data: dictonary derived from the technology json files
#     :return: Technology Class
#     """
#
#
#
# def create_technology_class(tec_name: str, load_path: Path):
#     """
#     Loads the technology data from load_path and preprocesses it.
#
#     :param str tec_name: technology name
#     :param Path load_path: load path
#     :param pd.DataFrame technology_data: Climate Data
#     :param dict location: Dictonary with node location
#     :return: Technology Class
#     """
#     tec_data = open_json(tec_name, load_path)
#     tec_data["name"] = tec_name
#     tec_data = technology_factory(tec_data)
#
#     return tec_data
#
#
# def create_network_class(netw_name: str, load_path: Path):
#     """
#     Loads the network data from load_path and preprocesses it.
#
#     :param str netw_name: network name
#     :param Path load_path: load path
#     #:param dict location: Dictonary with node location
#
#     :return: Network Class
#     """
#     netw_data = open_json(netw_name, load_path)
#     netw_data["name"] = netw_name
#     netw_data = network_factory(netw_data)
#
#     return netw_data
#
# Todo: plugin compressor
# def create_compressor_class(connection_info: dict, carrier: str, load_path: Path):
#     """
#     Loads the compressor data from load_path and preprocesses it.
#
#     :param dict connection_info: information about the connection
#     :param str carrier: compressed carrier
#     :param Path load_path: load path
#
#     :return: Compressor Class
#     """
#     comp_data = open_json(carrier, load_path)
#
#     comp_data["connection_info"] = connection_info
#     comp_data["name"] = (
#         f"{carrier}_Compressor_{comp_data['connection_info']['components'][0]}_{comp_data['connection_info']['components'][1]}"
#     )
#
#     if (
#         comp_data["connection_info"]["existing"][0] == 1
#         and comp_data["connection_info"]["existing"][1] == 1
#     ):
#         comp_data["name"] = comp_data["name"] + "_existing"
#
#     comp_data = Compressor(comp_data)
#
#     return comp_data


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

# Todo: plugin compressor
# def get_pressure_info(component, carrier: str, direction: str) -> dict:
#     """
#     Obtains pressure-related information for a given component, carrier, and flow direction (input/output).
#
#     :param component: the component from which to extract pressure data
#     :param str carrier: the energy carrier for which pressure data is requested
#     :param str direction: either 'Input' or 'Output', specifying whether to retrieve inlet or outlet pressure
#
#     :return dict: A dictionary containing:
#             - "name": name of the component.
#             - "pressure": the inlet or outlet pressure associated with the specified carrier and component.
#             - "type": the type of the component ('Technology' or 'Network').
#             - "existing": 1 if the component is existing; 0 otherwise.
#     """
#     pressure_data = component.performance_data["pressure"]
#     component_name = component.name
#     pressure = ()
#     if direction == "Input":
#         pressure = pressure_data[carrier]["inlet"]
#     elif direction == "Output":
#         pressure = pressure_data[carrier]["outlet"]
#     if isinstance(component, Technology):
#         type = "Technology"
#     elif isinstance(component, Network):
#         type = "Network"
#     return {
#         "name": component_name,
#         "pressure": pressure,
#         "type": type,
#         "existing": component.existing,
#     }
#
#
# def collect_possible_connections_at_node(pressure_data_at_node: dict):
#     """
#     Generates all possible compression connections between output and input components at a given node.
#
#     :param dict pressure_data_at_node: contains all components that can be inputs or outputs for compression
#
#     :return list: containing all possible connection between input and outputs for each node, with necessary information
#     """
#     connection_data_at_node = []
#     for output_i in pressure_data_at_node["outputs"]:
#         for input_i in pressure_data_at_node["inputs"]:
#             connection_data_at_node.append(
#                 {
#                     "components": (output_i["name"], input_i["name"]),
#                     "pressure": (output_i["pressure"], input_i["pressure"]),
#                     "type": (output_i["type"], input_i["type"]),
#                     "existing": (output_i["existing"], input_i["existing"]),
#                 }
#             )
#
#     return connection_data_at_node


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

    if config["performance"]["pressure"]["pressure_on"]["value"] == 1:
        target_carrier = config["performance"]["pressure"]["pressure_carriers"]["value"]
        for investment_period in topology["investment_periods"]:
            for compressor in target_carrier:
                # Check compressor_data
                check_compressor_data_path = (
                    path / investment_period / "compressor_data"
                )
                check_path_existance(
                    check_compressor_data_path / (compressor + ".json"),
                    f"A json file for {compressor} is missing in {check_compressor_data_path}",
                )

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