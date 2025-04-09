from pathlib import Path
import pandas as pd
import pvlib
import os
import json

from ..components.networks import *
from ..components.technologies import *

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


def network_factory(netw_data: dict):
    """
    Returns the correct subclass for a network

    :param dict netw_data: dictonary derived from the network json files
    :return: Network Class
    """
    # Generic netw
    if netw_data["network_type"] == "fluid":
        return Fluid(netw_data)
    elif netw_data["network_type"] == "electricity":
        return Electricity(netw_data)
    elif netw_data["network_type"] == "simple":
        return Simple(netw_data)


def technology_factory(tec_data: dict):
    """
    Returns the correct subclass for a technology

    :param dict tec_data: dictonary derived from the technology json files
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
    elif tec_data["tec_type"] == "SINK":
        return Sink(tec_data)
    # Specific tecs
    elif tec_data["tec_type"] == "DAC_Adsorption":
        return DacAdsorption(tec_data)
    elif tec_data["tec_type"].startswith("GasTurbine"):
        return GasTurbine(tec_data)
    elif tec_data["tec_type"].startswith("HeatPump"):
        return HeatPump(tec_data)
    elif tec_data["tec_type"] == "HydroOpen":
        return HydroOpen(tec_data)
    elif tec_data["tec_type"] == "CCPP":
        return CCPP(tec_data)


def create_technology_class(tec_name: str, load_path: Path):
    """
    Loads the technology data from load_path and preprocesses it.

    :param str tec_name: technology name
    :param Path load_path: load path
    :param pd.DataFrame climate_data: Climate Data
    :param dict location: Dictonary with node location
    :return: Technology Class
    """
    tec_data = open_json(tec_name, load_path)
    tec_data["name"] = tec_name
    tec_data = technology_factory(tec_data)

    # CCS
    if tec_data.component_options.ccs_possible:
        tec_data.ccs_data = open_json(tec_data.component_options.ccs_type, load_path)
    return tec_data


def create_network_class(netw_name: str, load_path: Path):
    """
    Loads the network data from load_path and preprocesses it.

    :param str netw_name: network name
    :param Path load_path: load path
    #:param dict location: Dictonary with node location

    :return: Network Class
    """
    netw_data = open_json(netw_name, load_path)
    netw_data["name"] = netw_name
    netw_data = network_factory(netw_data)

    return netw_data


def open_json(tec: str, load_path: Path) -> dict:
    """
    Loops through load_path and subdirectories and returns json with name tec + ".json"

    :param str tec: name of technology to read json for
    :param Path load_path: directory path to loop through all subdirectories and search for tec + ".json"
    :return: Dictionary containing the json data
    :rtype: dict
    """
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


def add_tech_to_list(tech, carrier, str):
    if str == "Input":
        pressure = tech["Performance"]["pressure"][carrier]["inlet"]
    elif str == "Output":
        pressure = tech["Performance"]["pressure"][carrier]["outlet"]
    return tech, pressure


def add_netw_to_list(netw, carrier, str):
    if str == "Input":
        pressure = netw["Performance"]["pressure"][carrier]["inlet"]
    elif str == "Output":
        pressure = netw["Performance"]["pressure"][carrier]["outlet"]
    return netw, pressure


def check_input_data_consistency(path: Path):
    """
    Checks if the topology is consistent with the input data.

    Checks for:
    - is there a folder for each investment period?
    - is there a network file for each network defined?
    - are there all required files for all networks in the directory?
    - are node directories there?
    - is ClimateData, CarbonCost for each node there?
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
    log.info(log_msg)
