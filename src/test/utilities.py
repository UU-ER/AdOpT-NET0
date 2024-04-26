import json
import random
from pathlib import Path
import pandas as pd

from src.data_preprocessing import *
from src.data_management.utilities import open_json, select_technology
from src.components.technologies.technology import Technology


def select_random_list_from_list(ls: list) -> list:
    """
    Create a random list form an existing list

    :param list ls: list to use
    :return list: list with random items
    """
    num_items = random.randint(1, len(ls))
    return random.sample(ls, num_items)


def load_json(folder_path: Path) -> dict:
    """
    Loads json to a dict
    :param Path folder_path: folder path to save to
    :return dict:
    """
    with open(folder_path, "r") as json_file:
        return json.load(json_file)


def save_json(dict: dict, folder_path: Path) -> None:
    """
    Save dict to folder path as json
    :param dict dict: dict to save
    :param Path folder_path: folder path to save to
    """
    with open(folder_path, "w") as f:
        json.dump(dict, f, indent=4)


def get_topology_data(folder_path: Path) -> (list, list, list):
    """
    Gets investment periods, nodes and carriers from path
    :param Path folder_path: folder path containing topology
    :return: tuple of lists with investment_period, nodes and carriers
    """
    topology = load_json(folder_path / "Topology.json")
    investment_periods = topology["investment_periods"]
    nodes = topology["nodes"]
    carriers = topology["carriers"]
    return investment_periods, nodes, carriers


def create_basic_case_study(folder_path: Path) -> None:
    """
    Creates a basix case study with
    - two investment periods
    - two nodes
    - one carrier
    - no technologies
    - no networks
    :param Path folder_path: folder path containing topology
    """
    topology = initialize_topology_templates()
    configuration = initialize_configuration_templates()

    topology["carriers"] = ["electricity"]
    configuration["solveroptions"]["solver"]["value"] = "glpk"

    save_json(topology, folder_path / "Topology.json")
    save_json(configuration, folder_path / "ConfigModel.json")


def get_technology_data(technology: str, load_path: Path) -> Technology:
    """
    Reads in technology data.

    :param str technology: name of the technology.
    :param Path load_path: path input data folder.
    :return dict: dictionary containing the technology data.
    """
    tec_data = open_json(technology, load_path)
    tec_data["name"] = technology
    tec_data = select_technology(tec_data)

    return tec_data


def make_climate_data(start_date, end_date):
    timesteps = pd.date_range(
        start=start_date,
        end=end_date,
        freq="1h",
    )
    climate_data = pd.DataFrame(
        index=timesteps,
        columns=["ghi", "dni", "dhi", "temp_air", "rh", "TECHNOLOGYNAME_hydro_inflow"],
    )
    climate_data["ghi"] = 22
    climate_data["dni"] = 47.7
    climate_data["dhi"] = 11
    climate_data["temp_air"] = 4
    climate_data["rh"] = 10
    climate_data["ws10"] = 10

    return climate_data
