import json
import random
from pathlib import Path
import pandas as pd

from src.data_preprocessing import *
from src.data_management import DataHandle


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


def make_climate_data(start_date: str, nr_periods: int):
    timesteps = pd.date_range(
        start=start_date,
        periods=nr_periods,
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


def read_topology_patch(self):
    """
    Reads topology from template
    """
    self.topology = initialize_topology_templates()

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


def make_data_for_technology_testing(nr_timesteps):

    # Create DataHandle and monkey patch it
    dh = DataHandle()
    dh.start_period = 0
    dh.end_period = dh.start_period + nr_timesteps
    dh._read_topology = read_topology_patch.__get__(dh, DataHandle)
    dh._read_topology()

    data = {}
    data["topology"] = dh.topology
    data["config"] = initialize_configuration_templates()

    return data
