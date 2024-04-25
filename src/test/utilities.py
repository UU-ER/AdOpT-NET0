import json
import random
from pathlib import Path


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
    Creates a basix case study with one two investment periods, two nodes, no technologies
    :param Path folder_path: folder path containing topology
    """
    pass
