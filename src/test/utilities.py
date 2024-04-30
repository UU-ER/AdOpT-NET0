import json
import random
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_preprocessing import *
from src.data_management import DataHandle
from src.data_preprocessing.template_creation import (
    create_climate_data,
    create_carrier_data,
    create_carbon_cost_data,
)
from src.energyhub import EnergyHub


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


def make_climate_data(start_date: str, nr_periods: int = 1):
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
    Monkey Patch: Reads topology from template
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


def _read_energybalance_options_patch(self):
    for investment_period in self.topology["investment_periods"]:
        self.energybalance_options[investment_period] = {}
        for node in self.topology["nodes"]:
            energybalance_options = {
                carrier: {"curtailment_possible": 0}
                for carrier in self.topology["carriers"]
            }
            self.energybalance_options[investment_period][node] = energybalance_options


def _read_time_series_patch(self):
    """
    Monkey Patch: Reads time series
    """

    def replace_nan_in_list(ls: list) -> list:
        """
        Replaces nan with zeros and writes warning to logger
        """
        if any(np.isnan(x) for x in ls):
            ls = [0 if np.isnan(x) else x for x in ls]
            return ls
        else:
            return ls

    data = {}
    for investment_period in self.topology["investment_periods"]:
        for node in self.topology["nodes"]:
            # Carbon Costs
            var = "CarbonCost"
            carrier = "global"
            carbon_cost = create_carbon_cost_data(
                self.topology["time_index"]["full"]
            ).to_dict(orient="list")
            for key in carbon_cost.keys():
                data[(investment_period, node, var, carrier, key)] = (
                    replace_nan_in_list(carbon_cost[key])
                )

            # Carrier Data
            var = "CarrierData"
            for carrier in self.topology["carriers"]:
                carrier_data = create_carrier_data(
                    self.topology["time_index"]["full"]
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


def _read_technology_data_patch(self):
    technology_data = {}
    for investment_period in self.topology["investment_periods"]:
        technology_data[investment_period] = {}
        for node in self.topology["nodes"]:
            technology_data[investment_period][node] = {}

    self.technology_data["full"] = technology_data


def _read_network_data_data_patch(self):
    self.network_data["full"] = {}
    for investment_period in self.topology["investment_periods"]:
        self.network_data["full"][investment_period] = {}


def read_input_data_patch(self):

    self.model_config = initialize_configuration_templates()
    self._read_topology()
    self._read_time_series()
    self._read_energybalance_options()
    self._read_technology_data()
    self._read_network_data()


def create_patched_datahandle(nr_timesteps):
    """
    Creates a patched datahandle with:
    - nr_timesteps specified
    - two nodes
    - two investment periods
    - no technologies
    - no networks
    """

    # Create DataHandle and monkey patch it
    dh = DataHandle()

    dh._read_topology = read_topology_patch.__get__(dh)
    dh._read_time_series = _read_time_series_patch.__get__(dh)
    dh._read_energybalance_options = _read_energybalance_options_patch.__get__(dh)
    dh._read_technology_data = _read_technology_data_patch.__get__(dh)
    dh._read_network_data = _read_network_data_data_patch.__get__(dh)
    dh.read_input_data = read_input_data_patch.__get__(dh)

    dh.start_period = 0
    dh.end_period = dh.start_period + nr_timesteps
    dh.read_input_data()

    return dh
