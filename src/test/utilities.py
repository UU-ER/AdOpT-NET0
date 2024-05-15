import json
import random
from pathlib import Path
import pandas as pd
import numpy as np
from pyomo.core import ConcreteModel, Objective, minimize
from pyomo.opt import TerminationCondition, SolverFactory

from src.data_preprocessing import (
    initialize_configuration_templates,
    initialize_topology_templates,
)
from src.data_management import DataHandle
from src.data_preprocessing.template_creation import (
    create_carrier_data,
    create_carbon_cost_data,
)


def select_random_list_from_list(ls: list) -> list:
    """
    Create a random list form an existing list

    :param list ls: list to use
    :return: list with random items
    :rtype: list
    """
    num_items = random.randint(1, len(ls))
    return random.sample(ls, num_items)


def load_json(folder_path: Path) -> dict:
    """
    Loads json to a dict

    :param Path folder_path: folder path to save to
    :return: dict read from folder_path
    :rtype: dict
    """
    with open(folder_path, "r") as json_file:
        return json.load(json_file)


def save_json(d: dict, folder_path: Path):
    """
    Save dict to folder path as json

    :param dict d: dict to save
    :param Path folder_path: folder path to save to
    """
    with open(folder_path, "w") as f:
        json.dump(d, f, indent=4)


def get_topology_data(folder_path: Path) -> (list, list, list):
    """
    Gets investment periods, nodes and carriers from path

    :param Path folder_path: folder path containing topology
    :return: tuple of lists with investment_period, nodes and carriers
    :rtype: tuple
    """
    topology = load_json(folder_path / "Topology.json")
    investment_periods = topology["investment_periods"]
    nodes = topology["nodes"]
    carriers = topology["carriers"]
    return investment_periods, nodes, carriers


def create_basic_case_study(folder_path: Path):
    """
    Creates a basix case study and saves it to folder_path with
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


def make_climate_data(start_date: str, nr_periods: int = 1) -> pd.DataFrame:
    """
    Makes climate data with random values

    :param start_date: str in datetime format e.g. 2022-10-03 12:00
    :param nr_periods: how many periods to use (frequency is always 1h)
    :return: dataframe with mock climate data
    :rtype: pd.DataFrame
    """
    timesteps = pd.date_range(
        start=start_date,
        periods=nr_periods,
        freq="1h",
    )
    climate_data = pd.DataFrame(
        index=timesteps,
        columns=["ghi", "dni", "dhi", "temp_air", "rh", "TestTec_Hydro_Open_inflow"],
    )
    climate_data["ghi"] = 152
    climate_data["dni"] = 162.9
    climate_data["dhi"] = 112
    climate_data["temp_air"] = 4
    climate_data["rh"] = 81
    climate_data["ws10"] = 6.17
    climate_data["TestTec_Hydro_Open_inflow"] = 1

    return climate_data


def read_topology_patch(self):
    """
    Monkey patch topology reading
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


def make_data_for_testing(nr_timesteps: int) -> dict:
    """
    Makes a dict with config and topology used for testing

    :param int nr_timesteps: Number of time steps
    :return: dict with config and topology
    :rtype: dict
    """
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
    """
    Monkey patch energy balance options
    """
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
    """
    Monkey patch read technology data
    """
    technology_data = {}
    for investment_period in self.topology["investment_periods"]:
        technology_data[investment_period] = {}
        for node in self.topology["nodes"]:
            technology_data[investment_period][node] = {}

    self.technology_data = technology_data


def _read_network_data_data_patch(self):
    """
    Monkey patch read network data
    """
    self.network_data = {}
    for investment_period in self.topology["investment_periods"]:
        self.network_data[investment_period] = {}


def read_input_data_patch(self):
    """
    Monkey patch read data
    """
    self.model_config = initialize_configuration_templates()
    self._read_topology()
    self._read_time_series()
    self._read_energybalance_options()
    self._read_technology_data()
    self._read_network_data()


def make_data_handle(nr_timesteps: int, topology=None):
    """
    Creates a patched datahandle with:
    - nr_timesteps specified
    - two nodes
    - two investment periods
    - no technologies
    - no networks

    :param int nr_timesteps: number of time steps used
    :param topology: topology
    :return: mock DataHandle for testing
    """

    # Create DataHandle and monkey patch it
    dh = DataHandle()

    if topology is None:
        dh.topology = initialize_topology_templates()
    else:
        dh.topology = topology

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


def run_model(model, solver: str, objective: str = "capex_tot"):
    """
    Runs a model and returns termination condition

    :param model: pyomo model
    :param str solver: solver to used
    :param str objective: objective to optimize
    :return: termination condition for respective model
    """
    if objective == "capex_tot":
        model.obj = Objective(expr=model.var_capex_tot, sense=minimize)
    elif objective == "capex":
        model.obj = Objective(expr=model.var_capex, sense=minimize)
    elif objective == "emissions":
        model.obj = Objective(
            expr=sum(model.var_tec_emissions_pos[t] for t in model.set_t),
            sense=minimize,
        )

    solver = SolverFactory(solver)
    solution = solver.solve(model)

    return solution.solver.termination_condition
