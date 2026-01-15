import json
import random
from pathlib import Path
import pandas as pd
from pyomo import environ as pyo
from pyomo.core import Objective, minimize, ConcreteModel, Set, Constraint
from pyomo.opt import SolverFactory
import types

from adopt_net0 import copy_technology_data
from adopt_net0.core.modelhub import ModelHub
from adopt_net0.core.data_preprocessing.template_creation import (
    initialize_configuration_templates,
    initialize_topology_templates,
)
from adopt_net0.core.data_management.utilities import (
    get_temporal_information
)
from adopt_net0.core.data_preprocessing import (
    create_carrier_data,
    create_carbon_cost_data,
)
from adopt_net0.core import create_empty_network_matrix
from adopt_net0.core.components.utilities import perform_disjunct_relaxation


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


def get_topology_data(folder_path: Path) -> (dict[str], dict[str], dict[str]):
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


def make_testing_modelhub(nr_timesteps, nr_nodes: int = 1) -> ModelHub:
    """
    Makes a model hub for testing
    """
    modelhub = ModelHub()

    #Monkey patch read_data
    modelhub.read_data = types.MethodType(read_data_patch, modelhub)
    modelhub.read_data(nr_timesteps, nr_nodes)
    modelhub.aggregate_data()

    return modelhub

def make_testing_time_series(modelhub: ModelHub) -> pd.DataFrame:
    """
    Makes testing time series for a model hub

    :param ModelHub modelhub: model hub
    :return: time series dataframe
    :rtype: pd.DataFrame
    """
    data = {}

    carrier_data = create_carrier_data(modelhub.data["topology"]["temporal_information"]["time_index"]).fillna(0).to_dict()
    carbon_cost = create_carbon_cost_data(modelhub.data["topology"]["temporal_information"]["time_index"]).fillna(0).to_dict()

    for investment_period in modelhub.data["topology"]["investment_periods"]:
        for node in modelhub.data["topology"]["nodes"]:
            var = "CarbonCost"
            carrier = "global"
            for key in carbon_cost.keys():
                data[(investment_period, node, var, carrier, key)] = carbon_cost[key]
            var = "CarrierData"
            for carrier in modelhub.data["topology"]["carriers"]:
                for key in carrier_data.keys():
                    data[(investment_period, node, var, carrier, key)] = carrier_data[key]

    time_series = pd.DataFrame(data)
    time_series.index = modelhub.data["topology"]["temporal_information"]["time_index"]

    time_series.columns.set_names(
        ["InvestmentPeriod", "Node", "Key1", "Key2", "Key3"], inplace=True
    )

    return time_series

def read_data_patch(self, nr_timesteps: int, nr_nodes: int):
    """
    Monkey patch read data
    """
    self.data["topology"] = make_topology_for_testing(nr_timesteps, nr_nodes)
    self.data["config"] = initialize_configuration_templates()

    self.data["time_series_data"] = {}
    self.data["time_series_data"]["full_resolution"] = make_testing_time_series(self)

    self.data["network_data"] = {period: {}
                                    for period in self.data["topology"]["investment_periods"]
                                 }
    self.data["technology_data"] = {period: {
                                        node: {} for node in self.data["topology"]["nodes"]
                                        }
                                            for period in self.data["topology"]["investment_periods"]
                                    }
    self.component_constructors = {
        "network_constructors": {period: {}
                                    for period in self.data["topology"]["investment_periods"]
                                 },
        "technology_constructors": {period: {
                                        node: {} for node in self.data["topology"]["nodes"]
                                        }
                                            for period in self.data["topology"]["investment_periods"]
                                    }
    }
    self.data["config"]["node_config"] = {
        period: {
                node: {
                    carrier: {"curtailment_possible": 0}
                    for carrier in self.data["topology"]["carriers"]
                }
                for node in self.data["topology"]["nodes"]
            }
        for period in self.data["topology"]["investment_periods"]
    }

def make_topology_for_testing(nr_timesteps: int, nr_nodes: int) -> dict:
    """
    Monkey patch topology reading
    """
    topology = initialize_topology_templates()

    # get node locations and map to nodes
    topology["nodes"] = {
        f"node{i}": {"lon": 0, "lat": 0, "alt": 0}
        for i in range(1, nr_nodes + 1)
    }
    topology["temporal_information"] = get_temporal_information(
        topology["start_date"],
        topology["end_date"],
        topology["resolution"],
        0,
        nr_timesteps,
        "full_resolution"
    )
    return topology


def make_data_for_testing(nr_timesteps: int, config_update: dict) -> dict:
    """
    Makes a data for testing with specified number of time steps

    :param int nr_timesteps: Number of time steps
    :return: dict with config and topology
    :rtype: dict
    """
    data = {}
    data["topology"] = make_topology_for_testing(nr_timesteps, 1)
    config_template = initialize_configuration_templates()
    data["config"] = update_config(config_template, config_update)
    return data

def update_config(target: dict, update: dict):
    """ "
    Update model configuration where is needed

    :param dict target: original model configuration dictionary to be updated
    :param dict update: dictionary containing the updates to apply
    :return: updated configuration dictionary
    """

    for key, value in update.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            update_config(target[key], value)
        else:
            target[key] = value

    return target

def run_model(model, solver_name: str, objective: str = "capex"):
    """
    Runs a model and returns termination condition

    :param model: pyomo model
    :param str solver: solver to used
    :param str objective: objective to optimize
    :return: termination condition for respective model
    """
    if objective == "capex":
        model.obj = Objective(expr=model.var_capex, sense=minimize)
    elif objective == "emissions":
        model.obj = Objective(
            expr=sum(model.var_tec_emissions_pos[t] for t in model.set_t),
            sense=minimize,
        )

    solver = SolverFactory(solver_name)
    solution = solver.solve(model, tee=True)

    return solution.solver.termination_condition


def define_technology(
    tec_name: str,
    modelhub,
    technology_registry,
    load_path: Path,
    perf_type: int = None,
    capex_model: int = None,
    existing: int = 0,
    size_initial: float = 0,
    decommission: str = "impossible",
    additional_settings: dict = {}
):
    """
    Reads technology data and fits it

    :param str tec_name: name of the technology.
    :param modelhub: modelhub
    :param Path load_path: Path to load from
    :param int perf_type: performance function type (for generic conversion tecs)
    :param int capex_model: capex model (1,2,3,4)
    :param int existing: is technology existing or not,
    :param float size_initial: initial size of existing technology,
    :param str decommission: type of decommissioning "impossible", "continuous", "only_complete"
    :param dict additional_settings: dicts with additional settings to update in tec_data
    :return: Technology class
    """
    # Technology Class Creation
    with open(load_path / (tec_name + ".json")) as json_file:
        tec_data = json.load(json_file)
    tec_data["name"] = tec_name

    for setting_name, settings in additional_settings.items():
        tec_data[setting_name] = settings

    if perf_type:
        tec_data["Performance"]["performance_function_type"] = perf_type
    if capex_model:
        tec_data["Economics"]["capex_model"] = capex_model

    constructor = technology_registry.create(tec_data["tec_type"], tec_data)

    if existing:
        constructor.existing = existing
        constructor.size_initial = size_initial
        constructor.decommission = decommission

    # Technology fitting
    period = modelhub.data["topology"]["investment_periods"][0]
    node = list(modelhub.data["topology"]["nodes"].keys())[0]
    component_id = (period, node)
    constructor.fit_performance(modelhub, component_id)

    return constructor


def construct_tec_model(tec, modelhub, nr_timesteps):
    """
    Construct a mock technology model for testing

    :param Technology tec: Technology object.
    :param int nr_timesteps: Number of timesteps to create climate data for
    :param int dynamics: if dynamics should be used in mock model
    :return ConcreteModel m: Pyomo Concrete Model
    """

    m = ConcreteModel()
    m.set_t = Set(initialize=list(range(1, nr_timesteps + 1)))
    m.set_t_full = Set(initialize=list(range(1, nr_timesteps + 1)))
    # Todo: move to plugins
    # if dynamics:
    #     data["config"]["performance"]["dynamics"]["value"] = dynamics

    tec.construct_model(m, modelhub, m.set_t, m.set_t_full)
    if tec.big_m_transformation_required:
        perform_disjunct_relaxation(m)

    return m


def generate_output_constraint(model, demand: list, output_ratios: dict = None):
    """
    Generate an output constraint of a technology model

    :param model: pyomo model
    :param list demand: list of demand values to use
    :param output_ratios: output ratios to use
    :return: pyomo model
    """

    def init_output_constraint(const, t, car):
        if output_ratios:
            if isinstance(output_ratios.get(car), dict):
                alpha = output_ratios[car]["alpha1"]
            else:
                alpha = output_ratios[car]
            if isinstance(alpha, list):
                return model.var_output[t, car] >= demand[t - 1] * alpha[0]
            else:
                return model.var_output[t, car] == demand[t - 1] * alpha
        else:
            return model.var_output[t, car] == demand[t - 1]

    model.test_const_output1 = Constraint(
        model.set_t, model.set_output_carriers, rule=init_output_constraint
    )


def define_network(
    load_path: Path,
    netw_name: str,
    modelhub,
    network_registry,
    bidirectional_network: bool = False,
    energyconsumption: bool = False,
    existing: int = 0,
    size_initial: pd.DataFrame = None,
    decommission: str = "impossible",
):
    """
    reads TestNetwork from path and creates network object

    :param Path load_path:
    :param str netw_name:
    :param modelhub: modelhub
    :param bool bidirectional_network:
    :param bool energyconsumption:
    :return: Network object
    """
    with open(load_path / (f"TestNetwork{netw_name}.json")) as json_file:
        netw_data = json.load(json_file)

    netw_data["name"] = "TestNetwork"

    if bidirectional_network:
        netw_data["Performance"]["bidirectional_network"] = 1
        netw_data["Performance"]["bidirectional_network_precise"] = 1
    else:
        netw_data["Performance"]["bidirectional_network"] = 0

    if not energyconsumption:
        netw_data["Performance"]["energyconsumption"] = {}

    netw_matrix = create_empty_network_matrix(modelhub.data["topology"]["nodes"])
    netw_matrix.loc["node2", "node1"] = 1
    netw_matrix.loc["node1", "node2"] = 1

    netw_data["connection"] = netw_matrix
    netw_data["distance"] = netw_matrix
    if not existing:
        netw_data["size_max_arcs"] = netw_matrix * 10

    constructor = network_registry.create(netw_data["netw_type"], netw_data)

    if existing:
        constructor.existing = existing
        constructor.size_initial = size_initial
        constructor.decommission = decommission

    return constructor


def construct_netw_model(netw, modelhub, nr_timesteps: int):
    """
    Construct a mock network model for testing

    :param netw: Network object.
    :param int nr_timesteps: Number of timesteps to create network model for
    :return: Pyomo Model
    """
    period = modelhub.data["topology"]["investment_periods"][0]
    component_id = (period)
    netw.fit_performance(modelhub, component_id)

    m = pyo.ConcreteModel()
    m.set_t = pyo.Set(initialize=list(range(1, nr_timesteps + 1)))
    m.set_t_full = pyo.Set(initialize=list(range(1, nr_timesteps + 1)))
    m.set_nodes = pyo.Set(initialize=list(modelhub.data["topology"]["nodes"].keys()))

    netw.construct_model(m, modelhub, m.set_t, m.set_t_full, set_nodes = m.set_nodes)
    if netw.big_m_transformation_required:
        perform_disjunct_relaxation(m)

    return m


def generate_size_constraint(
    model, size: float = None, equality_constraint: bool = False
):
    """
    Adds a constraint on a technology size

    :param model: pyomo model
    :param float size: value to constrain size to
    :param equality_constraint: if True, equality constraint, otherwise less-equal
    :return: pyomo model
    """

    def init_size_constraint(const):
        if equality_constraint:
            return model.var_size == size
        else:
            return model.var_size <= size

    model.test_const_size = Constraint(rule=init_size_constraint)


def create_plugin_testing_mock_data(data_path: Path, node_name: str, period_name: str, technology:str):
    """
    Creates minimal mock input data folder structure for performance_from_climate_data at data_path

    :param Path data_path : directory to use
    :param str node_name: node name
    :param str period_name: period name
    :param str technology: technology name
    """
    topology_file = data_path / "Topology.json"
    topology = {
        "nodes": [node_name],
        "investment_periods": [period_name],
        "carriers": ["electricity"],
        "start_date": "2022-01-01 00:00",
        "end_date": "2022-12-31 23:00",
        "resolution": "1h",
        "investment_period_length": 1,
    }
    with open(topology_file, "w") as f:
        json.dump(topology, f, indent=4)

    node_locations = pd.DataFrame(
        index=topology["nodes"], columns=["lon", "lat", "alt"], data=[[4.9, 52, 10]]
    )
    node_locations.to_csv(data_path / "NodeLocations.csv", sep=";")

    (data_path / period_name).mkdir(parents=True, exist_ok=True)
    (data_path / period_name / "node_data" / node_name / "technology_time_series").mkdir(parents=True, exist_ok=True)
    (data_path / period_name / "node_data" / node_name / "technology_data").mkdir(parents=True, exist_ok=True)
    technologies = {"existing": {}, "new": [technology]}
    with open(
            data_path
            / period_name
            / "node_data"
            / node_name
            / "Technologies.json",
            "w",
    ) as f:
        json.dump(technologies, f, indent=4)
    copy_technology_data(data_path)
