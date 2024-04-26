import pytest
from pathlib import Path
from pyomo.environ import (
    ConcreteModel,
    Set,
    Constraint,
    Objective,
    TerminationCondition,
    SolverFactory,
    minimize,
)
import json

from src.test.utilities import create_patched_datahandle
from src.components.technologies.technology import Technology
from src.model_construction.construct_nodes import construct_node_block
from src.data_preprocessing import initialize_configuration_templates


def initialize_node_block(nr_timesteps: int) -> ConcreteModel:
    """
    Construct a mock technology model for testing

    :param Technology tec: Technology object.
    :param int nr_timesteps: Number of timesteps to create climate data for
    :return ConcreteModel m: Pyomo Concrete Model
    """

    m = ConcreteModel()
    m.set_t = Set(initialize=list(range(1, nr_timesteps + 1)))
    dh = create_patched_datahandle(nr_timesteps)

    node = dh.topology["nodes"][0]
    period = dh.topology["investment_periods"][0]

    data = {}
    data["topology"] = dh.topology
    data["config"] = initialize_configuration_templates()
    data["time_series"] = dh.time_series["full"][period][node]
    data["energybalance_options"] = dh.energybalance_options[period][node]
    data["technology_data"] = dh.technology_data["full"][period][node]
    data["network_data"] = dh.network_data["full"][period]

    m = construct_node_block(m, data, m.set_t)
    return m
