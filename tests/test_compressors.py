import warnings

import pyomo.core.base.param
import pytest
from pathlib import Path
from pyomo.environ import ConcreteModel, Set, Constraint, TerminationCondition
import json
import numpy as np
import math

from tests.utilities import (
    make_data_for_testing,
    run_model,
)
from adopt_net0.components.compressors.compressor import Compressor
from adopt_net0.data_management.utilities import open_json
from adopt_net0.components.utilities import annualize
from adopt_net0.components.utilities import perform_disjunct_relaxation


def define_compressor(
    compressor_name: str,
    carrier: str,
    connection_info: dict,
    load_path: Path,
    decommission: str = "impossible",
):
    """
    Reads compressor data and fits it

    :param str compressor_name: name of the compressor
    :param str carrier: carrier for compressor
    :param dict connection_info: connection information for compressor
    :param Path load_path: Path to load from
    :return: Technology class
    """
    # Compressor Class Creation
    with open(load_path / (compressor_name + ".json")) as json_file:
        compr = json.load(json_file)
    compr["name"] = compressor_name
    compr["connection_info"] = connection_info["connection_info"]
    compr["carrier"] = carrier

    compr = Compressor(compr)

    compr.fit_compressor_performance()

    return compr


def construct_compressor_model(compr, nr_timesteps: int):
    """
    Construct a mock compressor model for testing

    :param Compressor compr: Compressor object.
    :param int nr_timesteps: Number of timesteps
    :return ConcreteModel m: Pyomo Concrete Model
    """

    m = ConcreteModel()
    m.set_t = Set(initialize=list(range(1, nr_timesteps + 1)))
    m.set_t_full = Set(initialize=list(range(1, nr_timesteps + 1)))
    data = make_data_for_testing(nr_timesteps)

    # I took dynamics out

    m = compr.construct_compressor_model(m, data, m.set_t, m.set_t_full)
    if compr.big_m_transformation_required:
        m = perform_disjunct_relaxation(m)

    return m


def define_connection(component1, component2):
    """
    Construct a mock connection model for testing

    :param dict component1: Component1 for testing
    :param dict component2: Component2 for testing
    :return dict connection_data: Dictionary of the connection
    """
    return {
        "connection_info": {
            "components": [component1["name"], component2["name"]],
            "pressure": [component1["pressure"], component2["pressure"]],
            "type": [component1["type"], component2["type"]],
            "existing": [component1["existing"], component2["existing"]],
        }
    }


def generate_consumption_constraint(model, flow: list, ratio: float):
    """
    Generate an consumption constraint of a technology model

    :param model: pyomo model
    :param list flow: list of demand values to use
    :param float ratio: output ratios to use
    :return: pyomo model
    """

    def flow_consumption_constraint(const, t, car):
        return model.var_consumption_energy[t, car] == flow[t - 1] * ratio

    model.test_const_consumption = Constraint(
        model.set_t, model.set_consumed_carriers, rule=flow_consumption_constraint
    )

    return model


def test_hydrogenCompressor(request):
    """
    tests Hydrogen Compressor
    """
    time_steps = 1
    compressor = "TestCompressor_hydrogen"
    component1 = {
        "name": "Electrolyzer",
        "pressure": 30.0,
        "type": "technology",
        "existing": 0,
    }
    component2 = {
        "name": "Pipeline_HighP",
        "pressure": 60.0,
        "type": "network",
        "existing": 0,
    }
    connection_info = define_connection(component1, component2)

    compr = define_compressor(
        compressor,
        "hydrogen",
        connection_info,
        request.config.compressor_data_folder_path,
    )

    isentropic_efficiency = 0.75
    R = 8.314  # kJ/kmol/K
    k = 1.4
    T_in = 298.15  # K
    Z = 1
    pressure_ratio_per_stage = 2.1
    n_stages = math.ceil(
        math.log(
            connection_info["connection_info"]["pressure"][1]
            / connection_info["connection_info"]["pressure"][0]
        )
        / math.log(pressure_ratio_per_stage)
    )

    consumption = (
        Z
        / 120
        / 2
        * T_in
        * (R / 1000)
        * n_stages
        * (k / (k - 1))
        * (1 / isentropic_efficiency)
        * (
            (
                connection_info["connection_info"]["pressure"][1]
                / connection_info["connection_info"]["pressure"][0]
            )
            ** ((k - 1) / (n_stages * k))
        )
        - 1
    )

    # INFEASIBILITY CASES
    model = construct_compressor_model(compr, nr_timesteps=time_steps)
    model = generate_consumption_constraint(model, [9], consumption)
    model.test_const_flow = Constraint(expr=model.var_flow[1] == 10)
    model.test_const_consumption = Constraint(
        expr=model.var_consumption_energy[1, "electricity"] == 0
    )

    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_compressor_model(compr, nr_timesteps=time_steps)
    model.test_const_flow = Constraint(expr=model.var_flow[1] == 10)

    termination = run_model(model, request.config.solver)
    assert termination == TerminationCondition.optimal
    assert (
        model.var_consumption_energy[1, "electricity"].value >= 10 * consumption * 0.99
    )
    assert model.var_size.value >= model.var_consumption_energy[1, "electricity"].value
