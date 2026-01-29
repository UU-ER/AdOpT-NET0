from pyomo.environ import Constraint, TerminationCondition, Objective, minimize
import pytest
import json

from adopt_net0 import *
from adopt_net0.plugins.plugin_manager import PluginManager
from tests.utilities import (
    generate_size_constraint
)
from adopt_net0.core.modelhub import ModelHub


def _make_data_for_retrofit_tests(request, carriers, retrofit_plugin_name, settings):
    technology = "TestTec_Conv1"
    data_path = request.config.data_folder_path

    create_optimization_templates(data_path)

    # Topology
    with open(data_path / "Topology.json", "r") as json_file:
        topology = json.load(json_file)
    topology["nodes"] = ["node1"]
    topology["carriers"] = carriers
    topology["start_date"] = '2022-01-01 00:00'
    topology["end_date"] = '2022-01-01 01:00'
    with open(data_path / "Topology.json", "w") as json_file:
        json.dump(topology, json_file, indent=4)

    plugin_list = {"modeling_plugins.technology_retrofits":
        {
            "config":
                {
                    "retrofits": [retrofit_plugin_name]
                }
        }
    }
    with open(data_path / "Plugins.json", "w") as json_file:
        json.dump(plugin_list, json_file, indent=4)

    create_input_data_folder_template(data_path)

    # Add technology
    with open(data_path / "period1" / "node_data" / "node1" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)
    technologies["new"] = [technology]
    with open(data_path / "period1" / "node_data" / "node1" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    copy_technology_data(data_path, request.config.technology_data_folder_path)

    with open(data_path / "period1" / "node_data" / "node1" / "technology_data" / (technology + ".json"),
              "r") as json_file:
        tec_data = json.load(json_file)
    tec_data["settings_technology_retrofits"] = settings
    with open(data_path / "period1" / "node_data" / "node1" / "technology_data" / (technology + ".json"),
              "w") as json_file:
        json.dump(tec_data, json_file, indent=4)

def test_plugin_technology_retrofit(request):

    pm = PluginManager()
    plugin_list = {"modeling_plugins.technology_retrofits":
      {
        "config":
        {
          "retrofits": ["Efficiency_Retrofit", "Ccs_Retrofit"]
        }
      }
    }

    pm.register(plugin_list)

def test_efficiency_retrofit(request):
    data_path = request.config.data_folder_path
    carriers = ["electricity", "hydrogen", "heat", "gas"]
    retrofit_plugin_name = "Efficiency_Retrofit"

    settings = {
        "enable_efficiency_retrofit": True,
        "efficiency_retrofit": {
            "unit_capex": 10,
            "coefficients":[
                {
                    "output_carrier": "electricity",
                    "coefficient": 0.0,
                    "opex_variable": 1,
                },
                {
                    "output_carrier": "heat",
                    "coefficient": 0.0,
                    "opex_variable": 1,
                },
        ]
        }
    }
    _make_data_for_retrofit_tests(request, carriers, retrofit_plugin_name, settings)

    fill_carrier_data(data_path, value_or_data=10, columns=['Import limit'],
                      carriers=['gas'], nodes=['node1'])
    fill_carrier_data(data_path, value_or_data=1, columns=['Demand'],
                      carriers=['electricity'], nodes=['node1'])
    fill_carrier_data(data_path, value_or_data=1, columns=['Demand'],
                      carriers=['heat'], nodes=['node1'])

    # INFEASIBLE TEST
    modelhub = ModelHub()
    modelhub.read_data(data_path, start_period=0, end_period=1)
    modelhub.process_data()
    modelhub.aggregate_data()
    modelhub.construct_components()
    modelhub.construct_balances()

    b_tec = modelhub.model["full_resolution"].periods["period1"].node_blocks["node1"].tech_blocks_active["TestTec_Conv1"]
    generate_size_constraint(
        b_tec, 2
    )

    modelhub.solve()
    termination_condition = modelhub.solution.solver[0]["Termination condition"]
    assert termination_condition in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBLE TEST
    settings["efficiency_retrofit"]["coefficients"][0]["coefficient"] = 1
    settings["efficiency_retrofit"]["coefficients"][1]["coefficient"] = 1
    _make_data_for_retrofit_tests(request, carriers, retrofit_plugin_name, settings)

    fill_carrier_data(data_path, value_or_data=10, columns=['Import limit'],
                      carriers=['gas'], nodes=['node1'])
    fill_carrier_data(data_path, value_or_data=1, columns=['Demand'],
                      carriers=['electricity'], nodes=['node1'])
    fill_carrier_data(data_path, value_or_data=1, columns=['Demand'],
                      carriers=['heat'], nodes=['node1'])

    modelhub = ModelHub()
    modelhub.read_data(data_path, start_period=0, end_period=1)
    modelhub.process_data()
    modelhub.aggregate_data()
    modelhub.construct_components()
    modelhub.construct_balances()

    b_tec = modelhub.model["full_resolution"].periods["period1"].node_blocks["node1"].tech_blocks_active[
        "TestTec_Conv1"]
    generate_size_constraint(
        b_tec, 2
    )

    modelhub.solve()
    termination_condition = modelhub.solution.solver[0]["Termination condition"]

    efficiency_retrofit_block = modelhub.model["full_resolution"].periods["period1"].node_blocks["node1"].tech_blocks_active["TestTec_Conv1"].efficiency_retrofit

    assert efficiency_retrofit_block.var_delta_capex.value != pytest.approx(0, rel=1e-3)
    assert efficiency_retrofit_block.var_delta_emissions[1].value == pytest.approx(0, rel=1e-3)
    assert efficiency_retrofit_block.var_delta_opex_fix.value == pytest.approx(0, rel=1e-3)
    assert efficiency_retrofit_block.var_delta_opex_var.value == pytest.approx(1, rel=1e-3)

    assert termination_condition in [
        TerminationCondition.optimal,
    ]


def test_ccs_retrofit(request):
    """
    tests CCS
    """
    data_path = request.config.data_folder_path
    carriers = ["electricity", "hydrogen", "CO2captured", "heat", "gas"]
    retrofit_plugin_name = "CCS_Retrofit"
    with open(request.config.technology_data_folder_path / "TestTec_CCS_MEA.json", "r") as json_file:
        ccs_settings = json.load(json_file)
    settings = {
        "enable_ccs_retrofit": True,
        "ccs_retrofit": ccs_settings
    }
    _make_data_for_retrofit_tests(request, carriers, retrofit_plugin_name, settings)

    fill_carrier_data(data_path, value_or_data=10, columns=['Import limit'],
                            carriers=['gas'], nodes=['node1'])
    fill_carrier_data(data_path, value_or_data=10, columns=['Import limit'],
                            carriers=['heat'], nodes=['node1'])
    fill_carrier_data(data_path, value_or_data=10, columns=['Export limit'],
                            carriers=['CO2captured'], nodes=['node1'])
    fill_carrier_data(data_path, value_or_data=1, columns=['Demand'],
                            carriers=['electricity'], nodes=['node1'])

    # Infeasible case
    modelhub = ModelHub()
    modelhub.read_data(data_path, start_period=0, end_period=1)
    modelhub.data["config"]["optimization"]["objective"]["value"] = "costs_emissionlimit"
    modelhub.data["config"]["optimization"]["emission_limit"]["value"] = 0
    modelhub.quick_solve()
    termination_condition = modelhub.solution.solver[0]["Termination condition"]
    assert termination_condition in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # Feasible case / Cost optimization
    modelhub = ModelHub()
    modelhub.read_data(data_path, start_period=0, end_period=1)
    modelhub.data["config"]["optimization"]["objective"]["value"] = "costs"
    modelhub.quick_solve()
    termination_condition = modelhub.solution.solver[0]["Termination condition"]
    assert termination_condition == TerminationCondition.optimal
    cost_no_ccs = modelhub.model["full_resolution"].var_npv.value
    emissions_no_ccs = modelhub.model["full_resolution"].var_emissions_net.value

    ccs_retrofit_block = modelhub.model["full_resolution"].periods["period1"].node_blocks["node1"].tech_blocks_active["TestTec_Conv1"].ccs_retrofit
    assert round(ccs_retrofit_block.var_size.value, 3) == 0
    assert round(ccs_retrofit_block.var_delta_output[1, "CO2captured"].value, 3) == 0
    assert round(ccs_retrofit_block.var_delta_input[1, "heat"].value, 3) == 0
    assert round(ccs_retrofit_block.var_delta_input[1, "electricity"].value, 3) == 0

    modelhub = ModelHub()
    modelhub.read_data(data_path, start_period=0, end_period=1)
    modelhub.data["config"]["optimization"]["objective"]["value"] = "emissions_net"
    modelhub.quick_solve()
    termination_condition = modelhub.solution.solver[0]["Termination condition"]
    assert termination_condition == TerminationCondition.optimal
    cost_ccs = modelhub.model["full_resolution"].var_npv.value
    emissions_ccs = modelhub.model["full_resolution"].var_emissions_net.value

    ccs_retrofit_block = modelhub.model["full_resolution"].periods["period1"].node_blocks["node1"].tech_blocks_active["TestTec_Conv1"].ccs_retrofit
    assert round(ccs_retrofit_block.var_size.value, 3) >= 0.2
    assert round(ccs_retrofit_block.var_delta_output[1, "CO2captured"].value, 3) >= 0.9
    assert round(ccs_retrofit_block.var_delta_input[1, "heat"].value, 3) >= 0.1
    assert round(ccs_retrofit_block.var_delta_input[1, "electricity"].value, 3) >= 0.001
    assert cost_ccs > cost_no_ccs * 1.01
    assert round(emissions_ccs, 2) <= round(emissions_no_ccs * 0.1, 2)

