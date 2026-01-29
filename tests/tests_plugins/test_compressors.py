from pyomo.environ import Constraint, TerminationCondition
import json
import pytest
import os
import pandas as pd

from adopt_net0 import *
from adopt_net0.plugins.modeling_plugins.compression_energy import copy_compressor_data

def _make_data_for_compression_test(request, carriers, settings_tec, settings_netw):
    technology = "TestTec_Conv1"
    network = "TestNetworkSimple"
    data_path = request.config.data_folder_path

    create_optimization_templates(data_path)

    # Topology
    with open(data_path / "Topology.json", "r") as json_file:
        topology = json.load(json_file)
    topology["nodes"] = ["node1", "node2"]
    topology["carriers"] = carriers
    topology["start_date"] = '2022-01-01 00:00'
    topology["end_date"] = '2022-01-01 01:00'
    with open(data_path / "Topology.json", "w") as json_file:
        json.dump(topology, json_file, indent=4)

    plugin_list = {"modeling_plugins.compression_energy":
        {
            "config":
                {
                    "carriers": ["hydrogen"]
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
    tec_data["settings_compression_energy"] = settings_tec
    tec_data["Performance"]["input_carrier"] = ["electricity"]
    tec_data["Performance"]["output_carrier"] = ["hydrogen"]
    tec_data["Performance"]["main_input_carrier"] = "electricity"
    with open(data_path / "period1" / "node_data" / "node1" / "technology_data" / (technology + ".json"),
              "w") as json_file:
        json.dump(tec_data, json_file, indent=4)

    # Add network
    with open(data_path / "period1" / "Networks.json", "r") as json_file:
        networks = json.load(json_file)
    networks["existing"] = [network]
    with open(data_path / "period1" / "Networks.json", "w") as json_file:
        json.dump(networks, json_file, indent=4)

    copy_network_data(data_path, request.config.network_data_folder_path)

    with open(data_path / "period1" / "network_data" / (network + ".json"),
              "r") as json_file:
        netw_data = json.load(json_file)
    netw_data["settings_compression_energy"] = settings_netw
    netw_data["Performance"]["carrier"] = "hydrogen"
    with open(data_path / "period1" / "network_data" / (network + ".json"),
              "w") as json_file:
        json.dump(netw_data, json_file, indent=4)

    # Make a new folder for the new network
    os.makedirs(data_path / "period1" / "network_topology" / "existing" / network, exist_ok=True)

    # Connection
    connection = pd.read_csv(data_path / "period1" / "network_topology" / "existing" / "connection.csv", sep=";",
                             index_col=0)
    connection.loc["node1", "node2"] = 1
    connection.loc["node2", "node1"] = 1
    connection.to_csv(
        data_path / "period1" / "network_topology" / "existing" / network / "connection.csv", sep=";")

    # Distance
    distance = pd.read_csv(data_path / "period1" / "network_topology" / "existing" / "distance.csv", sep=";",
                           index_col=0)
    distance.loc["node1", "node2"] = 1
    distance.loc["node2", "node1"] = 1
    distance.to_csv(data_path / "period1" / "network_topology" / "existing" / network / "distance.csv",
                    sep=";")

    # Size
    distance = pd.read_csv(data_path / "period1" / "network_topology" / "existing" / "size.csv", sep=";",
                           index_col=0)
    distance.loc["node1", "node2"] = 10
    distance.loc["node2", "node1"] = 10
    distance.to_csv(data_path / "period1" / "network_topology" / "existing" / network / "size.csv",
                    sep=";")

    # Add compressor
    pressure_data = {
        "hydrogen": {
            "Demand": {
                "value": 60.0,
                "unit": "bar"
            },
        }
    }

    for node in ["node1", "node2"]:
        with open(data_path / "period1" / "node_data" / node / "carrier_data" / "PressureExchangeData.json", "w") as json_file:
            json.dump(pressure_data, json_file, indent=4)

    copy_compressor_data(data_path, request.config.compressor_data_folder_path)



def test_hydrogenCompressor(request):
    """
    tests Hydrogen Compressor
    """
    data_path = request.config.data_folder_path
    carriers = ["electricity", "hydrogen"]

    plugin_settings_tec = {
        "pressure_levels": {
            "hydrogen": {
                "outlet": 30
            }
        }
    }

    plugin_settings_netw = {
        "pressure_levels": {
            "hydrogen": {
                "inlet": 60,
                "outlet": 50
            }
        }
    }

    _make_data_for_compression_test(request, carriers, plugin_settings_tec, plugin_settings_netw)


    fill_carrier_data(data_path, value_or_data=10, columns=['Import limit'],
                      carriers=['electricity'], nodes=['node1'])
    fill_carrier_data(data_path, value_or_data=1, columns=['Demand'],
                      carriers=['hydrogen'], nodes=['node2'])

    # INFEASIBLE TEST
    modelhub = ModelHub()
    modelhub.read_data(data_path, start_period=0, end_period=1)
    modelhub.process_data()
    modelhub.aggregate_data()
    modelhub.construct_components()
    modelhub.construct_balances()

    # Set input to compressor to 0
    b_compressor = modelhub.model["full_resolution"].periods["period1"].node_blocks["node1"].compressor_blocks_active["hydrogen","TestTec_Conv1","TestNetworkSimple_existing"]

    def flow_consumption_constraint(const, t, car):
        return b_compressor.var_consumption_energy[t, car] == 0

    b_compressor.test_const_consumption = Constraint(
        modelhub.model["full_resolution"].periods["period1"].set_t_full, b_compressor.set_consumed_carriers, rule=flow_consumption_constraint
    )

    modelhub.solve()
    termination_condition = modelhub.solution.solver[0]["Termination condition"]
    assert termination_condition in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBLE TEST
    modelhub = ModelHub()
    modelhub.read_data(data_path, start_period=0, end_period=1)
    modelhub.quick_solve()
    termination_condition = modelhub.solution.solver[0]["Termination condition"]
    assert termination_condition in [
        TerminationCondition.optimal,
    ]

    b_compressor = modelhub.model["full_resolution"].periods["period1"].node_blocks["node1"].compressor_blocks_active["hydrogen","TestTec_Conv1","TestNetworkSimple_existing"]
    compressor_energy = b_compressor.var_consumption_energy[1, "electricity"].value
    compressor_size = b_compressor.var_size.value
    compressor_flow = b_compressor.var_flow[1].value
    network_flow = modelhub.model["full_resolution"].periods["period1"].network_block["TestNetworkSimple_existing"].arc_block["node1","node2"].var_flow[1].value

    assert compressor_size >= compressor_energy
    assert compressor_flow == pytest.approx(network_flow, rel=1e-3)
