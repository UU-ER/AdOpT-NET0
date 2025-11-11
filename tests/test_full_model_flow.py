from pathlib import Path
from warnings import warn
import pandas as pd
from adopt_net0.components.utilities import annualize


from pyomo.opt import TerminationCondition

from adopt_net0.modelhub import ModelHub
from adopt_net0.utilities import installed_capacities_existing


def test_full_model_flow(request):
    """
    Tests the full modelling pipeline with a small case study

    Topology:
    - Nodes: node1, node2
    - Investment Periods: period1
    - Technologies:
        - node1: existing gas power plant
        - node2: new electric boiler
        - node2: existing electrolyzer
    - Networks:
        - new electricity
    - Timeframe: 1 timestep

    Data:
    - Demand:
        - node1: electricity=1
        - node2: heat=1
        - node2: hydrogen=1
    - Import:
        - node1: gas
    - Import price:
        - node1: gas: 1

    The following is checked:
    - network size >=1
    - electric boiler size >= 1
    - output electric boiler = 1
    - hydrogen flow from eletrolyzer to demand = 1
    - total cost
    - total emissions
    """
    path = Path("tests/case_study_full_pipeline")

    adopthub = ModelHub()
    adopthub.read_data(path, start_period=0, end_period=1)
    adopthub.data.model_config["performance"]["pressure"]["pressure_on"]["value"] = 1
    adopthub.data.model_config["solveroptions"]["solver"][
        "value"
    ] = request.config.solver
    adopthub.data.model_config["reporting"]["save_summary_path"][
        "value"
    ] = request.config.result_folder_path
    adopthub.data.model_config["reporting"]["save_path"][
        "value"
    ] = request.config.result_folder_path
    adopthub.construct_model()
    adopthub.construct_balances()
    adopthub.solve()

    m = adopthub.model["full"]
    p = m.periods["period1"]

    # NETWORK CHECKS
    netw_block = p.network_block["electricitySimple"]

    # Size same in both directions
    s_arc1 = round(netw_block.arc_block["node1", "node2"].var_size.value, 3)
    s_arc2 = round(netw_block.arc_block["node2", "node1"].var_size.value, 3)
    assert s_arc1 == s_arc2

    # Flow in one direction is larger 1
    assert netw_block.arc_block["node1", "node2"].var_flow[1].value > 1
    # Flow in other direction is 0
    assert round(netw_block.arc_block["node2", "node1"].var_flow[1].value, 3) == 0

    # TECHNOLOGY CHECKS
    tec_block1 = p.node_blocks["node1"].tech_blocks_active[
        "TestTec_GasTurbine_simple_existing"
    ]
    # Size as assigned size
    assert round(tec_block1.var_size.value, 3) == 10
    # Output larger than heat+electricity demand
    assert tec_block1.var_output[1, "electricity"].value > 2
    # Gas import == gas consumption
    assert round(tec_block1.var_input[1, "gas"].value, 3) == round(
        m.periods["period1"].node_blocks["node1"].var_import_flow[1, "gas"].value, 3
    )

    tec_block2 = p.node_blocks["node2"].tech_blocks_active["TestTec_BoilerEl"]
    # Size larger heat demand
    assert tec_block2.var_size.value >= 1
    # Output equal to demand
    assert round(tec_block2.var_output[1, "heat"].value, 3) == 1

    # COST CHECKS
    assert m.var_npv.value > 0
    assert (
        "TestTec_WindTurbine"
        not in p.node_blocks["node1"].tech_blocks_active.index_set()
    )
    cost1 = m.var_npv.value

    # EMISSION CHECKS
    # Emission from gas combustion at gas turbine
    assert round(m.var_emissions_net.value, 3) == round(
        m.periods["period1"].node_blocks["node1"].var_import_flow[1, "gas"].value, 3
    )

    # test if technology can be added to a node
    adopthub.add_technology("period1", "node1", ["TestTec_WindTurbine"])
    adopthub.construct_balances()
    adopthub.solve()

    m = adopthub.model["full"]
    p = m.periods["period1"]
    cost2 = m.var_npv.value

    assert (
        "TestTec_WindTurbine" in p.node_blocks["node1"].tech_blocks_active.index_set()
    )
    assert (
        "TestTec_WindTurbine"
        not in p.node_blocks["node2"].tech_blocks_active.index_set()
    )
    assert cost2 < cost1


def test_full_model_flow_multiyear(request):
    """
    Tests the full modelling pipeline with multiple investment periods and a rolling horizon

    Topology:
    - Nodes: node1, node2
    - Investment Periods: Interval_1, Interval_2
    - Technologies:
        Interval_1:
        - node1: existing gas power plant (size = 10)
        - node2: new electric boiler
        - node2: existing electrolyzer (size = 3)
        - node2: new electrolyzer
        Interval_2:
        - node1: existing gas power plant
        - node2: new electric boiler
        - node2: existing electrolyzer
        - node2: new electrolyzer
    - Networks:
        Interval_1:
        - new electricity
        Interval_2:
        - new electricity
    - Timeframe: 1 timestep

    Data:
    - Demand:
        Interval_1
        - node1: electricity=1
        - node2: heat=3
        - node2: hydrogen=1
        Interval_1
        - node1: electricity=1
        - node2: heat=1
        - node2: hydrogen=3
    - Import:
        - node1: gas
    - Import price:
        - node1: gas: 1

    The following is checked:
    - Interval_1: network size >=1, Interval_2: network size >= Interval_1
    - Interval_1: hydrogen flow from electrolyzer to demand = 1, Interval_2: hydrogen flow from electrolyzer to demand = 3
    - Interval_2: has existing electric boiler capacity from previous interval
    - total cost
    """
    path = Path("tests/case_study_multiyear")

    # Build the model with investment intervals
    adopthub = {}
    intervals = ["Interval_1", "Interval_2"]

    # Construct and solve the model
    for i, interval in enumerate(intervals):
        path_interval = path / ("Case_" + interval)

        if i != 0:
            prev_interval = intervals[i - 1]
            installed_capacities_existing(
                adopthub, interval, prev_interval, path_interval
            )

        adopthub[interval] = ModelHub()
        adopthub[interval].read_data(path_interval, start_period=0, end_period=1)

        # Select options
        # adopthub[interval].data.model_config["solveroptions"]["solver"][
        #     "value"
        # ] = "gurobi"
        adopthub[interval].data.model_config["solveroptions"]["solver"][
            "value"
        ] = request.config.solver
        adopthub[interval].data.model_config["reporting"]["save_summary_path"][
            "value"
        ] = request.config.result_folder_path
        adopthub[interval].data.model_config["reporting"]["save_path"][
            "value"
        ] = request.config.result_folder_path
        adopthub[interval].data.model_config["reporting"]["case_name"][
            "value"
        ] = interval

        adopthub[interval].quick_solve()

    # Check results
    s_arc1 = {}
    electrolyzer_prod = {}
    for interval in intervals:
        m = adopthub[interval].model["full"]
        p = m.periods[interval]

        # Network flow
        flow_int = 0
        for netw in p.network_block:
            if "electricitySimple" in netw:
                netw_block = p.network_block[netw]
                flow_int += round(
                    netw_block.arc_block["node1", "node2"].var_flow[1].value, 3
                )

        # Arc flow
        s_arc1[interval] = flow_int

        # Hydrogen production
        prod_int = 0
        for tech in p.node_blocks["node2"].tech_blocks_active:
            if "Electrolyzer" in tech:
                tec_block = p.node_blocks["node2"].tech_blocks_active[tech]
                prod_int += round(tec_block.var_output[1, "hydrogen"].value, 3)

        electrolyzer_prod[interval] = prod_int

    # Check 1: Network flow increases in second interval
    assert s_arc1["Interval_1"] <= s_arc1["Interval_2"]

    # Check 2: Hydrogen production from electrolyzer is 1 in Interval_1 and 3 in Interval_2
    assert electrolyzer_prod["Interval_1"] == 1
    assert electrolyzer_prod["Interval_2"] == 3

    # Check heat supply
    node_block = (
        adopthub["Interval_1"].model["full"].periods["Interval_1"].node_blocks["node2"]
    )
    print(
        "Int1 TestTec_BoilerEl output",
        node_block.tech_blocks_active["TestTec_BoilerEl"].var_output[1, "heat"].value,
    )
    print(
        "Int1 TestTec_BoilerEl input",
        node_block.tech_blocks_active["TestTec_BoilerEl"]
        .var_input[1, "electricity"]
        .value,
    )
    print(
        "Int1 TestTec_BoilerEl size",
        node_block.tech_blocks_active["TestTec_BoilerEl"].var_size.value,
    )

    # Check 3: Existing electric boiler in Interval_2
    node_block = (
        adopthub["Interval_2"].model["full"].periods["Interval_2"].node_blocks["node2"]
    )
    print(
        "Int2 TestTec_BoilerEl",
        node_block.tech_blocks_active["TestTec_BoilerEl"].var_output[1, "heat"].value,
    )
    print(
        "Int2 TestTec_BoilerEl_existing",
        node_block.tech_blocks_active["TestTec_BoilerEl_existing"]
        .var_output[1, "heat"]
        .value,
    )
    assert "TestTec_BoilerEl_existing" in node_block.tech_blocks_active

    # COST CHECKS
    assert adopthub["Interval_1"].model["full"].var_npv.value > 0
    assert adopthub["Interval_2"].model["full"].var_npv.value > 0


def test_clustering_algo(request):
    """
    Tests method 1 and two of the clustering algorithm
    """

    path = Path("tests/case_study_full_pipeline")

    adopthub = ModelHub()
    adopthub.read_data(path, start_period=0, end_period=2 * 24)
    adopthub.data.model_config["performance"]["pressure"]["pressure_on"]["value"] = 0
    adopthub.data.model_config["solveroptions"]["solver"][
        "value"
    ] = request.config.solver
    adopthub.data.model_config["reporting"]["save_summary_path"][
        "value"
    ] = request.config.result_folder_path
    adopthub.data.model_config["reporting"]["save_path"][
        "value"
    ] = request.config.result_folder_path

    adopthub.construct_model()
    adopthub.construct_balances()
    adopthub.solve()

    m = adopthub.model["full"]
    npv_no_cluster = m.var_npv.value

    methods = [1, 2]
    N = [2, 1]
    adopthub = ModelHub()
    adopthub.data.set_settings(path)
    adopthub.data._read_topology()
    adopthub.data._read_model_config()
    adopthub.data.model_config["performance"]["pressure"]["pressure_on"]["value"] = 0
    adopthub.data.model_config["reporting"]["save_summary_path"][
        "value"
    ] = request.config.result_folder_path
    adopthub.data.model_config["reporting"]["save_path"][
        "value"
    ] = request.config.result_folder_path
    adopthub.data.model_config["solveroptions"]["solver"][
        "value"
    ] = request.config.solver
    for method in methods:
        for n in N:
            adopthub.data.model_config["optimization"]["typicaldays"]["N"]["value"] = n
            adopthub.data.model_config["optimization"]["typicaldays"]["method"][
                "value"
            ] = method
            adopthub.data._read_time_series()
            adopthub.data._read_node_locations()
            adopthub.data._read_energybalance_options()
            adopthub.data._read_technology_data()
            adopthub.data._read_network_data()

            # Clustering algorithms
            if (
                adopthub.data.model_config["optimization"]["typicaldays"]["N"]["value"]
                != 0
            ):
                adopthub.data._cluster_data()
            if adopthub.data.model_config["optimization"]["timestaging"]["value"] != 0:
                adopthub.data._average_data()

            adopthub.quick_solve()

            if n == 2:
                tol = 0.0001
            else:
                tol = 0.01

            assert (
                abs(npv_no_cluster - adopthub.model["clustered"].var_npv.value)
                / npv_no_cluster
            ) <= tol


def test_average_algo(request):
    """
    Tests two stage averaging algorithm
    """

    path = Path("tests/case_study_full_pipeline")

    adopthub = ModelHub()
    adopthub.read_data(path, start_period=0, end_period=2 * 24)
    adopthub.data.model_config["performance"]["pressure"]["pressure_on"]["value"] = 0
    adopthub.data.model_config["solveroptions"]["solver"][
        "value"
    ] = request.config.solver
    adopthub.data.model_config["reporting"]["save_summary_path"][
        "value"
    ] = request.config.result_folder_path
    adopthub.data.model_config["reporting"]["save_path"][
        "value"
    ] = request.config.result_folder_path

    adopthub.construct_model()
    adopthub.construct_balances()
    adopthub.solve()

    m = adopthub.model["full"]
    npv_no_cluster = m.var_npv.value

    adopthub = ModelHub()
    adopthub.data.set_settings(path)
    adopthub.data._read_topology()
    adopthub.data._read_model_config()

    adopthub.data.model_config["optimization"]["timestaging"]["value"] = 4
    adopthub.data.model_config["reporting"]["save_summary_path"][
        "value"
    ] = request.config.result_folder_path
    adopthub.data.model_config["performance"]["pressure"]["pressure_on"]["value"] = 0
    adopthub.data.model_config["reporting"]["save_path"][
        "value"
    ] = request.config.result_folder_path
    adopthub.data.model_config["solveroptions"]["solver"][
        "value"
    ] = request.config.solver

    adopthub.data._read_time_series()
    adopthub.data._read_node_locations()
    adopthub.data._read_energybalance_options()
    adopthub.data._read_technology_data()
    adopthub.data._read_network_data()

    # Averaging algorithms
    if adopthub.data.model_config["optimization"]["timestaging"]["value"] != 0:
        adopthub.data._average_data()

    adopthub.quick_solve()

    assert (
        abs(npv_no_cluster - adopthub.model["full"].var_npv.value) / npv_no_cluster
    ) <= 0.01

    assert (
        abs(npv_no_cluster - adopthub.model["averaged"].var_npv.value) / npv_no_cluster
    ) <= 0.1


def test_objective_functions(request):
    """
    Tests the following objective functions:

    - pareto
    - emissions_net
    - min emissions at min cost
    - min cost at emission limit
    """

    path = Path("tests/case_study_full_pipeline")

    adopthub = ModelHub()
    adopthub.read_data(path, start_period=0, end_period=1)

    adopthub.data.model_config["solveroptions"]["solver"][
        "value"
    ] = request.config.solver
    adopthub.data.model_config["performance"]["pressure"]["pressure_on"]["value"] = 0
    adopthub.data.model_config["reporting"]["save_summary_path"][
        "value"
    ] = request.config.result_folder_path
    adopthub.data.model_config["reporting"]["save_path"][
        "value"
    ] = request.config.result_folder_path

    adopthub.construct_model()
    adopthub.construct_balances()
    adopthub._define_solver_settings()

    adopthub._optimize_emissions_net()
    adopthub._optimize_costs_minE()
    adopthub._optimize_costs_emissionslimit()

    adopthub._solve_pareto()


def test_scaling(request):
    """
    Tests model scaling
    """
    path = Path("tests/case_study_full_pipeline")

    adopthub = ModelHub()
    adopthub.read_data(path, start_period=0, end_period=1)

    adopthub.data.model_config["scaling"]["scaling_on"]["value"] = 1
    adopthub.data.model_config["reporting"]["save_summary_path"][
        "value"
    ] = request.config.result_folder_path
    adopthub.data.model_config["reporting"]["save_path"][
        "value"
    ] = request.config.result_folder_path
    adopthub.data.model_config["performance"]["pressure"]["pressure_on"]["value"] = 0
    adopthub.data.model_config["solveroptions"]["solver"][
        "value"
    ] = request.config.solver

    adopthub.construct_model()
    adopthub.construct_balances()
    adopthub.solve()
