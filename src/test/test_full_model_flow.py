from pathlib import Path
from warnings import warn

from src.energyhub import EnergyHub


def test_full_model_flow(request):
    """
    Tests the full modelling pipeline with a small case study

    Topology:
    - Nodes: node1, node2
    - Investment Periods: period1
    - Technologies:
        - node1: existing gas power plant, PV
        - node2: new electric boiler
    - Networks:
        - new electricity
    - Timeframe: 1 timestep

    Data:
    - Demand:
        - node1: electricity=1
        - node2: heat=1
    - Import:
        - node1: gas
    - Import price:
        - node1: gas: 1

    The following is checked:
    - network size >=1
    - electric boiler size >= 1
    - output electric boiler = 1
    - total cost
    - total emissions
    """
    path = Path("src/test/case_study_full_pipeline")

    pyhub = EnergyHub()
    pyhub.read_data(path, start_period=0, end_period=1)
    pyhub.construct_model()
    pyhub.construct_balances()
    pyhub.data.model_config["solveroptions"]["solver"]["value"] = request.config.solver
    pyhub.solve()

    m = pyhub.model["full"]
    p = m.periods["period1"]

    if request.config.solver == "glpk":
        warn(
            "This turns out to be infeasible if solved with GLPK. Test on local machine!"
        )
    else:

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

        # EMISSION CHECKS
        # Emission from gas combustion at gas turbine
        assert round(m.var_emissions_net.value, 3) == round(
            m.periods["period1"].node_blocks["node1"].var_import_flow[1, "gas"].value, 3
        )


def test_clustering(request):

    path = Path("src/test/case_study_full_pipeline")

    pyhub = EnergyHub()
    pyhub.read_data(path, start_period=0, end_period=2 * 24)
    pyhub.construct_model()
    pyhub.construct_balances()
    pyhub.data.model_config["solveroptions"]["solver"]["value"] = request.config.solver
    pyhub.solve()

    m = pyhub.model["full"]
    npv_no_cluster = m.var_npv.value

    methods = [1, 2]
    N = [2, 1]
    pyhub = EnergyHub()
    pyhub.data.set_settings(path)
    pyhub.data._read_topology()
    pyhub.data._read_model_config()
    for method in methods:
        for n in N:
            pyhub.data.model_config["optimization"]["typicaldays"]["N"]["value"] = n
            pyhub.data.model_config["optimization"]["typicaldays"]["method"][
                "value"
            ] = method
            pyhub.data._read_time_series()
            pyhub.data._read_node_locations()
            pyhub.data._read_energybalance_options()
            pyhub.data._read_technology_data()
            pyhub.data._read_network_data()

            # Clustering/Averaging algorithms
            if (
                pyhub.data.model_config["optimization"]["typicaldays"]["N"]["value"]
                != 0
            ):
                pyhub.data._cluster_data()
            if pyhub.data.model_config["optimization"]["timestaging"]["value"] != 0:
                pyhub.data._average_data()

            # pyhub.read_data(path, start_period=0, end_period=24 * 3)
            pyhub.quick_solve()

            if n == 2:
                tol = 0.0001
            else:
                tol = 0.01

            assert (
                abs(npv_no_cluster - pyhub.model["clustered"].var_npv.value)
                / npv_no_cluster
            ) <= tol
