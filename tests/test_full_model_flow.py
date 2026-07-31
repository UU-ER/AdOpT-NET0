import json
import warnings
from pathlib import Path
import pandas as pd
from adopt_net0.modelhub import ModelHub
from adopt_net0.utilities import installed_capacities_existing
from adopt_net0.result_management.read_results import (
    add_values_to_summary,
    add_carry_over_annualization_to_summary,
    add_discounted_cost_to_summary,
)
import pyomo.environ as pyo


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

    compressor_1 = p.node_blocks["node2"].compressor_blocks_active[
        "hydrogen", "TestTec_Electrolyzer_existing", "Demand"
    ]
    # Flow equal to hydrogen demand
    assert round(compressor_1.var_flow[1].value, 3) == 1

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
    Tests the full modelling pipeline with multiple investment periods and a myopic foresight
    method, without lifetime tracking (intervals_between_years not provided).

    Topology:
    - Nodes: node1, node2
    - Investment Periods: Interval_1, Interval_2
    - Technologies:
        Interval_1:
        - node1: existing gas power plant (size = 10)
        - node2: new electric boiler
        - node2: new electrolyzer
        Interval_2:
        - node1: existing gas power plant
        - node2: new electric boiler
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
        Interval_2
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
    - carry_over_sizes and remaining_lifetime are NOT written to Technologies.json and Networks.json
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

        adopthub[interval].construct_model()
        adopthub[interval].construct_balances()

        # Fix size to prevent constraint violation in glpk
        if interval == "Interval_1":
            p = adopthub[interval].model["full"].periods[interval]
            b_tec = p.node_blocks["node2"].tech_blocks_active["TestTec_BoilerEl"]

            # Add additional constraint to force size in glpk: var_size >= 15
            def glpk_boiler_size(m):
                return b_tec.var_size >= 15

            b_tec.const_boiler_size = pyo.Constraint(rule=glpk_boiler_size)

            b_netw = p.network_block["electricitySimple"]

            def glpk_netw_size(m):
                return b_netw.arc_block["node1", "node2"].var_size >= 15

            b_tec.const_netw_size = pyo.Constraint(rule=glpk_netw_size)

        adopthub[interval].solve()

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

    # Check 3: Existing electric boiler in Interval_2
    if request.config.solver == "glpk":
        warnings.warn(
            "GLPK presolve fixes var_size and does not return it to Pyomo, so the "
            "carry-over cannot be verified; run with gurobi for this check."
        )
    else:
        node_block = (
            adopthub["Interval_2"]
            .model["full"]
            .periods["Interval_2"]
            .node_blocks["node2"]
        )
        assert "TestTec_BoilerEl_existing" in node_block.tech_blocks_active

    # COST CHECKS
    assert adopthub["Interval_1"].model["full"].var_npv.value > 0
    assert adopthub["Interval_2"].model["full"].var_npv.value > 0

    # Check 4: No carry_over tracking written when intervals_between_years is not provided
    tec_json = json.load(
        open(
            path
            / "Case_Interval_2"
            / "Interval_2"
            / "node_data"
            / "node2"
            / "Technologies.json"
        )
    )
    assert "carry_over_sizes" not in tec_json
    assert "remaining_lifetime" not in tec_json

    netw_json = json.load(
        open(path / "Case_Interval_2" / "Interval_2" / "Networks.json")
    )
    assert "carry_over_sizes" not in netw_json
    assert "remaining_lifetime" not in netw_json


def test_full_model_flow_multiyear_lifetime(request):
    """
    Tests the full modelling pipeline with multiple investment periods, a myopic foresight
    method, and lifetime tracking (intervals_between_years=[10]).

    Topology:
    - Nodes: node1, node2
    - Investment Periods: Interval_1, Interval_2
    - Technologies:
        Interval_1:
        - node1: existing gas power plant (size = 10)
        - node2: new electric boiler
        - node2: new electrolyzer
        Interval_2:
        - node1: existing gas power plant
        - node2: new electric boiler (existing from Interval_1)
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
        Interval_2
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
    - remaining_lifetime written for boiler carry_over in Interval_1: lifetime=25, step=10 → 15
    - carry_over_sizes written for boiler and consistent with existing size
    - network remaining_lifetime: technical_lifetime=100 preferred over lifetime=25, step=10 → 90
    - network carry_over_sizes written
    - per-carry_over CSV size_Interval_1.csv written for electricitySimple
    """
    path = Path("tests/case_study_multiyear")

    # Build the model with investment intervals
    adopthub = {}
    intervals = ["Interval_1", "Interval_2"]
    intervals_between_years = [10]

    # Construct and solve the model
    for i, interval in enumerate(intervals):
        path_interval = path / ("Case_" + interval)

        if i != 0:
            prev_interval = intervals[i - 1]
            installed_capacities_existing(
                adopthub,
                interval,
                prev_interval,
                path_interval,
                intervals_between_years,
                i,
            )

        adopthub[interval] = ModelHub()
        adopthub[interval].read_data(path_interval, start_period=0, end_period=1)

        # Select options
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

        adopthub[interval].construct_model()
        adopthub[interval].construct_balances()

        # Fix size to prevent constraint violation in glpk
        if interval == "Interval_1":
            p = adopthub[interval].model["full"].periods[interval]
            b_tec = p.node_blocks["node2"].tech_blocks_active["TestTec_BoilerEl"]

            # Add additional constraint to force size in glpk: var_size >= 15
            def glpk_boiler_size(m):
                return b_tec.var_size >= 15

            b_tec.const_boiler_size = pyo.Constraint(rule=glpk_boiler_size)

            b_netw = p.network_block["electricitySimple"]

            def glpk_netw_size(m):
                return b_netw.arc_block["node1", "node2"].var_size >= 15

            b_tec.const_netw_size = pyo.Constraint(rule=glpk_netw_size)

        adopthub[interval].solve()

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

    # Check 3: Existing electric boiler in Interval_2
    if request.config.solver == "glpk":
        warnings.warn(
            "GLPK presolve fixes var_size and does not return it to Pyomo, so the "
            "carry-over cannot be verified; run with gurobi for this check."
        )
    else:
        node_block = (
            adopthub["Interval_2"]
            .model["full"]
            .periods["Interval_2"]
            .node_blocks["node2"]
        )
        assert "TestTec_BoilerEl_existing" in node_block.tech_blocks_active

    # COST CHECKS
    assert adopthub["Interval_1"].model["full"].var_npv.value > 0
    assert adopthub["Interval_2"].model["full"].var_npv.value > 0

    # Checks 4-8: carry-over tracking (requires the solver to return var_size)
    if request.config.solver == "glpk":
        warnings.warn(
            "GLPK presolve fixes var_size and does not return it to Pyomo, so the "
            "carry-over tracking cannot be verified; run with gurobi for these checks."
        )
    else:
        tec_json = json.load(
            open(
                path
                / "Case_Interval_2"
                / "Interval_2"
                / "node_data"
                / "node2"
                / "Technologies.json"
            )
        )

        # Check 4: remaining_lifetime written (boiler lifetime=25, step=10 → 15)
        assert "remaining_lifetime" in tec_json
        assert tec_json["remaining_lifetime"]["TestTec_BoilerEl"]["Interval_1"] == 15

        # Check 5: carry_over_sizes written and consistent with existing size
        assert "carry_over_sizes" in tec_json
        boiler_carry_over_size = tec_json["carry_over_sizes"]["TestTec_BoilerEl"][
            "Interval_1"
        ]
        assert boiler_carry_over_size > 0
        assert tec_json["existing"]["TestTec_BoilerEl"] == boiler_carry_over_size

        netw_json = json.load(
            open(path / "Case_Interval_2" / "Interval_2" / "Networks.json")
        )

        # Check 6: network remaining_lifetime (technical_lifetime=100 preferred over lifetime=25, step=10 → 90)
        assert netw_json["remaining_lifetime"]["electricitySimple"]["Interval_1"] == 90

        # Check 7: network carry_over_sizes written
        assert netw_json["carry_over_sizes"]["electricitySimple"]["Interval_1"] > 0

        # Check 8: per-carry_over CSV written for electricitySimple
        assert (
            path
            / "Case_Interval_2"
            / "Interval_2"
            / "network_topology"
            / "existing"
            / "electricitySimple"
            / "size_Interval_1.csv"
        ).exists()

    # Post-processing of the summary file (read_results). These functions read the
    # written Summary.xlsx / h5 files and the JSON tracking, so they are exercised
    # independently of the solver.
    summary_path = request.config.result_folder_path / "Summary.xlsx"
    add_values_to_summary(summary_path)
    add_carry_over_annualization_to_summary(summary_path, path, intervals)

    # Discounting requires a global discount rate; set it temporarily in both intervals'
    # ConfigModel.json (the solves are already done, so this does not change results).
    config_paths = [path / ("Case_" + iv) / "ConfigModel.json" for iv in intervals]
    config_orig = [cp.read_text() for cp in config_paths]
    try:
        for cp in config_paths:
            cfg = json.loads(cp.read_text())
            cfg["economic"]["global_discountrate"]["value"] = 0.05
            cp.write_text(json.dumps(cfg, indent=4))
        add_discounted_cost_to_summary(
            summary_path, path, intervals, intervals_between_years
        )
    finally:
        for cp, orig in zip(config_paths, config_orig):
            cp.write_text(orig)

    # Check 9: annualization and discount columns were added (one row per interval)
    summary = pd.read_excel(summary_path)
    assert len(summary) == len(intervals)
    assert "cost_annualization" in summary.columns
    assert "total_cost_with_carry_over_annualization" in summary.columns
    assert "discounted_total_cost" in summary.columns


def test_full_model_flow_multiyear_extra_feature(request):
    """
    Tests the carry-over lifetime bookkeeping (expiry, decommission reconciliation and
    capex proration) for both technologies and networks.

    These branches operate on the ``carry_over_sizes`` / ``remaining_lifetime`` /
    ``carry_over_capex`` / ``remaining_econ_lifetime`` dicts read from the previous
    interval's JSON, so they are exercised by pre-seeding that tracking state in
    Interval_1 before the transition. This is independent of the solver (GLPK does not
    return ``var_size`` for this small model, so the reconciliation against the solved
    ``_existing`` size fully decommissions the seeded carry_overs).

    Topology / Data: identical to :func:`test_full_model_flow_multiyear_lifetime`,
    with lifetime tracking (intervals_between_years=[10]).

    The following is checked (solver-independent):
    - The expiring boiler carry_over ('I_a', remaining_lifetime=5 < step=10) is dropped.
    - The surviving-but-decommissioned boiler carry_over ('I_b') is dropped as well,
      because the solved existing size is 0 (reconciliation removes it).
    - The network carry_over ('I_a') is likewise not kept.
    Interval_1's files are restored afterwards.
    """
    path = Path("tests/case_study_multiyear")
    tec_path = (
        path
        / "Case_Interval_1"
        / "Interval_1"
        / "node_data"
        / "node2"
        / "Technologies.json"
    )
    netw_path = path / "Case_Interval_1" / "Interval_1" / "Networks.json"
    existing_netw_dir = (
        path
        / "Case_Interval_1"
        / "Interval_1"
        / "network_topology"
        / "existing"
        / "electricitySimple"
    )

    tec_orig = tec_path.read_text()
    netw_orig = netw_path.read_text()

    try:
        # Pre-seed carry_over tracking in Interval_1: two boiler vintages (one expiring,
        # one surviving the lifetime check) and one network vintage.
        tec_json = json.loads(tec_orig)
        tec_json["carry_over_sizes"] = {"TestTec_BoilerEl": {"I_a": 10.0, "I_b": 10.0}}
        tec_json["remaining_lifetime"] = {"TestTec_BoilerEl": {"I_a": 5, "I_b": 20}}
        tec_json["carry_over_capex"] = {"TestTec_BoilerEl": {"I_a": 2.0, "I_b": 3.0}}
        tec_json["remaining_econ_lifetime"] = {
            "TestTec_BoilerEl": {"I_a": 5, "I_b": 15}
        }
        tec_path.write_text(json.dumps(tec_json, indent=4))

        netw_json = json.loads(netw_orig)
        netw_json["carry_over_sizes"] = {"electricitySimple": {"I_a": 10.0}}
        netw_json["remaining_lifetime"] = {"electricitySimple": {"I_a": 20}}
        netw_json["carry_over_capex"] = {"electricitySimple": {"I_a": 1.0}}
        netw_json["remaining_econ_lifetime"] = {"electricitySimple": {"I_a": 15}}
        netw_path.write_text(json.dumps(netw_json, indent=4))

        # Per-vintage arc-size CSV for the seeded network carry_over
        existing_netw_dir.mkdir(parents=True, exist_ok=True)
        arc_matrix = pd.DataFrame(
            0.0, index=["node1", "node2"], columns=["node1", "node2"]
        )
        arc_matrix.loc["node1", "node2"] = 10.0
        arc_matrix.index.name = ""
        arc_matrix.to_csv(
            existing_netw_dir / "size_I_a.csv",
            sep=";",
            decimal=".",
            float_format="%.6f",
        )

        adopthub = {}
        intervals = ["Interval_1", "Interval_2"]
        intervals_between_years = [10]

        for i, interval in enumerate(intervals):
            path_interval = path / ("Case_" + interval)

            if i != 0:
                installed_capacities_existing(
                    adopthub,
                    interval,
                    intervals[i - 1],
                    path_interval,
                    intervals_between_years,
                    i,
                )

            adopthub[interval] = ModelHub()
            adopthub[interval].read_data(path_interval, start_period=0, end_period=1)
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

            adopthub[interval].construct_model()
            adopthub[interval].construct_balances()

            # Force sizes in Interval_1 (as in the other multiyear tests) so the boiler
            # new-build size is 15 with gurobi, matching the committed Interval_2 output.
            if interval == "Interval_1":
                p = adopthub[interval].model["full"].periods[interval]
                b_tec = p.node_blocks["node2"].tech_blocks_active["TestTec_BoilerEl"]

                def glpk_boiler_size(m):
                    return b_tec.var_size >= 15

                b_tec.const_boiler_size = pyo.Constraint(rule=glpk_boiler_size)

                b_netw = p.network_block["electricitySimple"]

                def glpk_netw_size(m):
                    return b_netw.arc_block["node1", "node2"].var_size >= 15

                b_tec.const_netw_size = pyo.Constraint(rule=glpk_netw_size)

            adopthub[interval].solve()

        # The seeded carry_overs are expired / decommissioned at the transition. The
        # expiring vintage ('I_a') is dropped by the lifetime check; the surviving one
        # ('I_b') is removed by the decommission reconciliation (the seeded vintages have
        # no ``_existing`` block, so the kept size is 0). These facts hold for any solver.
        tec_json_out = json.load(
            open(
                path
                / "Case_Interval_2"
                / "Interval_2"
                / "node_data"
                / "node2"
                / "Technologies.json"
            )
        )
        boiler_carry_overs = tec_json_out.get("carry_over_sizes", {}).get(
            "TestTec_BoilerEl", {}
        )
        assert "I_a" not in boiler_carry_overs
        assert "I_b" not in boiler_carry_overs

        netw_json_out = json.load(
            open(path / "Case_Interval_2" / "Interval_2" / "Networks.json")
        )
        netw_carry_overs = netw_json_out.get("carry_over_sizes", {}).get(
            "electricitySimple", {}
        )
        assert "I_a" not in netw_carry_overs
    finally:
        tec_path.write_text(tec_orig)
        netw_path.write_text(netw_orig)
        (existing_netw_dir / "size_I_a.csv").unlink(missing_ok=True)


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
