import pyomo.environ as pyo

from tests.utilities import make_testing_modelhub, run_model, define_network, construct_netw_model
from adopt_net0.core.data_preprocessing import create_empty_network_matrix
from adopt_net0.core.registries import NetworkRegistry
from adopt_net0.core.components.networks import *


def test_network_unidirectional(request):
    """
    Tests a network that can only transport in one direction

    INFEASIBILITY CASES
    1) flow in both directions is constraint to 1

    FEASIBILITY CASES
    2) flow in one direction is constraint to 1, checked that size in both directions
    is correct
    """
    nr_timesteps = 1
    modelhub = make_testing_modelhub(nr_timesteps, 2)
    network_registry = NetworkRegistry()
    network_registry.register("SIMPLE", Simple)
    netw = define_network(
        request.config.network_data_folder_path,
        "Simple",
        modelhub,
        network_registry,
        bidirectional_network=True,
        energyconsumption=False,
    )

    # INFEASIBILITY CASE
    m = construct_netw_model(netw, modelhub, nr_timesteps)
    m.test_const_outflow1 = pyo.Constraint(
        expr=m.var_inflow[1, "electricity", "node1"] == 1
    )
    m.test_const_outflow2 = pyo.Constraint(
        expr=m.var_inflow[1, "electricity", "node2"] == 1
    )
    termination = run_model(m, request.config.solver, objective="capex")
    assert termination in [
        pyo.TerminationCondition.infeasibleOrUnbounded,
        pyo.TerminationCondition.infeasible,
    ]

    # FEASIBILITY CASE
    m = construct_netw_model(netw, modelhub, nr_timesteps)
    m.test_const_outflow1 = pyo.Constraint(
        expr=m.var_inflow[1, "electricity", "node1"] == 1
    )

    termination = run_model(m, request.config.solver, objective="capex")
    assert termination == pyo.TerminationCondition.optimal
    assert round(m.arc_block["node2", "node1"].var_size.value, 3) == round(
        m.arc_block["node1", "node2"].var_size.value, 3
    )
    assert m.var_capex.value > 0

    # Size_min =! 0 CASE
    netw.size_min = 100
    m = construct_netw_model(netw, modelhub, nr_timesteps)
    m.test_const_outflow1 = pyo.Constraint(
        expr=m.var_inflow[1, "electricity", "node1"] == 0
    )

    termination = run_model(m, request.config.solver, objective="capex")
    assert termination == pyo.TerminationCondition.optimal
    assert m.arc_block["node2", "node1"].var_flow[1].value <= netw.size_min


def test_network_bidirectional(request):
    """
    Tests a network that can transport in two directions

    FEASIBILITY CASES
    1) flow in one direction is constraint to 1, in the other direction to 2, checked
    that size in both directions is different.
    """
    nr_timesteps = 1
    modelhub = make_testing_modelhub(nr_timesteps, 2)
    network_registry = NetworkRegistry()
    network_registry.register("SIMPLE", Simple)
    netw = define_network(
        request.config.network_data_folder_path,
        "Simple",
        modelhub,
        network_registry,
        bidirectional_network=False,
        energyconsumption=False,
    )

    # FEASIBILITY CASE
    m = construct_netw_model(netw, modelhub, nr_timesteps)
    m.test_const_outflow1 = pyo.Constraint(
        expr=m.var_inflow[1, "electricity", "node1"] == 1
    )
    m.test_const_outflow2 = pyo.Constraint(
        expr=m.var_inflow[1, "electricity", "node2"] == 2
    )
    termination = run_model(m, request.config.solver, objective="capex")
    assert termination == pyo.TerminationCondition.optimal
    assert round(m.arc_block["node2", "node1"].var_size.value, 3) >= 1
    assert round(m.arc_block["node2", "node1"].var_size.value, 3) <= 2
    assert round(m.arc_block["node1", "node2"].var_size.value, 3) >= 2
    assert round(m.arc_block["node1", "node2"].var_size.value, 3) <= 3
    assert m.var_capex.value > 0

def test_network_connection(request):
    """
    Tests a network that can only transport in one direction and it is a simple connection

    INFEASIBILITY CASES
    1) flow in both directions is constraint to 1

    FEASIBILITY CASES
    2) flow in one direction is constraint to 1, checked that size in both directions
    is correct
    """
    nr_timesteps = 1
    modelhub = make_testing_modelhub(nr_timesteps, 2)
    network_registry = NetworkRegistry()
    network_registry.register("SIMPLE", Simple)
    netw = define_network(
        request.config.network_data_folder_path,
        "Simple",
        modelhub,
        network_registry,
        bidirectional_network=True
    )

    # INFEASIBILITY CASE
    m = construct_netw_model(netw, modelhub, nr_timesteps)
    m.test_const_outflow1 = pyo.Constraint(
        expr=m.var_inflow[1, "electricity", "node1"] == 1
    )
    m.test_const_outflow2 = pyo.Constraint(
        expr=m.var_inflow[1, "electricity", "node2"] == 1
    )
    termination = run_model(m, request.config.solver, objective="capex")
    assert termination in [
        pyo.TerminationCondition.infeasibleOrUnbounded,
        pyo.TerminationCondition.infeasible,
    ]

    # FEASIBILITY CASE
    m = construct_netw_model(netw, modelhub, nr_timesteps)
    m.test_const_outflow1 = pyo.Constraint(
        expr=m.var_inflow[1, "electricity", "node1"] == 1
    )

    termination = run_model(m, request.config.solver, objective="capex")
    assert termination == pyo.TerminationCondition.optimal
    assert round(m.arc_block["node2", "node1"].var_size.value, 3) == round(
        m.arc_block["node1", "node2"].var_size.value, 3
    )
    assert m.var_capex.value > 0


def test_network_decommission(request):
    """
    Tests a network that can only transport in one direction and it is a simple connection

    INFEASIBILITY CASES
    1) flow in both directions is constraint to 1

    FEASIBILITY CASES
    2) flow in one direction is constraint to 1, checked that size in both directions
    is correct
    """
    nr_timesteps = 1
    modelhub = make_testing_modelhub(nr_timesteps, 2)


    size_initial = create_empty_network_matrix(modelhub.data["topology"]["nodes"].keys())
    size_initial.loc["node2", "node1"] = 10
    size_initial.loc["node1", "node2"] = 10

    network_registry = NetworkRegistry()
    network_registry.register("SIMPLE", Simple)
    netw = define_network(
        request.config.network_data_folder_path,
        "Simple",
        modelhub,
        network_registry,
        existing=1,
        size_initial=size_initial,
        decommission="impossible",
    )

    m = construct_netw_model(netw, modelhub, nr_timesteps)

    # run model
    run_model(m, request.config.solver, objective="capex")

    # No decommissioning
    assert (
        m.arc_block["node1", "node2"].var_size.value
        == size_initial.loc["node1", "node2"]
    )

    # Technology can decommission
    netw = define_network(
        request.config.network_data_folder_path,
        "Simple",
        modelhub,
        network_registry,
        existing=1,
        size_initial=size_initial,
        decommission="continuous",
    )

    m = construct_netw_model(netw, modelhub, nr_timesteps)

    m.test_const_size_zero = pyo.Constraint(
        expr=m.arc_block["node1", "node2"].var_size == 5
    )
    termination = run_model(m, request.config.solver, objective="capex")

    assert termination in [pyo.TerminationCondition.optimal]

    # Only complete decommissioning
    netw = define_network(
        request.config.network_data_folder_path,
        "Simple",
        modelhub,
        network_registry,
        existing=1,
        size_initial=size_initial,
        decommission="only_complete",
    )

    m = construct_netw_model(netw, modelhub, nr_timesteps)

    m.test_const_size_zero = pyo.Constraint(
        expr=m.arc_block["node1", "node2"].var_size == 5
    )
    termination = run_model(m, request.config.solver, objective="capex")

    assert termination in [
        pyo.TerminationCondition.infeasibleOrUnbounded,
        pyo.TerminationCondition.infeasible,
    ]
