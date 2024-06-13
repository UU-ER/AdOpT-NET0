from pathlib import Path
import json
import pyomo.environ as pyo

from adopt_net0.components.networks import Network
from adopt_net0.test.utilities import make_data_for_testing, run_model
from adopt_net0.data_preprocessing.template_creation import create_empty_network_matrix
from adopt_net0.components.utilities import perform_disjunct_relaxation


def define_network(
    load_path: Path, bidirectional: bool = False, energyconsumption: bool = False
):
    """
    reads TestNetwork from path and creates network object

    :param Path load_path:
    :param bool bidirectional:
    :param bool energyconsumption:
    :return: Network object
    """
    with open(load_path / ("TestNetwork.json")) as json_file:
        netw_data = json.load(json_file)

    netw_data["name"] = "TestNetwork"

    if bidirectional:
        netw_data["Performance"]["bidirectional"] = 1
        netw_data["Performance"]["bidirectional_precise"] = 1
    else:
        netw_data["Performance"]["bidirectional"] = 0

    if not energyconsumption:
        netw_data["Performance"]["energyconsumption"] = {}

    netw_data = Network(netw_data)

    return netw_data


def construct_netw_model(
    netw,
    nr_timesteps: int,
):
    """
    Construct a mock network model for testing

    :param netw: Network object.
    :param int nr_timesteps: Number of timesteps to create network model for
    :return: Pyomo Model
    """

    data = make_data_for_testing(nr_timesteps)

    netw_matrix = create_empty_network_matrix(data["topology"]["nodes"])
    netw_matrix.loc["node1"]["node2"] = 1
    netw_matrix.loc["node2"]["node1"] = 1

    netw.connection = netw_matrix
    netw.distance = netw_matrix
    netw.size_max_arcs = netw_matrix * 10
    netw.fit_network_performance()

    m = pyo.ConcreteModel()
    m.set_t = pyo.Set(initialize=list(range(1, nr_timesteps + 1)))
    m.set_t_full = pyo.Set(initialize=list(range(1, nr_timesteps + 1)))
    m.set_nodes = pyo.Set(initialize=data["topology"]["nodes"])

    m = netw.construct_netw_model(m, data, m.set_nodes, m.set_t, m.set_t_full)
    if netw.big_m_transformation_required:
        m = perform_disjunct_relaxation(m)

    return m


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
    netw = define_network(
        request.config.network_data_folder_path,
        bidirectional=True,
        energyconsumption=False,
    )

    # INFEASIBILITY CASE
    m = construct_netw_model(netw, nr_timesteps)
    m.test_const_outflow1 = pyo.Constraint(
        expr=m.var_inflow[1, "hydrogen", "node1"] == 1
    )
    m.test_const_outflow2 = pyo.Constraint(
        expr=m.var_inflow[1, "hydrogen", "node2"] == 1
    )
    termination = run_model(m, request.config.solver, objective="capex")
    assert termination in [
        pyo.TerminationCondition.infeasibleOrUnbounded,
        pyo.TerminationCondition.infeasible,
    ]

    # FEASIBILITY CASE
    m = construct_netw_model(netw, nr_timesteps)
    m.test_const_outflow1 = pyo.Constraint(
        expr=m.var_inflow[1, "hydrogen", "node1"] == 1
    )

    termination = run_model(m, request.config.solver, objective="capex")
    assert termination == pyo.TerminationCondition.optimal
    assert round(m.arc_block["node2", "node1"].var_size.value, 3) == round(
        m.arc_block["node1", "node2"].var_size.value, 3
    )
    assert m.var_capex.value > 0


def test_network_bidirectional(request):
    """
    Tests a network that can transport in two directions

    FEASIBILITY CASES
    1) flow in one direction is constraint to 1, in the other direction to 2, checked
    that size in both directions is different.
    """
    nr_timesteps = 1
    netw = define_network(
        request.config.network_data_folder_path,
        bidirectional=False,
        energyconsumption=False,
    )

    # FEASIBILITY CASE
    m = construct_netw_model(netw, nr_timesteps)
    m.test_const_outflow1 = pyo.Constraint(
        expr=m.var_inflow[1, "hydrogen", "node1"] == 1
    )
    m.test_const_outflow2 = pyo.Constraint(
        expr=m.var_inflow[1, "hydrogen", "node2"] == 2
    )
    termination = run_model(m, request.config.solver, objective="capex")
    assert termination == pyo.TerminationCondition.optimal
    assert round(m.arc_block["node2", "node1"].var_size.value, 3) >= 1
    assert round(m.arc_block["node2", "node1"].var_size.value, 3) <= 2
    assert round(m.arc_block["node1", "node2"].var_size.value, 3) >= 2
    assert round(m.arc_block["node1", "node2"].var_size.value, 3) <= 3
    assert m.var_capex.value > 0


def test_network_energyconsumption(request):
    """
    Tests a network with an energy consumption

    INFEASIBILITY CASES
    1) constrains the flow in one direction to 1 and the energy consumption to 0

    FEASIBILITY CASES
    2) constrains the flow in one direction to 1 and checks if energy consumption is
    larger 0
    """
    nr_timesteps = 1
    netw = define_network(
        request.config.network_data_folder_path,
        bidirectional=True,
        energyconsumption=True,
    )

    # INFEASIBILITY CASE
    m = construct_netw_model(netw, nr_timesteps)
    m.test_const_outflow1 = pyo.Constraint(
        expr=m.var_inflow[1, "hydrogen", "node1"] == 1
    )
    m.test_const_econs = pyo.Constraint(
        expr=m.var_consumption[1, "electricity", "node2"] == 0
    )

    termination = run_model(m, request.config.solver, objective="capex")
    assert termination in [
        pyo.TerminationCondition.infeasibleOrUnbounded,
        pyo.TerminationCondition.infeasible,
    ]

    # FEASIBILITY CASE
    m = construct_netw_model(netw, nr_timesteps)
    m.test_const_outflow1 = pyo.Constraint(
        expr=m.var_inflow[1, "hydrogen", "node1"] == 1
    )

    termination = run_model(m, request.config.solver, objective="capex")
    assert termination == pyo.TerminationCondition.optimal
    assert m.var_consumption[1, "electricity", "node2"].value > 0
