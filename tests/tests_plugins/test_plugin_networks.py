import pyomo.environ as pyo

from adopt_net0.plugins.modeling_plugins.custom_networks import *
from adopt_net0.core.registries import NetworkRegistry
from adopt_net0.plugins.plugin_manager import PluginManager
from adopt_net0.plugins.hooks import Hook
from tests.utilities import (
    run_model,
    make_testing_modelhub, define_network, construct_netw_model
)

def test_plugin_custom_networks(request):
    network_registry = NetworkRegistry()
    network_registry.register_builtin_networks()

    pm = PluginManager()
    plugin_list = {"modeling_plugins.custom_networks":
      {
        "config":
        {
          "networks": ["FLUID"]
        }
      }
    }

    pm.register(plugin_list)
    pm.emit(Hook.NETWORK_REGISTRATION, network_registry=network_registry)


def test_network_fluid(request):
    """
    Tests a network with an energy consumption
    TODO: Make sure fluid is tested completly

    INFEASIBILITY CASES
    1) constrains the flow in one direction to 1 and the energy consumption to 0

    FEASIBILITY CASES
    2) constrains the flow in one direction to 1 and checks if energy consumption is
    larger 0
    """
    nr_timesteps = 1
    modelhub = make_testing_modelhub(nr_timesteps, 2)
    network_registry = NetworkRegistry()
    network_registry.register("FLUID", Fluid)

    netw = define_network(
        request.config.network_data_folder_path,
        "Fluid",
        modelhub,
        network_registry,
        bidirectional_network=True,
        energyconsumption=True,
    )

    # INFEASIBILITY CASE
    m = construct_netw_model(netw, modelhub, nr_timesteps)
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
    m = construct_netw_model(netw, modelhub, nr_timesteps)
    m.test_const_outflow1 = pyo.Constraint(
        expr=m.var_inflow[1, "hydrogen", "node1"] == 1
    )

    termination = run_model(m, request.config.solver, objective="capex")
    assert termination == pyo.TerminationCondition.optimal
    assert m.var_consumption[1, "electricity", "node2"].value > 0


def test_network_electricity(request):
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
    network_registry.register("ELECTRICITY", Electricity)

    netw = define_network(
        request.config.network_data_folder_path,
        "Electricity",
        modelhub,
        network_registry,
        bidirectional_network=True,
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

