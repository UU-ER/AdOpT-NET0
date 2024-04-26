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

from src.test.utilities import create_patched_datahandle
from src.components.technologies.technology import Technology
from src.energyhub import EnergyHub
from src.model_construction.construct_balances import *


def construct_model(dh):

    ehub = EnergyHub()
    ehub.data = dh
    ehub.construct_model()
    m = ehub.model["full"]

    return m


def solve_model(m):
    solver = SolverFactory("glpk")
    solution = solver.solve(m)
    termination_condition = solution.solver.termination_condition
    return termination_condition


def test_model_nodal_energy_balance():
    """
    Construct a mock technology model for testing

    :param Technology tec: Technology object.
    :param int nr_timesteps: Number of timesteps to create climate data for
    :return ConcreteModel m: Pyomo Concrete Model
    """
    nr_timesteps = 1

    dh = create_patched_datahandle(nr_timesteps)
    config = {"energybalance": {"violation": {"value": 0}}}
    period = dh.topology["investment_periods"][0]
    node = dh.topology["nodes"][0]
    carrier = dh.topology["carriers"][0]

    # INFEASIBILITY CASE
    dh.time_series["full"].loc[:, (period, node, "CarrierData", carrier, "Demand")] = 1

    m = construct_model(dh)
    m = construct_network_constraints(m)
    m = construct_nodal_energybalance(m, config)

    termination_condition = solve_model(m)

    assert termination_condition == TerminationCondition.infeasible

    # FEASIBILITY CASE
    # Through violation
    config["energybalance"]["violation"]["value"] = 1
    m = construct_model(dh)
    m = construct_network_constraints(m)
    m = construct_nodal_energybalance(m, config)

    termination_condition = solve_model(m)

    assert termination_condition == TerminationCondition.optimal
    assert m.periods[period].var_violation[1, carrier, node].value == 1

    # Through import
    config["energybalance"]["violation"]["value"] = 0
    dh.time_series["full"].loc[
        :, (period, node, "CarrierData", carrier, "Import limit")
    ] = 1
    m = construct_model(dh)
    m = construct_network_constraints(m)
    m = construct_nodal_energybalance(m, config)

    termination_condition = solve_model(m)

    assert termination_condition == TerminationCondition.optimal
    assert m.periods[period].node_blocks[node].var_import_flow[1, carrier].value == 1


# TODO: Test global energy balance
# Todo: Reduce number of nodes and periods to 1
