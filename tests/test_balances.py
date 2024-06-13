from pyomo.environ import (
    ConcreteModel,
    Constraint,
    TerminationCondition,
    SolverFactory,
)

from tests.utilities import make_data_handle
from adopt_net0.modelhub import ModelHub
from adopt_net0.model_construction.construct_balances import (
    construct_global_balance,
    construct_global_energybalance,
    construct_emission_balance,
    construct_nodal_energybalance,
    construct_network_constraints,
    construct_system_cost,
)
from adopt_net0.data_management import DataHandle


def construct_model(dh):
    """
    Constructs a model from DataHandle object

    :param dh: DataHandle
    :return: pyomo model
    """

    ehub = ModelHub()
    ehub.data = dh
    ehub.construct_model()
    m = ehub.model["full"]

    return m


def solve_model(m):
    """
    Solves model and returns termination condition

    :param m: pyomo model to solve
    :return: termination condition
    """
    solver = SolverFactory("gurobi")
    solution = solver.solve(m)
    termination_condition = solution.solver.termination_condition
    return termination_condition


def test_model_nodal_energy_balance():
    """
    Tests the energybalance on a nodal level

    This testing function contains three subtests:

    INFEASIBILITY CASES
    1) Demand at first node is 1, with no imports and exports

    FEASIBILITY CASES
    2) Demand at first node is 1, violation is allowed
    3) Demand at first node is 1, import is allowed
    """
    nr_timesteps = 1

    dh = make_data_handle(nr_timesteps)
    config = {
        "energybalance": {"violation": {"value": 0}},
        "optimization": {"typicaldays": {"N": {"value": 0}}},
    }
    period = dh.topology["investment_periods"][0]
    node = dh.topology["nodes"][0]
    carrier = dh.topology["carriers"][0]

    # INFEASIBILITY CASE
    dh.time_series["full"].loc[:, (period, node, "CarrierData", carrier, "Demand")] = 1

    m = construct_model(dh)
    m = construct_network_constraints(m, config)
    m = construct_nodal_energybalance(m, config)

    termination_condition = solve_model(m)

    assert termination_condition == TerminationCondition.infeasible

    # FEASIBILITY CASE
    # Through violation
    config["energybalance"]["violation"]["value"] = 1
    m = construct_model(dh)
    m = construct_network_constraints(m, config)
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
    m = construct_network_constraints(m, config)
    m = construct_nodal_energybalance(m, config)

    termination_condition = solve_model(m)

    assert termination_condition == TerminationCondition.optimal
    assert m.periods[period].node_blocks[node].var_import_flow[1, carrier].value == 1


def test_model_global_energy_balance():
    """
    Tests the energybalance on a global level

    This testing function contains three subtests:

    INFEASIBILITY CASES
    1) Demand at first node is 1, with no imports and exports

    FEASIBILITY CASES
    2) Demand at first node is 1, violation is allowed
    3) Demand at first node is 1, import is allowed at node 2
    """
    nr_timesteps = 1

    dh = make_data_handle(nr_timesteps)
    config = {
        "energybalance": {"violation": {"value": 0}},
        "optimization": {"typicaldays": {"N": {"value": 0}}},
    }
    period = dh.topology["investment_periods"][0]
    node1 = dh.topology["nodes"][0]
    node2 = dh.topology["nodes"][1]
    carrier = dh.topology["carriers"][0]

    # INFEASIBILITY CASE
    dh.time_series["full"].loc[:, (period, node1, "CarrierData", carrier, "Demand")] = 1

    m = construct_model(dh)
    m = construct_network_constraints(m, config)
    m = construct_global_energybalance(m, config)

    termination_condition = solve_model(m)

    assert termination_condition == TerminationCondition.infeasible

    # FEASIBILITY CASE
    # Through violation
    config["energybalance"]["violation"]["value"] = 1
    m = construct_model(dh)
    m = construct_network_constraints(m, config)
    m = construct_global_energybalance(m, config)

    termination_condition = solve_model(m)

    assert termination_condition == TerminationCondition.optimal
    assert m.periods[period].var_violation[1, carrier, node1].value == 1

    # Through import at other node
    config["energybalance"]["violation"]["value"] = 0
    dh.time_series["full"].loc[
        :, (period, node2, "CarrierData", carrier, "Import limit")
    ] = 1
    m = construct_model(dh)
    m = construct_network_constraints(m, config)
    m = construct_global_energybalance(m, config)

    termination_condition = solve_model(m)

    assert termination_condition == TerminationCondition.optimal
    assert m.periods[period].node_blocks[node2].var_import_flow[1, carrier].value == 1


def test_model_emission_balance():
    """
    Tests the emission balance

    This testing function contains two subtests:

    INFEASIBILITY CASES
    1) Demand at first node is 1, import is allowed at an import emission factor.
    Total emissions are set to zero.

    FEASIBILITY CASES
    2) Demand at first node is 1, import is allowed at an import emission factor.
    """
    nr_timesteps = 1

    dh = make_data_handle(nr_timesteps)
    config = {
        "energybalance": {"violation": {"value": 0}, "copperplate": {"value": 0}},
        "optimization": {"typicaldays": {"N": {"value": 0}}},
    }
    period = dh.topology["investment_periods"][0]
    node = dh.topology["nodes"][0]
    carrier = dh.topology["carriers"][0]

    # INFEASIBILITY CASE
    dh.time_series["full"].loc[:, (period, node, "CarrierData", carrier, "Demand")] = 1
    dh.time_series["full"].loc[
        :, (period, node, "CarrierData", carrier, "Import limit")
    ] = 1
    dh.time_series["full"].loc[
        :, (period, node, "CarrierData", carrier, "Import emission factor")
    ] = 1

    m = construct_model(dh)
    m = construct_network_constraints(m, config)
    m = construct_nodal_energybalance(m, config)
    m = construct_emission_balance(m, dh)

    def init_emissions_to_zero(const, period):
        return m.periods[period].var_emissions_net == 0

    m.test_const_emissions_to_zero = Constraint(
        m.set_periods, rule=init_emissions_to_zero
    )

    termination_condition = solve_model(m)

    assert termination_condition == TerminationCondition.infeasible

    # FEASIBILITY CASE
    m = construct_model(dh)
    m = construct_network_constraints(m, config)
    m = construct_nodal_energybalance(m, config)
    m = construct_emission_balance(m, dh)

    termination_condition = solve_model(m)

    assert termination_condition == TerminationCondition.optimal
    assert m.periods[period].var_emissions_net.value == 1
    assert m.periods[period].var_emissions_pos.value == 1
    assert m.periods[period].var_emissions_neg.value == 0


def test_model_cost_balance():
    """
    Tests the cost balance balance

    This testing function contains two subtests:

    INFEASIBILITY CASES
    1) Demand at first node is 1, import is allowed at an import price. Total cost
    is set to zero.

    FEASIBILITY CASES
    2) Demand at first node is 1, import is allowed at an import price.
    """
    nr_timesteps = 1

    dh = make_data_handle(nr_timesteps)
    config = {
        "energybalance": {"violation": {"value": 0}, "copperplate": {"value": 0}},
        "optimization": {"typicaldays": {"N": {"value": 0}}},
    }
    period = dh.topology["investment_periods"][0]
    node = dh.topology["nodes"][0]
    carrier = dh.topology["carriers"][0]

    # INFEASIBILITY CASE
    dh.time_series["full"].loc[:, (period, node, "CarrierData", carrier, "Demand")] = 1
    dh.time_series["full"].loc[
        :, (period, node, "CarrierData", carrier, "Import limit")
    ] = 1
    dh.time_series["full"].loc[
        :, (period, node, "CarrierData", carrier, "Import price")
    ] = 1

    m = construct_model(dh)
    m = construct_network_constraints(m, config)
    m = construct_global_energybalance(m, config)
    m = construct_system_cost(m, dh)
    m = construct_global_balance(m)

    m.test_const_system_costs = Constraint(expr=m.var_npv == 0)

    termination_condition = solve_model(m)
    assert termination_condition == TerminationCondition.infeasible

    # FEASIBILITY CASE
    m = construct_model(dh)
    m = construct_network_constraints(m, config)
    m = construct_global_energybalance(m, config)
    m = construct_system_cost(m, dh)
    m = construct_global_balance(m)

    termination_condition = solve_model(m)
    assert termination_condition == TerminationCondition.optimal
    assert m.periods[period].node_blocks[node].var_import_flow[1, carrier].value == 1
    assert m.periods[period].var_cost_imports.value == 1
    assert m.var_npv.value == 1
