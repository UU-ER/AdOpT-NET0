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
import json

from src.test.utilities import make_climate_data, make_data_for_technology_testing
from src.components.technologies.technology import Technology
from src.data_management.utilities import open_json, select_technology


def define_technology(tec_name: str, nr_timesteps: int) -> Technology:
    """
    Reads technology data and fits it

    :param str tec_name: name of the technology.
    :param int nr_timesteps: Number of timesteps to create climate data for
    :return Technology tec: Technology class
    """
    # Technology Class Creation
    load_path = Path("./technology_data/")
    with open(load_path / (tec_name + ".json")) as json_file:
        tec = json.load(json_file)
    tec["name"] = tec_name
    tec = select_technology(tec)

    # Technology fitting
    climate_data = make_climate_data("2022-01-01 12:00", nr_timesteps)
    location = {}
    location["lon"] = 10
    location["lat"] = 52
    location["alt"] = 0
    tec.fit_technology_performance(climate_data, location)

    return tec


def construct_tec_model(tec: Technology, nr_timesteps: int) -> ConcreteModel:
    """
    Construct a mock technology model for testing

    :param Technology tec: Technology object.
    :param int nr_timesteps: Number of timesteps to create climate data for
    :return ConcreteModel m: Pyomo Concrete Model
    """

    m = ConcreteModel()
    m.set_t = Set(initialize=list(range(1, nr_timesteps + 1)))
    m.set_t_full = Set(initialize=list(range(1, nr_timesteps + 1)))
    data = make_data_for_technology_testing(nr_timesteps)

    m = tec.construct_tech_model(m, data, m.set_t, m.set_t_full)

    return m


def run_with_output_constraint(
    model: ConcreteModel, demand: float
) -> TerminationCondition:
    """
    Standard test solve where output == demand

    :param ConcreteModel model: Pyomo model
    :return TerminationCondition: Pyomo Termination Condition
    """

    def init_output_constraint(const, t, car):
        return model.var_output[t, car] == demand

    model.test_const_output = Constraint(
        model.set_t, model.set_output_carriers, rule=init_output_constraint
    )

    model.obj = Objective(expr=model.var_capex, sense=minimize)
    solver = SolverFactory("glpk")
    solution = solver.solve(model)

    return solution.solver.termination_condition


@pytest.mark.technologies
def test_res_pv(request):
    """
    tests res technology
    """
    root_folder = request.config.root_folder_path

    time_steps = 1
    technology = "TestTec_ResPhotovoltaic"
    tec = define_technology(technology, time_steps)

    # Technology Model
    model = construct_tec_model(tec, time_steps)

    # INFEASIBILITY CASES
    termination = run_with_output_constraint(
        model, tec.size_max * 1.1 * tec.fitted_performance.rated_power
    )
    assert termination == TerminationCondition.infeasible

    # FEASIBILITY CASES
    termination = run_with_output_constraint(model, 1)
    assert termination == TerminationCondition.optimal


@pytest.mark.technologies
def test_res_wt():
    """
    tests res technology
    """
    time_steps = 1
    technology = "TestTec_WindTurbine"
    tec = define_technology(technology, time_steps)

    # Technology Model
    model = construct_tec_model(tec, time_steps)

    # INFEASIBILITY CASES
    termination = run_with_output_constraint(
        model, tec.size_max * 1.1 * tec.fitted_performance.rated_power
    )
    assert termination == TerminationCondition.infeasible

    # FEASIBILITY CASES
    termination = run_with_output_constraint(model, 1)
    assert termination == TerminationCondition.optimal
