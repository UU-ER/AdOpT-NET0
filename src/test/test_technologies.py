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
import numpy as np

from src.test.utilities import make_climate_data, make_data_for_technology_testing
from src.components.technologies.technology import Technology
from src.data_management.utilities import open_json, select_technology
from src.components.utilities import annualize
from src.components.utilities import perform_disjunct_relaxation


def define_technology(
    tec_name: str,
    nr_timesteps: int,
    load_path: Path,
    perf_type: int = None,
    CAPEX_model: int = None,
) -> Technology:
    """
    Reads technology data and fits it

    :param str tec_name: name of the technology.
    :param int nr_timesteps: Number of timesteps to create climate data for
    :return Technology tec: Technology class
    """
    # Technology Class Creation
    with open(load_path / (tec_name + ".json")) as json_file:
        tec = json.load(json_file)
    tec["name"] = tec_name

    if perf_type:
        tec["TechnologyPerf"]["performance_function_type"] = perf_type
    if CAPEX_model:
        tec["Economics"]["CAPEX_model"] = CAPEX_model

    tec = select_technology(tec)

    # Technology fitting
    climate_data = make_climate_data("2022-01-01 12:00", nr_timesteps)
    location = {}
    location["lon"] = 5.5
    location["lat"] = 52.5
    location["alt"] = 0
    tec.fit_technology_performance(climate_data, location)

    return tec


def construct_tec_model(
    tec: Technology, nr_timesteps: int, dynamics: int = None
) -> ConcreteModel:
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

    if dynamics:
        data["config"]["performance"]["dynamics"]["value"] = dynamics

    m = tec.construct_tech_model(m, data, m.set_t, m.set_t_full)
    if tec.big_m_transformation_required:
        m = perform_disjunct_relaxation(m)

    return m


def generate_output_constraint(
    model: ConcreteModel, demand: list, output_ratios: dict = None
) -> ConcreteModel:

    def init_output_constraint(const, t, car):
        if output_ratios:
            if isinstance(output_ratios.get(car), dict):
                alpha = output_ratios[car]["alpha1"]
            else:
                alpha = output_ratios[car]
            if isinstance(alpha, list):
                return model.var_output[t, car] >= demand[t - 1] * alpha[0]
            else:
                return model.var_output[t, car] == demand[t - 1] * alpha
        else:
            return model.var_output[t, car] == demand[t - 1]

    model.test_const_output = Constraint(
        model.set_t, model.set_output_carriers, rule=init_output_constraint
    )

    return model


def generate_var_x_constraint(
    model: ConcreteModel,
    var_x: list = None,
) -> ConcreteModel:
    def init_var_x_constraint(const, t):
        return model.var_x[t] == var_x[t - 1]

    model.test_const_var_x = Constraint(model.set_t, rule=init_var_x_constraint)
    return model


def generate_size_constraint(
    model: ConcreteModel, size: float = None, equality_constraint: bool = False
) -> ConcreteModel:
    def init_size_constraint(const):
        if equality_constraint:
            return model.var_size == size
        else:
            return model.var_size <= size

    model.test_const_size = Constraint(rule=init_size_constraint)
    return model


def run_model(model: ConcreteModel) -> TerminationCondition:
    model.obj = Objective(expr=model.var_capex, sense=minimize)
    solver = SolverFactory("gurobi")
    solution = solver.solve(model)

    return solution.solver.termination_condition


def run_with_first_last_step_constraint(
    model: ConcreteModel,
    demand: list,
    output_ratios: dict = None,
) -> ConcreteModel:
    """
    Standard test solve where output >= demand

    :param ConcreteModel model: Pyomo model
    :return TerminationCondition: Pyomo Termination Condition
    """

    def init_output_constraint(const, t, car):
        if t == 1 or t == len(model.set_t):
            if output_ratios:
                if isinstance(output_ratios.get(car), dict):
                    alpha = output_ratios[car]["alpha1"]
                else:
                    alpha = output_ratios[car]
                if isinstance(alpha, list):
                    return model.var_output[t, car] >= demand[t - 1] * alpha[0]
                else:
                    return model.var_output[t, car] == demand[t - 1] * alpha
            else:
                return model.var_output[t, car] == demand[t - 1]
        else:
            return Constraint.Skip

    model.test_const_output = Constraint(
        model.set_t, model.set_output_carriers, rule=init_output_constraint
    )

    return model


def calculate_piecewise_function(x, bp_x, bp_y):
    """
    Calculate the value of a piecewise function at a given point.

    This function evaluates a piecewise linear function at the point x,
    defined by the breakpoints (bp_x) and corresponding values (bp_y).

    Args:
        x (float): The point at which to evaluate the function.
        bp_x (list of float): List of breakpoints defining the intervals.
        bp_y (list of float): List of function values corresponding to the breakpoints.

    Returns:
        float: The value of the piecewise function at the given point x.
    """
    if x <= bp_x[0]:
        return bp_y[0]
    elif x >= bp_x[-1]:
        return bp_y[-1]
    else:
        for i in range(len(bp_x) - 1):
            if bp_x[i] <= x < bp_x[i + 1]:
                return bp_y[i] + ((x - bp_x[i]) / (bp_x[i + 1] - bp_x[i])) * (
                    bp_y[i + 1] - bp_y[i]
                )


@pytest.mark.technologies
def test_res_pv(request):
    """
    tests res technology
    """
    time_steps = 1
    technology = "TestTec_ResPhotovoltaic"
    tec = define_technology(
        technology, time_steps, request.config.technology_data_folder_path
    )

    # Technology Model
    model = construct_tec_model(tec, nr_timesteps=time_steps)

    # INFEASIBILITY CASES
    oversize = (
        np.ones(time_steps) * tec.size_max * 1.1 * tec.fitted_performance.rated_power
    )

    model = generate_output_constraint(model, oversize)
    termination = run_model(model)
    assert termination == TerminationCondition.infeasibleOrUnbounded

    # FEASIBILITY CASES
    model = generate_output_constraint(model, [1])
    termination = run_model(model)
    assert termination == TerminationCondition.optimal


@pytest.mark.technologies
def test_res_wt(request):
    """
    tests res technology
    """
    time_steps = 1
    technology = "TestTec_WindTurbine"
    tec = define_technology(
        technology, time_steps, request.config.technology_data_folder_path
    )

    # Technology Model
    model = construct_tec_model(tec, nr_timesteps=time_steps)

    # INFEASIBILITY CASES
    oversize = (
        np.ones(time_steps) * tec.size_max * 1.1 * tec.fitted_performance.rated_power
    )
    model = generate_output_constraint(model, oversize)
    termination = run_model(model)
    assert termination == TerminationCondition.infeasibleOrUnbounded

    # FEASIBILITY CASES
    model = generate_output_constraint(model, [1])
    termination = run_model(model)
    assert termination == TerminationCondition.optimal


@pytest.mark.technologies
def test_conv_perf(request):
    """
    tests generic conversion technologies
    """

    time_steps = 1

    for conv_type in [1, 2, 3, 4]:
        technology = "TestTec_Conv" + str(conv_type)

        for perf_type in [1, 2, 3]:
            # Technology Model
            tec = define_technology(
                technology,
                time_steps,
                request.config.technology_data_folder_path,
                perf_type=perf_type,
            )
            model = construct_tec_model(tec, nr_timesteps=time_steps)

            if conv_type == 2 or conv_type == 3:
                output_ratios = tec.fitted_performance.coefficients
            elif conv_type == 4:
                output_ratios = tec.performance_data["output_ratios"]
            else:
                output_ratios = None

            if perf_type == 1:
                # INFEASIBILITY CASES
                oversize = (
                    np.ones(time_steps)
                    * tec.size_max
                    * 1.1
                    * tec.fitted_performance.rated_power
                )
                model = generate_output_constraint(model, oversize)
                termination = run_model(model)
                assert termination == TerminationCondition.infeasibleOrUnbounded

                # FEASIBILITY CASES
                model = generate_output_constraint(
                    model, [1], output_ratios=output_ratios
                )
                termination = run_model(model)
                assert termination == TerminationCondition.optimal

                # Check performance for performance type 1
                if conv_type == 1:
                    car_input = sum(
                        model.var_input[car].value for car in model.var_input
                    )
                    car_output = sum(
                        model.var_output[car].value for car in model.var_output
                    )
                    assert round(car_output, 2) == round(
                        tec.fitted_performance.coefficients["out"]["alpha1"]
                        * car_input,
                        2,
                    )
                if conv_type == 2:
                    car_input = sum(
                        model.var_input[car].value for car in model.var_input
                    )
                    for car in model.var_output:
                        car_output = model.var_output[car].value
                        assert round(car_output, 2) == round(
                            tec.fitted_performance.coefficients[car[1]]["alpha1"]
                            * car_input,
                            2,
                        )
                if conv_type == 3:
                    main_car_input = model.var_input[
                        1, tec.performance_data["main_input_carrier"]
                    ].value
                    for car in model.var_input:
                        car_input = model.var_input[car].value
                        assert (
                            car_input
                            == tec.performance_data["input_ratios"][car[1]]
                            * main_car_input
                        )
                    for car in model.var_output:
                        car_output = model.var_output[car].value
                        assert (
                            car_output
                            == tec.fitted_performance.coefficients[car[1]]["alpha1"]
                            * main_car_input
                        )

            elif perf_type == 2:
                # Check minimum load
                minsize = 10
                demand = [
                    minsize * tec.performance_data["min_part_load"] * 0.1
                ] * time_steps
                model = generate_size_constraint(
                    model, minsize, equality_constraint=True
                )
                model = generate_output_constraint(model, demand)
                termination = run_model(model)
                model.pprint()
                assert termination == TerminationCondition.infeasibleOrUnbounded

            elif perf_type == 3 and conv_type != 4:
                # FEASIBILITY CASES
                model = generate_output_constraint(
                    model, [1], output_ratios=output_ratios
                )
                termination = run_model(model)
                assert termination == TerminationCondition.optimal

                # Check performance for performance type 1
                if conv_type == 1:
                    bp_x = tec.fitted_performance.coefficients["out"]["bp_x"]
                    bp_y = tec.fitted_performance.coefficients["out"]["bp_y"]
                    car_input = sum(
                        model.var_input[car].value for car in model.var_input
                    )
                    car_output = sum(
                        model.var_output[car].value for car in model.var_output
                    )
                    assert round(car_output, 2) == round(
                        calculate_piecewise_function(car_input, bp_x, bp_y) * car_input,
                        2,
                    )
                if conv_type == 2:
                    car_input = sum(
                        model.var_input[car].value for car in model.var_input
                    )
                    for car in model.var_output:
                        bp_x = tec.fitted_performance.coefficients[car[1]]["bp_x"]
                        bp_y = tec.fitted_performance.coefficients[car[1]]["bp_y"]
                        car_output = model.var_output[car].value
                        assert round(car_output, 2) == round(
                            calculate_piecewise_function(car_input, bp_x, bp_y)
                            * car_input,
                            2,
                        )
                if conv_type == 3:
                    main_car_input = model.var_input[
                        1, tec.performance_data["main_input_carrier"]
                    ].value
                    for car in model.var_input:
                        car_input = model.var_input[car].value
                        assert round(car_input, 2) == round(
                            tec.performance_data["input_ratios"][car[1]]
                            * main_car_input,
                            2,
                        )
                    for car in model.var_output:
                        bp_x = tec.fitted_performance.coefficients[car[1]]["bp_x"]
                        bp_y = tec.fitted_performance.coefficients[car[1]]["bp_y"]
                        car_output = model.var_output[car].value
                        assert round(car_output, 2) == round(
                            calculate_piecewise_function(main_car_input, bp_x, bp_y)
                            * main_car_input,
                            2,
                        )


def test_conv_CAPEX(request):
    """
    tests CAPEX models
    """

    time_steps = 1

    for capex_model in range(1, 4):
        technology = "TestTec_Conv1"
        tec = define_technology(
            technology,
            time_steps,
            request.config.technology_data_folder_path,
            CAPEX_model=capex_model,
        )

        # Technology Model
        model = construct_tec_model(tec, nr_timesteps=time_steps)
        f = time_steps / 8760
        t = tec.economics.lifetime
        r = tec.economics.discount_rate
        a = annualize(r, t, f)

        # Check CAPEX
        model = generate_output_constraint(model, [1])
        termination = run_model(model)
        assert termination == TerminationCondition.optimal

        if capex_model == 1:
            assert (
                model.var_capex.value
                == tec.economics.capex_data["unit_capex"] * model.var_size.value * a
            )

        if capex_model == 2:
            bp_x = tec.economics.capex_data["piecewise_capex"]["bp_x"]
            bp_y = tec.economics.capex_data["piecewise_capex"]["bp_y"]
            assert round(model.var_capex.value, 4) == round(
                calculate_piecewise_function(model.var_size.value, bp_x, bp_y) * a, 4
            )

        if capex_model == 3:
            assert round(model.var_capex.value, 5) == round(
                (
                    tec.economics.capex_data["unit_capex"] * model.var_size.value
                    + tec.economics.capex_data["fix_capex"]
                )
                * a,
                5,
            )


def test_tec_storage(request):
    """
    tests storage technology
    """

    time_steps = 3
    technology = "TestTec_StorageBattery"
    tec = define_technology(
        technology, time_steps, request.config.technology_data_folder_path
    )

    # Technology Model
    model = construct_tec_model(tec, nr_timesteps=time_steps)

    # INFEASIBILITY CASES
    oversize = (
        np.ones(time_steps) * tec.size_max * 1.1 * tec.fitted_performance.rated_power
    )
    model = generate_output_constraint(model, oversize)
    termination = run_model(model)
    assert termination == TerminationCondition.infeasibleOrUnbounded

    # FEASIBILITY CASES
    demand = [0, 1, 0]

    def init_output_constraint(const, t, car):
        return model.var_output[t, car] == demand[t - 1]

    model.test_const_output = Constraint(
        model.set_t, model.set_output_carriers, rule=init_output_constraint
    )

    model.obj = Objective(expr=model.var_capex, sense=minimize)
    solver = SolverFactory("gurobi")
    solution = solver.solve(model)
    termination = solution.solver.termination_condition

    assert termination == TerminationCondition.optimal


def test_tec_sink(request):

    time_steps = 2
    technology = "TestTec_Sink"
    tec = define_technology(
        technology, time_steps, request.config.technology_data_folder_path
    )

    # Technology Model
    model = construct_tec_model(tec, nr_timesteps=time_steps)

    # INFEASIBILITY CASES
    model.test_const_input = Constraint(expr=model.var_input[1, "CO2captured"] == 2)
    termination = run_model(model)
    assert termination == TerminationCondition.infeasibleOrUnbounded

    # # FEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model.test_const_input = Constraint(expr=model.var_input[2, "CO2captured"] == 1)
    model.test_const_level = Constraint(expr=model.var_storage_level[1] == 0)
    termination = run_model(model)

    assert termination == TerminationCondition.optimal
    assert model.var_storage_level[2].value == 1
    assert model.var_input[2, "electricity"].value == 1
    assert model.var_capex.value > 0

    # demand = [0, 1, 0]
    #
    # def init_output_constraint(const, t, car):
    #     return model.var_output[t, car] == demand[t - 1]
    #
    # model.test_const_output = Constraint(
    #     model.set_t, model.set_output_carriers, rule=init_output_constraint
    # )
    #
    # model.obj = Objective(expr=model.var_capex, sense=minimize)
    # solver = SolverFactory("gurobi")
    # solution = solver.solve(model)
    # termination = solution.solver.termination_condition
    #
    # assert termination == TerminationCondition.optimal


def test_dynamics_fast(request):
    """
    tests dynamic operation
    """

    time_steps = 5

    for conv_type in [1, 2, 3]:
        technology = "TestTec_Conv" + str(conv_type)

        for perf_type in [1, 2, 3]:
            # Technology Model
            tec = define_technology(
                technology,
                time_steps,
                request.config.technology_data_folder_path,
                perf_type=perf_type,
            )

            if conv_type == 1:
                output_ratios = None
            else:
                output_ratios = tec.fitted_performance.coefficients

            if perf_type == 1:
                # Set parameters
                tec.performance_data["ramping_time"] = 4
                tec.performance_data["ref_size"] = 4
                output = [1, 1, 5, 1, 1]
                model = construct_tec_model(tec, nr_timesteps=time_steps)

                model = generate_output_constraint(
                    model, output, output_ratios=output_ratios
                )
                termination = run_model(model)
                assert termination == TerminationCondition.infeasibleOrUnbounded

                # Check minimum load
                minsize = 10
                demand = [
                    minsize * tec.performance_data["min_part_load"] * 0.1
                ] * time_steps

                model = generate_size_constraint(
                    model, minsize, equality_constraint=True
                )
                model = generate_output_constraint(model, demand)
                termination = run_model(model)
                assert termination == TerminationCondition.infeasibleOrUnbounded

            elif perf_type > 1:
                # Set parameters
                tec.performance_data["standby_power"] = 0.2
                tec.performance_data["min_part_load"] = 0.3
                tec.performance_data["max_startups"] = 1
                output = [1, 0, 1, 0.5, 1]
                var_x = [1, 0, 1, 1, 1]
                model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)

                model = generate_output_constraint(
                    model, output, output_ratios=output_ratios
                )
                model = generate_var_x_constraint(model, var_x)
                termination = run_model(model)
                assert termination == TerminationCondition.optimal

                # Check max startups
                assert (
                    sum(model.var_x[t].value for t in model.var_x)
                    >= time_steps - tec.performance_data["max_startups"]
                )

                # Check standbypower
                main_car = tec.performance_data["main_input_carrier"]
                assert round(model.var_input[2, main_car].value, 4) == round(
                    model.var_size.value * tec.performance_data["standby_power"], 4
                )

                # Check SUSD loads
                tec.performance_data["SU_load"] = 0.6
                tec.performance_data["SD_load"] = 0.4
                output = [1, 0, 1, 0.5, 1]
                model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)

                model = generate_output_constraint(
                    model, output, output_ratios=output_ratios
                )
                model = generate_var_x_constraint(model, var_x)
                model = generate_size_constraint(model, 1)
                termination = run_model(model)
                assert termination == TerminationCondition.infeasibleOrUnbounded

                # Check ramping rate with tech on
                tec.performance_data["SU_load"] = 1
                tec.performance_data["SD_load"] = 1
                tec.performance_data["ramping_time"] = 4
                tec.performance_data["ref_size"] = 1
                tec.performance_data["ramping_const_int"] = 1
                output = [1, 0, 1, 0.5, 1]
                model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)

                model = generate_output_constraint(
                    model, output, output_ratios=output_ratios
                )
                model = generate_var_x_constraint(model, var_x)
                model = generate_size_constraint(model, 1)
                termination = run_model(model)
                assert termination == TerminationCondition.infeasibleOrUnbounded


# def test_dynamics_slow(request):
#     """
#     tests dynamic operation
#     """
#
#     time_steps = 7
#
#     for conv_type in [1, 2, 3]:
#         technology = "TestTec_Conv" + str(conv_type)
#
#         # Technology Model
#         perf_type = 4
#         tec = define_technology(technology, time_steps, request.config.technology_data_folder_path, perf_type=perf_type)
#
#         if conv_type == 1:
#             output_ratios = None
#         else:
#             output_ratios = tec.fitted_performance.coefficients
#
#         # change SU time and SD time
#         SU_time = 2
#         SD_time = 1
#         min_part_load = 0.6
#         tec.performance_data["SU_time"] = SU_time
#         tec.performance_data["SD_time"] = SD_time
#         tec.performance_data["min_part_load"] = min_part_load
#         output = [1, 1, 1, 0, 1, 1, 1]
#         var_x = [1, 1, 0, 0, 0, 0, 1]
#         model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)
#
#         model = generate_output_constraint(model, output, output_ratios=output_ratios)
#         model = generate_var_x_constraint(model, var_x)
#         model = generate_size_constraint(model, 1)
#         termination = run_model(model)
#         assert termination == TerminationCondition.infeasibleOrUnbounded
#
#         # # Calculate SU and SD trajectories
#         # SU_trajectory = []
#         # for i in range(1, SU_time + 1):
#         #     SU_trajectory.append((min_part_load / (SU_time + 1)) * i)
#         #
#         # SD_trajectory = []
#         # for i in range(1, SD_time + 1):
#         #     SD_trajectory.append((min_part_load / (SD_time + 1)) * i)
#         # SD_trajectory = sorted(SD_trajectory, reverse=True)
#
#         # output = [1, 0.6, 0.3, 0, 0.2, 0.4, 0.6]
#         var_x_at_t = [4] * time_steps
#         model = generate_var_x_constraint(model, var_x_at_t)
#         model = generate_size_constraint(model, 1)
#         model = run_with_first_last_step_constraint(
#             model, demand=output, output_ratios=output_ratios)
#         termination = run_model(model)
#
#
#         assert termination == TerminationCondition.optimal
