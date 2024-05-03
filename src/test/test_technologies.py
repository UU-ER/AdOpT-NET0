import warnings

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

from src.test.utilities import make_climate_data, make_data_for_testing, run_model
from src.components.technologies.technology import Technology
from src.components.technologies import fit_ccs_data
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
    if tec.ccs:
        ccs_data = open_json(tec.performance_data["ccs"]["ccs_type"], load_path)
        tec.ccs_data = fit_ccs_data(
            tec.performance_data["ccs"]["co2_concentration"], ccs_data, climate_data
        )

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
    data = make_data_for_testing(nr_timesteps)

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
                return model.var_output_tot[t, car] >= demand[t - 1] * alpha[0]
            else:
                return model.var_output_tot[t, car] == demand[t - 1] * alpha
        else:
            return model.var_output_tot[t, car] == demand[t - 1]

    model.test_const_output1 = Constraint(
        model.set_t, model.set_output_carriers, rule=init_output_constraint
    )

    return model


def generate_output_constraint_start_timestep(
    model: ConcreteModel, demand: float, output_ratios: dict = None
) -> ConcreteModel:

    def init_output_constraint(const, car):
        if output_ratios:
            if isinstance(output_ratios.get(car), dict):
                alpha = output_ratios[car]["alpha1"]
            else:
                alpha = output_ratios[car]
            if isinstance(alpha, list):
                return model.var_output_tot[1, car] >= demand * alpha[0]
            else:
                return model.var_output_tot[1, car] == demand * alpha
        else:
            return model.var_output_tot[1, car] == demand

    model.test_const_output2 = Constraint(
        model.set_output_carriers, rule=init_output_constraint
    )

    return model


def generate_output_constraint_end_timestep(
    model: ConcreteModel, demand: float, output_ratios: dict = None
) -> ConcreteModel:

    def init_output_constraint(const, car):
        if output_ratios:
            if isinstance(output_ratios.get(car), dict):
                alpha = output_ratios[car]["alpha1"]
            else:
                alpha = output_ratios[car]
            if isinstance(alpha, list):
                return model.var_output_tot[len(model.set_t), car] >= demand * alpha[0]
            else:
                return model.var_output_tot[len(model.set_t), car] == demand * alpha
        else:
            return model.var_output_tot[len(model.set_t), car] == demand

    model.test_const_output3 = Constraint(
        model.set_output_carriers, rule=init_output_constraint
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


def generate_var_y_constraint(
    model: ConcreteModel,
    var_y: list = None,
) -> ConcreteModel:
    def init_var_y_constraint(const, t):
        return model.var_y[t] == var_y[t - 1]

    model.test_const_var_y = Constraint(model.set_t, rule=init_var_y_constraint)
    return model


def generate_var_z_constraint(
    model: ConcreteModel,
    var_z: list = None,
) -> ConcreteModel:
    def init_var_z_constraint(const, t):
        return model.var_z[t] == var_z[t - 1]

    model.test_const_var_z = Constraint(model.set_t, rule=init_var_z_constraint)
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
                    return model.var_output_tot[t, car] >= demand[t - 1] * alpha[0]
                else:
                    return model.var_output_tot[t, car] == demand[t - 1] * alpha
            else:
                return model.var_output_tot[t, car] == demand[t - 1]
        else:
            return Constraint.Skip

    model.test_const_output4 = Constraint(
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
    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model = generate_output_constraint(model, [1])
    termination = run_model(model, request.config.solver)
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
    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model = generate_output_constraint(model, [1])
    termination = run_model(model, request.config.solver)
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
                termination = run_model(model, request.config.solver)
                assert termination in [
                    TerminationCondition.infeasibleOrUnbounded,
                    TerminationCondition.infeasible,
                    TerminationCondition.other,
                ]

                # FEASIBILITY CASES
                model = generate_output_constraint(
                    model, [1], output_ratios=output_ratios
                )
                termination = run_model(model, request.config.solver)
                assert termination == TerminationCondition.optimal

                # Check performance for performance type 1
                if conv_type == 1:
                    car_input = sum(
                        model.var_input_tot[car].value for car in model.var_input_tot
                    )
                    car_output = sum(
                        model.var_output_tot[car].value for car in model.var_output_tot
                    )
                    assert round(car_output, 2) == round(
                        tec.fitted_performance.coefficients["out"]["alpha1"]
                        * car_input,
                        2,
                    )
                if conv_type == 2:
                    car_input = sum(
                        model.var_input_tot[car].value for car in model.var_input_tot
                    )
                    for car in model.var_output_tot:
                        car_output = model.var_output_tot[car].value
                        assert round(car_output, 2) == round(
                            tec.fitted_performance.coefficients[car[1]]["alpha1"]
                            * car_input,
                            2,
                        )
                if conv_type == 3:
                    main_car_input = model.var_input_tot[
                        1, tec.performance_data["main_input_carrier"]
                    ].value
                    for car in model.var_input_tot:
                        car_input = model.var_input_tot[car].value
                        assert (
                            car_input
                            == tec.performance_data["input_ratios"][car[1]]
                            * main_car_input
                        )
                    for car in model.var_output_tot:
                        car_output = model.var_output_tot[car].value
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
                termination = run_model(model, request.config.solver)
                model.pprint()
                assert termination in [
                    TerminationCondition.infeasibleOrUnbounded,
                    TerminationCondition.infeasible,
                    TerminationCondition.other,
                ]

            elif perf_type == 3 and conv_type != 4:
                # FEASIBILITY CASES
                model = generate_output_constraint(
                    model, [1], output_ratios=output_ratios
                )
                termination = run_model(model, request.config.solver)
                assert termination == TerminationCondition.optimal

                # Check performance for performance type 1
                if conv_type == 1:
                    bp_x = tec.fitted_performance.coefficients["out"]["bp_x"]
                    bp_y = tec.fitted_performance.coefficients["out"]["bp_y"]
                    car_input = sum(
                        model.var_input_tot[car].value for car in model.var_input_tot
                    )
                    car_output = sum(
                        model.var_output_tot[car].value for car in model.var_output_tot
                    )
                    assert round(car_output, 2) == round(
                        calculate_piecewise_function(car_input, bp_x, bp_y) * car_input,
                        2,
                    )
                if conv_type == 2:
                    car_input = sum(
                        model.var_input_tot[car].value for car in model.var_input_tot
                    )
                    for car in model.var_output_tot:
                        bp_x = tec.fitted_performance.coefficients[car[1]]["bp_x"]
                        bp_y = tec.fitted_performance.coefficients[car[1]]["bp_y"]
                        car_output = model.var_output_tot[car].value
                        assert round(car_output, 2) == round(
                            calculate_piecewise_function(car_input, bp_x, bp_y)
                            * car_input,
                            2,
                        )
                if conv_type == 3:
                    main_car_input = model.var_input_tot[
                        1, tec.performance_data["main_input_carrier"]
                    ].value
                    for car in model.var_input_tot:
                        car_input = model.var_input_tot[car].value
                        assert round(car_input, 2) == round(
                            tec.performance_data["input_ratios"][car[1]]
                            * main_car_input,
                            2,
                        )
                    for car in model.var_output_tot:
                        bp_x = tec.fitted_performance.coefficients[car[1]]["bp_x"]
                        bp_y = tec.fitted_performance.coefficients[car[1]]["bp_y"]
                        car_output = model.var_output_tot[car].value
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

        if capex_model == 2 and request.config.solver == "glpk":
            warnings.warn(
                "SOS constraints dont work with glpk, test on local machine"
                " with gurobi"
            )
        else:
            termination = run_model(model, request.config.solver)
            assert termination == TerminationCondition.optimal

            if capex_model == 1:
                assert round(model.var_capex.value, 4) == round(
                    tec.economics.capex_data["unit_capex"] * model.var_size.value * a, 4
                )

            if capex_model == 2:
                bp_x = tec.economics.capex_data["piecewise_capex"]["bp_x"]
                bp_y = tec.economics.capex_data["piecewise_capex"]["bp_y"]
                assert round(model.var_capex.value, 4) == round(
                    calculate_piecewise_function(model.var_size.value, bp_x, bp_y) * a,
                    4,
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
    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)

    def init_output_constraint(const, t, car):
        demand = [0, 1, 0]
        return model.var_output_tot[t, car] == demand[t - 1]

    model.test_const_output5 = Constraint(
        model.set_t, model.set_output_carriers, rule=init_output_constraint
    )

    termination = run_model(model, request.config.solver)

    assert termination == TerminationCondition.optimal
    assert model.var_size.value > 0
    assert model.var_capex_aux.value > 0
    assert sum(model.var_input_tot[t, "electricity"].value for t in model.set_t) >= 1


def test_tec_sink(request):

    time_steps = 2
    technology = "TestTec_Sink"
    tec = define_technology(
        technology, time_steps, request.config.technology_data_folder_path
    )

    # Technology Model
    model = construct_tec_model(tec, nr_timesteps=time_steps)

    # INFEASIBILITY CASES
    model.test_const_input = Constraint(expr=model.var_input_tot[1, "CO2captured"] == 2)
    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # # FEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model.test_const_input = Constraint(expr=model.var_input_tot[2, "CO2captured"] == 1)
    model.test_const_level = Constraint(expr=model.var_storage_level[1] == 0)
    termination = run_model(model, request.config.solver)

    assert termination == TerminationCondition.optimal
    assert model.var_storage_level[2].value == 1
    assert model.var_input_tot[2, "electricity"].value == 1
    assert model.var_capex.value > 0


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
                termination = run_model(model, request.config.solver)
                assert termination in [
                    TerminationCondition.infeasibleOrUnbounded,
                    TerminationCondition.infeasible,
                    TerminationCondition.other,
                ]

                # Check minimum load
                model = construct_tec_model(tec, nr_timesteps=time_steps)
                minsize = 10
                demand = [
                    minsize * tec.performance_data["min_part_load"] * 0.1
                ] * time_steps

                model = generate_size_constraint(
                    model, minsize, equality_constraint=True
                )
                model = generate_output_constraint(model, demand)
                termination = run_model(model, request.config.solver)
                assert termination in [
                    TerminationCondition.infeasibleOrUnbounded,
                    TerminationCondition.infeasible,
                    TerminationCondition.other,
                ]

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
                termination = run_model(model, request.config.solver)
                assert termination == TerminationCondition.optimal

                # Check max startups
                assert (
                    sum(model.var_x[t].value for t in model.var_x)
                    >= time_steps - tec.performance_data["max_startups"]
                )

                # Check standbypower
                main_car = tec.performance_data["main_input_carrier"]
                assert round(model.var_input_tot[2, main_car].value, 4) == round(
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
                termination = run_model(model, request.config.solver)
                assert termination in [
                    TerminationCondition.infeasibleOrUnbounded,
                    TerminationCondition.infeasible,
                    TerminationCondition.other,
                ]

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
                termination = run_model(model, request.config.solver)
                assert termination in [
                    TerminationCondition.infeasibleOrUnbounded,
                    TerminationCondition.infeasible,
                    TerminationCondition.other,
                ]


def test_dynamics_slow(request):
    """
    tests dynamic operation
    """

    time_steps = 5

    for conv_type in [1, 2, 3]:
        technology = "TestTec_Conv" + str(conv_type)

        # Technology Model
        perf_type = 4
        tec = define_technology(
            technology,
            nr_timesteps=time_steps,
            load_path=request.config.technology_data_folder_path,
            perf_type=perf_type,
        )

        if conv_type == 1:
            output_ratios = None
        else:
            output_ratios = tec.fitted_performance.coefficients

        # check SD time
        SD_time = 2
        min_part_load = 0.6
        tec.performance_data["SD_time"] = SD_time
        tec.performance_data["min_part_load"] = min_part_load
        output_start = 0.6
        var_z = [0, 0, 1, 0, 0]
        var_x = [1, 1, 0, 0, 0]
        model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)

        model = generate_var_z_constraint(model, var_z)
        model = generate_var_x_constraint(model, var_x)
        model = generate_output_constraint_start_timestep(
            model, demand=output_start, output_ratios=output_ratios
        )
        termination = run_model(model, request.config.solver)
        assert termination == TerminationCondition.optimal

        main_car = tec.performance_data["main_input_carrier"]
        trajectory = model.var_size.value * min_part_load / (SD_time + 1)

        if conv_type < 3:
            input_at_SD1 = (
                model.var_input_tot[3, "gas"].value
                + model.var_input_tot[3, "hydrogen"].value
            )
            input_at_SD2 = (
                model.var_input_tot[4, "gas"].value
                + model.var_input_tot[4, "hydrogen"].value
            )
        else:
            input_at_SD1 = model.var_input_tot[3, main_car].value
            input_at_SD2 = model.var_input_tot[4, main_car].value

        assert round(input_at_SD2, 3) == round(trajectory, 3)
        assert round(input_at_SD1, 3) == round(trajectory * SD_time, 3)

        # Check infeasibility case
        var_x = [1, 1, 0, 0, 1]
        model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)

        model = generate_var_z_constraint(model, var_z)
        model = generate_var_x_constraint(model, var_x)
        model = generate_output_constraint_start_timestep(model, demand=output_start)
        termination = run_model(model, request.config.solver)
        assert termination in [
            TerminationCondition.infeasibleOrUnbounded,
            TerminationCondition.infeasible,
            TerminationCondition.other,
        ]

        # check SU time
        tec = define_technology(
            technology,
            nr_timesteps=time_steps,
            load_path=request.config.technology_data_folder_path,
            perf_type=perf_type,
        )

        SU_time = 2
        min_part_load = 0.6
        tec.performance_data["SU_time"] = SU_time
        tec.performance_data["min_part_load"] = min_part_load
        output_start = 0.6
        var_y = [0, 0, 0, 1, 0]
        var_x = [0, 0, 0, 1, 1]
        model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)

        model = generate_var_y_constraint(model, var_y)
        model = generate_var_x_constraint(model, var_x)
        model = generate_output_constraint_end_timestep(
            model, demand=output_start, output_ratios=output_ratios
        )
        termination = run_model(model, request.config.solver)
        assert termination == TerminationCondition.optimal

        main_car = tec.performance_data["main_input_carrier"]
        trajectory = model.var_size.value * min_part_load / (SU_time + 1)

        if conv_type < 3:
            input_at_SU1 = (
                model.var_input_tot[2, "gas"].value
                + model.var_input_tot[2, "hydrogen"].value
            )
            input_at_SU2 = (
                model.var_input_tot[3, "gas"].value
                + model.var_input_tot[3, "hydrogen"].value
            )
        else:
            input_at_SU1 = model.var_input_tot[2, main_car].value
            input_at_SU2 = model.var_input_tot[3, main_car].value

        assert round(input_at_SU1, 3) == round(trajectory, 3)
        assert round(input_at_SU2, 3) == round(trajectory * SU_time, 3)

        # Check infeasibility case
        var_x = [0, 0, 1, 1, 1]
        model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)

        model = generate_var_y_constraint(model, var_y)
        model = generate_var_x_constraint(model, var_x)
        model = generate_output_constraint_start_timestep(model, demand=output_start)
        termination = run_model(model, request.config.solver)
        assert termination in [
            TerminationCondition.infeasibleOrUnbounded,
            TerminationCondition.infeasible,
            TerminationCondition.other,
        ]


def test_dac(request):
    """
    tests DAC Adsorption
    """
    time_steps = 1
    technology = "TestTec_DAC_adsorption"
    tec = define_technology(
        technology, time_steps, request.config.technology_data_folder_path
    )

    # INFEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model = generate_output_constraint(model, [1])
    model.test_const_input = Constraint(expr=model.var_input_tot[1, "electricity"] == 0)

    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model = generate_output_constraint(model, [1])

    termination = run_model(model, request.config.solver)
    assert termination == TerminationCondition.optimal
    assert model.var_input_tot[1, "heat"].value > 0.1
    assert model.var_input_tot[1, "electricity"].value > 0.01
    assert model.var_size.value > 1
    assert model.var_capex.value > 0


def test_hydro_open(request):
    """
    tests Open Hydro
    """
    time_steps = 3
    technology = "TestTec_Hydro_Open"
    tec = define_technology(
        technology, time_steps, request.config.technology_data_folder_path
    )

    # INFEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model = generate_output_constraint(model, [1, 1, 1])

    def init_test_input(const, t):
        return model.var_input_tot[t, "electricity"] == 0

    model.test_const_input = Constraint(model.set_t, rule=init_test_input)

    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model = generate_output_constraint(model, [0, 1, 1])

    def init_test_input(const, t):
        return model.var_input_tot[t, "electricity"] == 0

    model.test_const_input = Constraint(model.set_t, rule=init_test_input)

    termination = run_model(model, request.config.solver)
    assert termination == TerminationCondition.optimal
    assert model.var_size.value == 2


def test_heat_pump(request):
    """
    tests Open Hydro
    """
    time_steps = 1
    technology = "TestTec_HeatPump_AirSourced"
    tec = define_technology(
        technology, time_steps, request.config.technology_data_folder_path
    )

    # INFEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model = generate_output_constraint(model, [1])
    model.test_const_input = Constraint(expr=model.var_input_tot[1, "electricity"] == 0)

    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model = generate_output_constraint(model, [1])

    termination = run_model(model, request.config.solver)
    assert termination == TerminationCondition.optimal
    assert model.var_size.value >= 0.1
    assert model.var_input_tot[1, "electricity"].value >= 0.1


def test_gasturbine(request):
    """
    tests Gas Turbine
    """
    time_steps = 1
    technology = "TestTec_GasTurbine_NG_10"
    tec = define_technology(
        technology, time_steps, request.config.technology_data_folder_path
    )

    # INFEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model = generate_output_constraint(model, [9])
    model.test_const_input = Constraint(expr=model.var_input_tot[1, "gas"] == 0)

    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model.test_const_output = Constraint(
        expr=model.var_output_tot[1, "electricity"] == 10
    )
    model.test_const_input = Constraint(expr=model.var_input_tot[1, "hydrogen"] == 0)

    termination = run_model(model, request.config.solver)
    assert termination == TerminationCondition.optimal
    assert model.var_size.value == 1
    assert model.var_input_tot[1, "gas"].value >= 10 / 0.4
    assert model.var_output_tot[1, "heat"].value >= 10 * 0.5


def test_ccs(request):
    time_steps = 1
    technology = "TestTec_Conv1_ccs"
    tec = define_technology(
        technology, time_steps, request.config.technology_data_folder_path
    )

    # INFEASIBILITY CASE
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model.test_const_output = Constraint(
        expr=model.var_output_tot[1, "electricity"] == 1
    )
    model.test_const_emissions = Constraint(
        expr=model.var_output_tot[1, "electricity"] == 1
    )
    model.test_const_emissions = Constraint(expr=model.var_tec_emissions_pos[1] == 0)
    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASBILITY CASES
    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model.test_const_output = Constraint(
        expr=model.var_output_tot[1, "electricity"] == 1
    )
    termination = run_model(model, request.config.solver)
    cost_no_ccs = model.var_capex_tot.value
    emissions_no_ccs = sum([model.var_tec_emissions_pos[t].value for t in model.set_t])

    assert termination == TerminationCondition.optimal
    assert round(model.var_size_ccs.value, 3) == 0
    assert round(model.var_output_tot[1, "CO2captured"].value, 3) == 0
    assert round(model.var_input_tot[1, "heat"].value, 3) == 0
    assert round(model.var_input_tot[1, "electricity"].value, 3) == 0

    model = construct_tec_model(tec, nr_timesteps=time_steps)
    model.test_const_output = Constraint(
        expr=model.var_output_tot[1, "electricity"] == 1
    )
    termination = run_model(model, request.config.solver, objective="emissions")
    assert termination == TerminationCondition.optimal

    cost_ccs = model.var_capex_tot.value
    emissions_ccs = sum([model.var_tec_emissions_pos[t].value for t in model.set_t])

    assert round(model.var_size_ccs.value, 3) >= 0.2
    assert round(model.var_output_tot[1, "CO2captured"].value, 3) >= 0.9
    assert round(model.var_input_tot[1, "heat"].value, 3) >= 0.1
    assert round(model.var_input_tot[1, "electricity"].value, 3) >= 0.001
    assert cost_ccs > cost_no_ccs * 1.01
    assert emissions_ccs < emissions_no_ccs * 0.11
