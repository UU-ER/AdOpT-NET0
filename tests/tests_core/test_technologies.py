import warnings

from pyomo.environ import Constraint, TerminationCondition
import numpy as np

from tests.utilities import (
    run_model,
    make_testing_modelhub, define_technology, construct_tec_model, generate_output_constraint, generate_size_constraint,
)
from adopt_net0.core.components.utilities import annualize
from adopt_net0.core.registries import TechnologyRegistry
from adopt_net0.core.components.technologies import *

def generate_output_constraint_start_timestep(
    model, demand: float, step: int, output_ratios: dict = None
):
    """
    Generate an output constraint of a technology model for the first timestep

    :param model: pyomo model
    :param float demand: demand values to use
    :param int step: timestep to set constraint for
    :param output_ratios: output ratios to use
    :return: pyomo model
    """

    def init_output_constraint(const, car):
        if output_ratios:
            if isinstance(output_ratios.get(car), dict):
                alpha = output_ratios[car]["alpha1"]
            else:
                alpha = output_ratios[car]
            if isinstance(alpha, list):
                return model.var_output[step, car] >= demand * alpha[0]
            else:
                return model.var_output[step, car] == demand * alpha
        else:
            return model.var_output[step, car] == demand

    model.test_const_output2 = Constraint(
        model.set_output_carriers, rule=init_output_constraint
    )

    return model


def generate_var_x_constraint(model, var_x: list):
    """
    Adds constraint on var_x of a model

    :param model: pyomo model
    :param list var_x: list of values
    :return: pyomo model
    """

    def init_var_x_constraint(const, t):
        return model.var_on_off[t] == var_x[t - 1]

    model.test_const_var_x = Constraint(model.set_t, rule=init_var_x_constraint)
    return model


def generate_var_y_constraint(model, var_y: list):
    """
    Adds constraint on var_y of a model

    :param model: pyomo model
    :param list var_y: list of values
    :return: pyomo model
    """

    def init_var_y_constraint(const, t):
        return model.var_in_startup[t] == var_y[t - 1]

    model.test_const_var_y = Constraint(model.set_t, rule=init_var_y_constraint)
    return model


def generate_var_z_constraint(model, var_z: list):
    """
    Adds constraint on var_z of a model

    :param model: pyomo model
    :param list var_z: list of values
    :return: pyomo model
    """

    def init_var_z_constraint(const, t):
        return model.var_in_shutdown[t] == var_z[t - 1]

    model.test_const_var_z = Constraint(model.set_t, rule=init_var_z_constraint)
    return model


def calculate_piecewise_function(x: float, bp_x: list, bp_y: list) -> float:
    """
    Calculate the value of a piecewise function at a given point.

    This function evaluates a piecewise linear function at the point x,
    defined by the breakpoints (bp_x) and corresponding values (bp_y).

    :param float x: The point at which to evaluate the function.
    :param list bp_x: List of breakpoints defining the intervals.
    :param list bp_y: List of function values corresponding to the breakpoints.
    :return: The value of the piecewise function at the given point x
    :rtype: float
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


def test_res(request):
    """
    tests pv technology
    """
    time_steps = 1
    technology = "TestTec_Res"
    modelhub = make_testing_modelhub(time_steps, 1)
    period = modelhub.data["topology"]["investment_periods"][0]
    node = list(modelhub.data["topology"]["nodes"].keys())[0]
    modelhub.data["time_series_data"]["full_resolution"][(period, node, "TechnologyTimeSeries", technology, "capfactor")] = 0.9
    technology_registry = TechnologyRegistry()
    technology_registry.register("RES", Res)
    tec = define_technology(technology, modelhub, technology_registry, request.config.technology_data_folder_path)

    # Technology Model
    model = construct_tec_model(tec, modelhub, time_steps)

    # INFEASIBILITY CASES
    oversize = (
        np.ones(time_steps)
        * tec.size_max
        * 1.1
        * tec.processed_coeff.time_independent["rated_capacity"]
    )

    generate_output_constraint(model, oversize)
    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_tec_model(tec, modelhub, time_steps)
    generate_output_constraint(model, [1])
    termination = run_model(model, request.config.solver)
    assert termination == TerminationCondition.optimal

def test_conv_perf(request):
    """
    tests generic conversion technologies
    """

    time_steps = 1
    modelhub = make_testing_modelhub(time_steps, 1)
    tec_class = {1: Conv1, 2: Conv2, 3: Conv3, 4: Conv4}

    for conv_type in [1, 2, 3, 4]:
        technology = "TestTec_Conv" + str(conv_type)
        technology_registry = TechnologyRegistry()
        technology_registry.register("CONV" + str(conv_type), tec_class[conv_type])

        for perf_type in [1, 2, 3]:

            # Technology Model
            tec = define_technology(technology, modelhub, technology_registry,
                                    request.config.technology_data_folder_path, perf_type=perf_type)
            model = construct_tec_model(tec, modelhub, time_steps)

            if conv_type == 2 or conv_type == 3:
                output_ratios = tec.processed_coeff.time_independent["fit"]
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
                        * tec.processed_coeff.time_independent["rated_capacity"]
                )
                generate_output_constraint(model, oversize)
                termination = run_model(model, request.config.solver)
                assert termination in [
                    TerminationCondition.infeasibleOrUnbounded,
                    TerminationCondition.infeasible,
                    TerminationCondition.other,
                ]

                # FEASIBILITY CASES
                generate_output_constraint(
                    model, [1], output_ratios=output_ratios
                )
                termination = run_model(model, request.config.solver)
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
                        tec.processed_coeff.time_independent["fit"]["out"]["alpha1"]
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
                            tec.processed_coeff.time_independent["fit"][car[1]][
                                "alpha1"
                            ]
                            * car_input,
                            2,
                            )
                if conv_type == 3:
                    main_car_input = model.var_input[1, tec.main_input_carrier].value
                    for car in model.var_input:
                        car_input = model.var_input[car].value
                        assert (
                                car_input
                                == tec.processed_coeff.time_independent["phi"][car[1]]
                                * main_car_input
                        )
                    for car in model.var_output:
                        car_output = model.var_output[car].value
                        assert (
                                car_output
                                == tec.processed_coeff.time_independent["fit"][car[1]][
                                    "alpha1"
                                ]
                                * main_car_input
                        )

            elif perf_type == 2:
                # Check minimum load
                minsize = 10
                demand = [
                             minsize
                             * tec.processed_coeff.time_independent["min_part_load"]
                             * 0.1
                         ] * time_steps
                generate_size_constraint(
                    model, minsize, equality_constraint=True
                )
                generate_output_constraint(model, demand)
                termination = run_model(model, request.config.solver)

                assert termination in [
                    TerminationCondition.infeasibleOrUnbounded,
                    TerminationCondition.infeasible,
                    TerminationCondition.other,
                ]

            elif perf_type == 3 and conv_type != 4:
                # FEASIBILITY CASES
                generate_output_constraint(
                    model, [1], output_ratios=output_ratios
                )
                termination = run_model(model, request.config.solver)
                assert termination == TerminationCondition.optimal

                # Check performance for performance type 1
                if conv_type == 1:
                    bp_x = tec.processed_coeff.time_independent["fit"]["out"]["bp_x"]
                    bp_y = tec.processed_coeff.time_independent["fit"]["out"]["bp_y"]
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
                        bp_x = tec.processed_coeff.time_independent["fit"][car[1]][
                            "bp_x"
                        ]
                        bp_y = tec.processed_coeff.time_independent["fit"][car[1]][
                            "bp_y"
                        ]
                        car_output = model.var_output[car].value
                        assert round(car_output, 2) == round(
                            calculate_piecewise_function(car_input, bp_x, bp_y)
                            * car_input,
                            2,
                            )
                if conv_type == 3:
                    main_car_input = model.var_input[1, tec.main_input_carrier].value
                    for car in model.var_input:
                        car_input = model.var_input[car].value
                        assert round(car_input, 2) == round(
                            tec.processed_coeff.time_independent["phi"][car[1]]
                            * main_car_input,
                            2,
                            )
                    for car in model.var_output:
                        bp_x = tec.processed_coeff.time_independent["fit"][car[1]][
                            "bp_x"
                        ]
                        bp_y = tec.processed_coeff.time_independent["fit"][car[1]][
                            "bp_y"
                        ]
                        car_output = model.var_output[car].value
                        assert round(car_output, 2) == round(
                            calculate_piecewise_function(main_car_input, bp_x, bp_y)
                            * main_car_input,
                            2,
                            )


def test_conv_capex(request):
    """
    tests CAPEX models
    """

    time_steps = 1
    modelhub = make_testing_modelhub(time_steps, 1)
    technology = "TestTec_Conv1"
    technology_registry = TechnologyRegistry()
    technology_registry.register("CONV1", Conv1)

    for capex_model in range(1, 4):
        tec = define_technology(technology, modelhub, technology_registry, request.config.technology_data_folder_path,
                                capex_model=capex_model)

        # Technology Model
        model = construct_tec_model(tec, modelhub, time_steps)
        f = time_steps / 8760
        t = tec.economics["lifetime"]
        r = tec.economics["discount_rate"]
        a = annualize(r, t, f)

        # Check CAPEX
        generate_output_constraint(model, [1])

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
                    tec.economics["unit_capex"] * model.var_size.value * a, 4
                )

            if capex_model == 2:
                bp_x = tec.economics["piecewise_capex"]["bp_x"]
                bp_y = tec.economics["piecewise_capex"]["bp_y"]
                assert round(model.var_capex.value, 4) == round(
                    calculate_piecewise_function(model.var_size.value, bp_x, bp_y) * a,
                    4,
                )

            if capex_model == 3:
                assert round(model.var_capex.value, 5) == round(
                    (
                        tec.economics["unit_capex"] * model.var_size.value
                        + tec.economics["fix_capex"]
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
    modelhub = make_testing_modelhub(time_steps, 1)
    technology_registry = TechnologyRegistry()
    technology_registry.register("STOR", Stor)

    tec = define_technology(technology, modelhub, technology_registry, request.config.technology_data_folder_path)

    # Technology Model
    model = construct_tec_model(tec, modelhub, time_steps)

    # INFEASIBILITY CASES
    oversize = (
        np.ones(time_steps)
        * tec.size_max
        * 1.1
        * tec.processed_coeff.time_independent["rated_capacity"]
    )
    generate_output_constraint(model, oversize)
    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_tec_model(tec, modelhub, time_steps)

    def init_output_constraint(const, t, car):
        demand = [0, 1, 0]
        return model.var_output[t, car] == demand[t - 1]

    model.test_const_output5 = Constraint(
        model.set_t, model.set_output_carriers, rule=init_output_constraint
    )

    termination = run_model(model, request.config.solver)

    assert termination == TerminationCondition.optimal
    assert model.var_size.value > 0
    assert model.var_capex_aux.value > 0
    assert sum(model.var_input[t, "electricity"].value for t in model.set_t) >= 1


def test_tec_sink(request):
    """
    tests sink technology
    """
    time_steps = 2
    technology = "TestTec_Sink"
    modelhub = make_testing_modelhub(time_steps, 1)
    technology_registry = TechnologyRegistry()
    technology_registry.register("SINK", Sink)

    tec = define_technology(technology, modelhub, technology_registry, request.config.technology_data_folder_path)

    # Technology Model
    model = construct_tec_model(tec, modelhub, time_steps)

    # INFEASIBILITY CASES
    model.test_const_input = Constraint(expr=model.var_input[1, "CO2captured"] == 2)
    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # # FEASIBILITY CASES
    model = construct_tec_model(tec, modelhub, time_steps)
    model.test_const_input = Constraint(expr=model.var_input[2, "CO2captured"] == 1)
    model.test_const_level = Constraint(expr=model.var_storage_level[1] == 0)
    termination = run_model(model, request.config.solver)

    assert termination == TerminationCondition.optimal
    assert model.var_storage_level[2].value == 1
    assert model.var_input[2, "electricity"].value == 1
    assert model.var_capex.value > 0


def test_decommissioning(request):
    """
    tests decommissioning of technology
    """
    time_steps = 1
    technology = "TestTec_WindTurbine_decommission"
    modelhub = make_testing_modelhub(time_steps, 1)
    period = modelhub.data["topology"]["investment_periods"][0]
    node = list(modelhub.data["topology"]["nodes"].keys())[0]
    modelhub.data["time_series_data"]["full_resolution"][(period, node, "TechnologyTimeSeries", technology, "capfactor")] = 0.9
    technology_registry = TechnologyRegistry()
    technology_registry.register("RES", Res)

    # No decommissioning
    tec = define_technology(technology, modelhub, technology_registry, request.config.technology_data_folder_path,
                            existing=1, size_initial=15)
    model = construct_tec_model(tec, modelhub, time_steps)

    # run model
    run_model(model, request.config.solver)

    assert model.var_size.value == 15

    # Technology can decommission
    tec = define_technology(technology, modelhub, technology_registry, request.config.technology_data_folder_path,
                            existing=1, size_initial=15, decommission="continuous")
    model = construct_tec_model(tec, modelhub, time_steps)
    model.test_const_size_zero = Constraint(expr=model.var_size == 5)
    termination = run_model(model, request.config.solver)
    assert termination in [TerminationCondition.optimal]

    # Only complete decommissioning
    tec = define_technology(technology, modelhub, technology_registry, request.config.technology_data_folder_path,
                            existing=1, size_initial=15, decommission="only_complete")
    model = construct_tec_model(tec, modelhub, time_steps)
    model.test_const_size_zero = Constraint(expr=model.var_size == 5)
    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]
