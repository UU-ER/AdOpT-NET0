from pyomo.environ import Constraint, TerminationCondition

from adopt_net0.plugins.modeling_plugins.custom_technologies import *
from adopt_net0.core.registries import TechnologyRegistry
from adopt_net0.plugins.plugin_manager import PluginManager
from adopt_net0.plugins.hooks import Hook
from tests.utilities import (
    run_model,
    make_testing_modelhub, define_technology, construct_tec_model, generate_output_constraint,
)

def test_plugin_custom_technologies(request):
    technology_registry = TechnologyRegistry()
    technology_registry.register_builtin_technologies()

    pm = PluginManager()
    plugin_list = {"modeling_plugins.custom_technologies":
      {
        "config":
        {
          "technologies": ["DAC_Adsorption"]
        }
      }
    }

    pm.register(plugin_list)
    pm.emit(Hook.TECHNOLOGY_REGISTRATION, technology_registry=technology_registry)

def test_dac(request):
    """
    tests DAC Adsorption
    """
    time_steps = 1
    technology = "TestTec_DAC_Adsorption"
    modelhub = make_testing_modelhub(time_steps, 1)
    modelhub.data["time_series_data"]["full_resolution"][("period1", "node1", "TechnologyTimeSeries", "TestTec_DAC_Adsorption", "rh")] = [81]
    modelhub.data["time_series_data"]["full_resolution"][("period1", "node1", "TechnologyTimeSeries", "TestTec_DAC_Adsorption", "temp_air")] = [4]
    technology_registry = TechnologyRegistry()
    technology_registry.register("DAC_Adsorption", DacAdsorption)

    tec = define_technology(technology, modelhub, technology_registry, request.config.technology_data_folder_path)

    # INFEASIBILITY CASES
    model = construct_tec_model(tec, modelhub, time_steps)
    generate_output_constraint(model, [1])
    model.test_const_input = Constraint(expr=model.var_input[1, "electricity"] == 0)

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

    heat_in = model.var_input[1, "heat"].value
    electricity_in = model.var_input[1, "electricity"].value
    size = model.var_size.value
    capex = model.var_capex.value

    assert termination == TerminationCondition.optimal
    assert heat_in > 0.1
    assert electricity_in > 0.01
    assert size > 1
    assert round(size, 3) % 1 == 0
    assert capex > 0

    tec.size_is_int = 0
    model = construct_tec_model(tec, modelhub, time_steps)
    generate_output_constraint(model, [1])
    termination = run_model(model, request.config.solver)

    heat_in2 = model.var_input[1, "heat"].value
    electricity_in2 = model.var_input[1, "electricity"].value
    size2 = model.var_size.value
    capex2 = model.var_capex.value

    assert termination == TerminationCondition.optimal
    assert round(heat_in2, 1) == round(heat_in, 1)
    assert round(electricity_in2, 1) == round(electricity_in, 1)
    assert size2 < size
    assert size2 % 1 != 0
    assert capex2 < capex


def test_hydro_open(request):
    """
    tests Open Hydro
    """
    time_steps = 3
    technology = "TestTec_Hydro_Open"
    modelhub = make_testing_modelhub(time_steps, 1)
    modelhub.data["time_series_data"]["full_resolution"][("period1", "node1", "TechnologyTimeSeries", "TestTec_Hydro_Open", "inflow")] = [1, 1, 1]
    technology_registry = TechnologyRegistry()
    technology_registry.register("HydroOpen", HydroOpen)

    tec = define_technology(technology, modelhub, technology_registry, request.config.technology_data_folder_path)
    # INFEASIBILITY CASES
    model = construct_tec_model(tec, modelhub, time_steps)
    generate_output_constraint(model, [1, 1, 1])

    def init_test_input(const, t):
        return model.var_input[t, "electricity"] == 0

    model.test_const_input = Constraint(model.set_t, rule=init_test_input)

    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_tec_model(tec, modelhub, time_steps)
    generate_output_constraint(model, [0, 1, 1])

    def init_test_input(const, t):
        return model.var_input[t, "electricity"] == 0

    model.test_const_input = Constraint(model.set_t, rule=init_test_input)

    termination = run_model(model, request.config.solver)
    assert termination == TerminationCondition.optimal
    assert model.var_size.value == 2

    # Size min =! 0, feasibility CASES
    tec.size_min = 10
    model = construct_tec_model(tec, modelhub, time_steps)
    generate_output_constraint(model, [0, 0, 0])

    def init_test_input(const, t):
        return model.var_input[t, "electricity"] == 0

    model.test_const_input = Constraint(model.set_t, rule=init_test_input)

    termination = run_model(model, request.config.solver)
    assert termination == TerminationCondition.optimal
    assert model.var_storage_level[1, "electricity"].value <= tec.size_min


def test_heat_pump(request):
    """
    tests heat pump
    """
    time_steps = 1
    technology = "TestTec_HeatPump_AirSourced"
    modelhub = make_testing_modelhub(time_steps, 1)
    modelhub.data["time_series_data"]["full_resolution"][("period1", "node1", "TechnologyTimeSeries", "TestTec_HeatPump_AirSourced", "temp_air")] = [4]

    technology_registry = TechnologyRegistry()
    technology_registry.register("HeatPump", HeatPump)

    for perf_funct in [1, 2, 3]:
        tec = define_technology(technology, modelhub, technology_registry, request.config.technology_data_folder_path,
                                perf_type=perf_funct)

        # INFEASIBILITY CASES
        model = construct_tec_model(tec, modelhub, time_steps)
        generate_output_constraint(model, [1])
        model.test_const_input = Constraint(expr=model.var_input[1, "electricity"] == 0)

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
        assert model.var_size.value >= 0.1
        assert model.var_input[1, "electricity"].value >= 0.1


def test_gasturbine(request):
    """
    tests Gas Turbine
    """
    time_steps = 1
    technology = "TestTec_GasTurbine_NG_10"
    modelhub = make_testing_modelhub(time_steps, 1)
    modelhub.data["time_series_data"]["full_resolution"][("period1", "node1", "TechnologyTimeSeries", "TestTec_GasTurbine_NG_10", "temp_air")] = [4]

    technology_registry = TechnologyRegistry()
    technology_registry.register("GasTurbine", GasTurbine)

    tec = define_technology(technology, modelhub, technology_registry, request.config.technology_data_folder_path)

    # INFEASIBILITY CASES
    model = construct_tec_model(tec, modelhub, time_steps)
    generate_output_constraint(model, [9])
    model.test_const_input = Constraint(expr=model.var_input[1, "gas"] == 0)

    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_tec_model(tec, modelhub, time_steps)
    model.test_const_output = Constraint(expr=model.var_output[1, "electricity"] == 10)
    model.test_const_input = Constraint(expr=model.var_input[1, "hydrogen"] == 0)

    termination = run_model(model, request.config.solver)
    assert termination == TerminationCondition.optimal
    assert model.var_size.value == 1
    assert model.var_input[1, "gas"].value >= 10 / 0.4
    assert model.var_output[1, "heat"].value >= 10 * 0.5

def test_combined_cycle_fixed_size(request):
    """
    tests Gas Turbine
    """
    time_steps = 1
    technology = "TestTec_CombinedCycle_fixed_size"
    modelhub = make_testing_modelhub(time_steps, 1)
    modelhub.data["time_series_data"]["full_resolution"][("period1", "node1", "TechnologyTimeSeries", "TestTec_CombinedCycle_fixed_size", "temp_air")] = [4]


    technology_registry = TechnologyRegistry()
    technology_registry.register("CCPP", CCPP)

    tec = define_technology(technology, modelhub, technology_registry, request.config.technology_data_folder_path)

    # INFEASIBILITY CASES
    model = construct_tec_model(tec, modelhub, time_steps)
    generate_output_constraint(model, [9])

    termination = run_model(model, request.config.solver)
    assert termination in [
        TerminationCondition.infeasibleOrUnbounded,
        TerminationCondition.infeasible,
        TerminationCondition.other,
    ]

    # FEASIBILITY CASES
    model = construct_tec_model(tec, modelhub, time_steps)
    model.test_const_output1 = Constraint(
        expr=model.var_output[1, "electricity"] == 200
    )
    model.test_const_output2 = Constraint(expr=model.var_output[1, "heat"] == 5)
    model.test_const_input = Constraint(expr=model.var_input[1, "hydrogen"] == 3)

    termination = run_model(model, request.config.solver)
    assert termination == TerminationCondition.optimal
    assert model.var_input[1, "gas"].value >= 140 / 0.5

