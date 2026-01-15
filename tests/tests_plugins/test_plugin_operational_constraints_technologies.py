from pyomo.environ import Constraint, TerminationCondition

from adopt_net0.core.components.technologies import Conv1
from adopt_net0.core.registries import TechnologyRegistry
from tests.utilities import (
    run_model,
    make_testing_modelhub, define_technology, construct_tec_model, generate_output_constraint, generate_size_constraint
)

def create_operational_constraints_settings():
    return {"settings_operational_constraints": {
        "enable_operational_constraints": True,
        "max_startups": {"enabled": False, "value": 1},
        "min_uptime": {"enabled": False, "value": 1},
        "min_downtime": {"enabled": False, "value": 1},
        # "startup_time": {"enabled": False, "value": 1},
        # "shutdown_time": {"enabled": False, "value": 1},
        # "min_part_load": {"enabled": False, "value": 1},
        # "standby_power": {
        #                     "enabled": False,
        #                     "on": "output",
        #                     "carriers":
        #                         {
        #                         "electricity": 0,
        #                     },
        #                     "operator": "sum"
        #                     },
        "relative_ramping_rate_up": {
                            "enabled": False,
                            "on": "output",
                            "carriers":
                                {
                                "electricity": 0,
                            },
                            "operator": "sum"
                            },
        "relative_ramping_rate_down": {
            "enabled": False,
            "on": "output",
            "carriers":
                {
                    "electricity": 0,
                },
            "operator": "sum"
        },
    }}

def _run_operational_constraints_test_case(request, output: list, parameter_name: str, parameter_values: tuple):
    time_steps = len(output)
    modelhub = make_testing_modelhub(time_steps, 1)
    tec_class = Conv1
    conv_type = 1
    technology = "TestTec_Conv1"
    technology_registry = TechnologyRegistry()
    technology_registry.register("CONV" + str(conv_type), tec_class)
    perf_type = 2
    plugin_list = {"modeling_plugins.operational_constraints_technologies":
        {
            "config":
                {
                    "technologies": [technology]
                }
        }}

    modelhub.plugin_manager.register(plugin_list)

    settings = create_operational_constraints_settings()

    # Infeasible case
    cases = [
        {parameter_name: parameter_values[0], "feasibility": [
            TerminationCondition.infeasibleOrUnbounded,
            TerminationCondition.infeasible,
            TerminationCondition.other,
        ]},
        {parameter_name: parameter_values[1], "feasibility": [
            TerminationCondition.optimal
        ]}
    ]
    for case in cases:
        settings["settings_operational_constraints"][parameter_name] = case[parameter_name]
        tec = define_technology(technology, modelhub, technology_registry,
                                request.config.technology_data_folder_path, perf_type=perf_type,
                                additional_settings=settings)
        model = construct_tec_model(tec, modelhub, time_steps)

        output_ratios = None
        generate_output_constraint(
            model, output, output_ratios=output_ratios
        )
        generate_size_constraint(model, 2.5)

        termination = run_model(model, request.config.solver)
        model.pprint()
        assert termination in case["feasibility"]


def test_max_startups(request):
    """
    tests max startups constraint
    """
    infeasible_case = {"enabled": True, "value": 0}
    feasible_case = {"enabled": True, "value": 1}
    _run_operational_constraints_test_case(request, [0, 1, 1], "max_startups", (infeasible_case,feasible_case))


def test_min_uptime(request):
    """
    tests min uptime
    """
    infeasible_case = {"enabled": True, "value": 3}
    feasible_case = {"enabled": True, "value": 2}
    _run_operational_constraints_test_case(request, [0, 1, 1], "min_uptime", (infeasible_case,feasible_case))

def test_min_downtime(request):
    """
    tests min downtime
    """
    infeasible_case = {"enabled": True, "value": 2}
    feasible_case = {"enabled": True, "value": 1}
    _run_operational_constraints_test_case(request, [1, 0, 1], "min_downtime", (infeasible_case,feasible_case))

def test_ramping_rate_up(request):
    """
    tests ramping rate up
    """
    infeasible_case = {
                            "enabled": True,
                            "on": "output",
                            "carriers":
                                {
                                "electricity": 0.1,
                            },
                            "operator": "sum"
                            }
    feasible_case = {
        "enabled": True,
        "on": "output",
        "carriers":
            {
                "electricity": 1,
            },
        "operator": "sum"
    }
    _run_operational_constraints_test_case(request, [0, 1, 1], "relative_ramping_rate_up", (infeasible_case,feasible_case))

def test_ramping_rate_down(request):
    """
    tests ramping rate down
    """
    infeasible_case = {
                            "enabled": True,
                            "on": "output",
                            "carriers":
                                {
                                "electricity": 0.1,
                            },
                            "operator": "sum"
                            }
    feasible_case = {
        "enabled": True,
        "on": "output",
        "carriers":
            {
                "electricity": 1,
            },
        "operator": "sum"
    }
    _run_operational_constraints_test_case(request, [1, 1, 0], "relative_ramping_rate_down", (infeasible_case,feasible_case))
#
# def test_dynamics_fast(request):
#     """
#     tests dynamic operation
#     """
#
#     time_steps = 5
#     modelhub = make_testing_modelhub(time_steps, 1)
#
#     for conv_type in [1, 2, 3]:
#         technology = "TestTec_Conv" + str(conv_type)
#
#         for perf_type in [1, 2, 3]:
#             # Technology Model
#             tec = define_technology(
#                 technology,
#                 modelhub,
#                 request.config.technology_data_folder_path,
#                 perf_type=perf_type,
#             )
#
#             if conv_type == 1:
#                 output_ratios = None
#             else:
#                 output_ratios = tec.processed_coeff.time_independent["fit"]
#
#             if perf_type == 1:
#                 # Set parameters
#                 tec.processed_coeff.dynamics["ramping_time"] = 4
#                 output = [1, 1, 5, 1, 1]
#                 model = construct_tec_model(tec, modelhub, time_steps)
#
#                 generate_output_constraint(
#                     model, output, output_ratios=output_ratios
#                 )
#                 termination = run_model(model, request.config.solver)
#                 assert termination in [
#                     TerminationCondition.infeasibleOrUnbounded,
#                     TerminationCondition.infeasible,
#                     TerminationCondition.other,
#                 ]
#
#                 # Check minimum load
#                 model = construct_tec_model(tec, modelhub, time_steps)
#                 minsize = 10
#                 demand = [
#                     minsize
#                     * tec.processed_coeff.time_independent["min_part_load"]
#                     * 0.1
#                 ] * time_steps
#
#                 model = generate_size_constraint(
#                     model, minsize, equality_constraint=True
#                 )
#                 generate_output_constraint(model, demand)
#                 termination = run_model(model, request.config.solver)
#                 assert termination in [
#                     TerminationCondition.infeasibleOrUnbounded,
#                     TerminationCondition.infeasible,
#                     TerminationCondition.other,
#                 ]
#
#             elif perf_type > 1:
#                 # Set parameters
#                 tec.processed_coeff.time_independent["standby_power"] = 0.2
#                 tec.processed_coeff.time_independent["min_part_load"] = 0.3
#                 tec.processed_coeff.dynamics["max_startups"] = 1
#                 output = [1, 0, 1, 0.5, 1]
#                 var_x = [1, 0, 1, 1, 1]
#                 model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)
#
#                 generate_output_constraint(
#                     model, output, output_ratios=output_ratios
#                 )
#                 model = generate_var_x_constraint(model, var_x)
#
#                 termination = run_model(model, request.config.solver)
#                 assert termination == TerminationCondition.optimal
#
#                 # Check max startups
#                 assert (
#                     sum(model.var_on_off[t].value for t in model.var_on_off)
#                     >= time_steps - tec.processed_coeff.dynamics["max_startups"]
#                 )
#
#                 # Check standbypower
#                 main_car = tec.main_input_carrier
#                 assert round(model.var_input[2, main_car].value, 4) == round(
#                     model.var_size.value
#                     * tec.processed_coeff.time_independent["standby_power"],
#                     4,
#                 )
#
#                 # Check SUSD loads
#                 tec.processed_coeff.dynamics["SU_load"] = 0.6
#                 tec.processed_coeff.dynamics["SD_load"] = 0.4
#                 output = [1, 0, 1, 0.5, 1]
#                 model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)
#
#                 generate_output_constraint(
#                     model, output, output_ratios=output_ratios
#                 )
#                 model = generate_var_x_constraint(model, var_x)
#                 model = generate_size_constraint(model, 1)
#                 termination = run_model(model, request.config.solver)
#                 assert termination in [
#                     TerminationCondition.infeasibleOrUnbounded,
#                     TerminationCondition.infeasible,
#                     TerminationCondition.other,
#                 ]
#
#                 # Check ramping rate with tech on
#                 tec.processed_coeff.dynamics["SU_load"] = 1
#                 tec.processed_coeff.dynamics["SD_load"] = 1
#                 tec.processed_coeff.dynamics["ramping_time"] = 4
#                 tec.processed_coeff.dynamics["ramping_const_int"] = 1
#                 output = [1, 0, 1, 0.5, 1]
#                 model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)
#
#                 generate_output_constraint(
#                     model, output, output_ratios=output_ratios
#                 )
#                 model = generate_var_x_constraint(model, var_x)
#                 model = generate_size_constraint(model, 1)
#                 termination = run_model(model, request.config.solver)
#                 assert termination in [
#                     TerminationCondition.infeasibleOrUnbounded,
#                     TerminationCondition.infeasible,
#                     TerminationCondition.other,
#                 ]
#
#
# def test_dynamics_slow(request):
#     """
#     tests dynamic operation
#     """
#
#     time_steps = 5
#     modelhub = make_testing_modelhub(time_steps, 1)
#
#     for conv_type in [1, 2, 3]:
#         technology = "TestTec_Conv" + str(conv_type)
#
#         # Technology Model
#         perf_type = 4
#         tec = define_technology(
#             technology,
#             modelhub,
#             load_path=request.config.technology_data_folder_path,
#             perf_type=perf_type,
#         )
#
#         if conv_type == 1:
#             output_ratios = None
#         else:
#             output_ratios = tec.processed_coeff.time_independent["fit"]
#
#         # check SD time
#         SD_time = 2
#         min_part_load = 0.6
#         tec.processed_coeff.dynamics["SD_time"] = SD_time
#         tec.processed_coeff.time_independent["min_part_load"] = min_part_load
#         output_start = 0.6
#         var_z = [0, 0, 1, 0, 0]
#         var_x = [1, 1, 0, 0, 0]
#         model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)
#
#         model = generate_var_z_constraint(model, var_z)
#         model = generate_var_x_constraint(model, var_x)
#         generate_output_constraint_start_timestep(
#             model, demand=output_start, step=1, output_ratios=output_ratios
#         )
#         termination = run_model(model, request.config.solver)
#         assert termination == TerminationCondition.optimal
#
#         main_car = tec.main_input_carrier
#         trajectory = model.var_size.value * min_part_load / (SD_time + 1)
#
#         if conv_type < 3:
#             input_at_SD1 = (
#                 model.var_input[3, "gas"].value + model.var_input[3, "hydrogen"].value
#             )
#             input_at_SD2 = (
#                 model.var_input[4, "gas"].value + model.var_input[4, "hydrogen"].value
#             )
#         else:
#             input_at_SD1 = model.var_input[3, main_car].value
#             input_at_SD2 = model.var_input[4, main_car].value
#
#         assert round(input_at_SD2, 3) == round(trajectory, 3)
#         assert round(input_at_SD1, 3) == round(trajectory * SD_time, 3)
#
#         # Check infeasibility case
#         var_x = [1, 1, 0, 0, 1]
#         model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)
#
#         model = generate_var_z_constraint(model, var_z)
#         model = generate_var_x_constraint(model, var_x)
#         generate_output_constraint_start_timestep(
#             model, demand=output_start, step=1
#         )
#         termination = run_model(model, request.config.solver)
#         assert termination in [
#             TerminationCondition.infeasibleOrUnbounded,
#             TerminationCondition.infeasible,
#             TerminationCondition.other,
#         ]
#
#         # check SU time
#         tec = define_technology(
#             technology,
#             modelhub,
#             load_path=request.config.technology_data_folder_path,
#             perf_type=perf_type,
#         )
#
#         SU_time = 2
#         min_part_load = 0.6
#         tec.processed_coeff.dynamics["SU_time"] = SU_time
#         tec.processed_coeff.time_independent["min_part_load"] = min_part_load
#         output_start = 0.6
#         var_y = [0, 0, 0, 1, 0]
#         var_x = [0, 0, 0, 1, 1]
#         model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)
#
#         model = generate_var_y_constraint(model, var_y)
#         model = generate_var_x_constraint(model, var_x)
#         generate_output_constraint_start_timestep(
#             model,
#             demand=output_start,
#             step=len(model.set_t),
#             output_ratios=output_ratios,
#         )
#         termination = run_model(model, request.config.solver)
#         assert termination == TerminationCondition.optimal
#
#         main_car = tec.main_input_carrier
#         trajectory = model.var_size.value * min_part_load / (SU_time + 1)
#
#         if conv_type < 3:
#             input_at_SU1 = (
#                 model.var_input[2, "gas"].value + model.var_input[2, "hydrogen"].value
#             )
#             input_at_SU2 = (
#                 model.var_input[3, "gas"].value + model.var_input[3, "hydrogen"].value
#             )
#         else:
#             input_at_SU1 = model.var_input[2, main_car].value
#             input_at_SU2 = model.var_input[3, main_car].value
#
#         assert round(input_at_SU1, 3) == round(trajectory, 3)
#         assert round(input_at_SU2, 3) == round(trajectory * SU_time, 3)
#
#         # Check infeasibility case
#         var_x = [0, 0, 1, 1, 1]
#         model = construct_tec_model(tec, nr_timesteps=time_steps, dynamics=1)
#
#         model = generate_var_y_constraint(model, var_y)
#         model = generate_var_x_constraint(model, var_x)
#         generate_output_constraint_start_timestep(
#             model, demand=output_start, step=1
#         )
#         termination = run_model(model, request.config.solver)
#         assert termination in [
#             TerminationCondition.infeasibleOrUnbounded,
#             TerminationCondition.infeasible,
#             TerminationCondition.other,
#         ]
