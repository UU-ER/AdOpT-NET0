import random
from pathlib import Path
from pyomo.environ import (
    ConcreteModel,
    Set,
    Block,
    Objective,
    Var,
    Constraint,
    TransformationFactory,
    minimize,
    Suffix,
    SolverStatus,
)
import os
import time
import numpy as np
import pandas as pd
import sys
import datetime

from pyomo.gdp import Disjunction, Disjunct

from .data_management import DataHandle
from .model_construction import *
from .utilities import get_glpk_parameters, get_gurobi_parameters, log_event
from .result_management import *
from .components.utilities import (
    annualize,
    set_discount_rate,
    perform_disjunct_relaxation,
)
from .components.technologies.utilities import set_capex_model


class EnergyHub:
    r"""
    Class to construct and manipulate an energy system model.

    When constructing an instance, it reads data to the instance and initializes all attributes of the EnergyHub
    class:

    - self.logger: Logger
    - self.data: Data container
    - self.model: Model container
    - self.solution: Solution container
    - self.solver: Solver container
    """

    def __init__(self):
        """
        Constructor
        """
        self.data = None
        self.model = {}
        self.solution = {}
        self.solver = None
        self.last_solve_info = {}
        self.info_pareto = {}
        self.info_pareto["pareto_point"] = -1
        self.info_solving_algorithms = {}
        self.info_monte_carlo = {}

    def read_data(
        self, data_path: Path | str, start_period: int = None, end_period: int = None
    ) -> None:
        """
        Reads in data from the specified path. The data is specified as the DataHandle class
        Specifying the start_period and end_period parameter allows to run a shorter time horizon than as specified
        in the topology (e.g. for testing)

        :param Path, str data_path: Path of folder structure to read data from
        :param int start_period: starting period of the model
        :param int end_period: end period of the model
        """
        log_event("--- Reading in data ---")

        self.data = DataHandle()
        self.data.read_input_data(data_path, start_period, end_period)
        self._perform_preprocessing_checks()

        log_event("--- Reading in data complete ---")

    def construct_model(self):
        """
        Constructs the model. The model structure is as follows:

        **Global sets**

        - set_periods: set of investment periods
        - set_nodes: set of nodes
        - set_carriers: set of carriers modelled

        **Global variables**

        - var_npv: net present value of all costs
        - var_emissions_net: net emissions over all investment periods

        **Rest of model**
        The rest of the model is organized in nested, hierarchical pyomo modelling blocks:

        Investment Period Block

            Network Block

            Node Block

                Technology Block
        """
        log_event("--- Constructing Model ---")
        start = time.time()

        # INITIALIZE MODEL
        aggregation_type = "full"

        # TODO: Add clustered, averaged here
        self.model[aggregation_type] = ConcreteModel()

        # GET DATA
        model = self.model[aggregation_type]
        topology = self.data.topology
        config = self.data.model_config

        # DEFINE GLOBAL SETS
        # Nodes, Carriers, Technologies, Networks
        model.set_periods = Set(initialize=topology["investment_periods"])
        model.set_nodes = Set(initialize=topology["nodes"])
        model.set_carriers = Set(initialize=topology["carriers"])

        # DEFINE GLOBAL VARIABLES
        model.var_npv = Var()
        model.var_emissions_net = Var()

        # INVESTMENT PERIOD BLOCK
        def init_period_block(b_period):
            """Pyomo rule to initialize a block holding all investment periods"""

            # Get data for investment period
            investment_period = b_period.index()
            data_period = get_data_for_investment_period(
                self.data, investment_period, aggregation_type
            )
            log_event(f"--- Constructing Investment Period {investment_period}")

            # Add sets, parameters, variables, constraints to block
            b_period = construct_investment_period_block(b_period, data_period)

            # NETWORK BLOCK
            if not config["energybalance"]["copperplate"]["value"]:

                def init_network_block(b_netw, netw):
                    """Pyomo rule to initialize a block holding all networks"""
                    log_event(f"------ Constructing Network {netw}")

                    # Add sets, parameters, variables, constraints to block
                    b_netw = construct_network_block(
                        b_netw,
                        data_period,
                        model.set_nodes,
                        b_period.set_t_full,
                        b_period.set_t_clustered,
                    )

                    return b_netw

                b_period.network_block = Block(
                    b_period.set_networks, rule=init_network_block
                )

            # NODE BLOCK
            def init_node_block(b_node, node):
                """Pyomo rule to initialize a block holding all nodes"""
                log_event(f"------ Constructing Node {node}")

                # Get data for node
                data_node = get_data_for_node(data_period, node)

                # Add sets, parameters, variables, constraints to block
                b_node = construct_node_block(b_node, data_node, b_period.set_t_full)

                # TECHNOLOGY BLOCK
                def init_technology_block(b_tec, tec):
                    log_event(f"------ Constructing Technology {tec}")

                    b_tec = construct_technology_block(
                        b_tec, data_node, b_period.set_t_full, b_period.set_t_clustered
                    )

                    return b_tec

                b_node.tech_blocks_active = Block(
                    b_node.set_technologies, rule=init_technology_block
                )

                return b_node

            b_period.node_blocks = Block(model.set_nodes, rule=init_node_block)

            return b_period

        model.periods = Block(model.set_periods, rule=init_period_block)

        log_event(f"Constructing model completed in {str(round(time.time() - start))}s")

    def _perform_preprocessing_checks(self):
        """
        Checks some things, before constructing or solving the model
        Todo: Document what is done here
        :return:
        """
        config = self.data.model_config

        # Check if save-path exists
        save_path = Path(config["reporting"]["save_path"]["value"])
        if not os.path.exists(save_path) or not os.path.isdir(save_path):
            raise FileNotFoundError(
                f"The folder '{save_path}' does not exist. Create the folder or change the folder "
                f"name in the configuration"
            )

        # check if technologies have dynamic parameters
        if config["performance"]["dynamics"]["value"]:
            for node in self.data.topology.nodes:
                for tec in self.data.technology_data[node]:
                    if self.data.technology_data[node][tec].technology_model in [
                        "CONV1",
                        "CONV2",
                        "CONV3",
                    ]:
                        par_check = [
                            "max_startups",
                            "min_uptime",
                            "min_downtime",
                            "SU_load",
                            "SD_load",
                            "SU_time",
                            "SD_time",
                        ]
                        count = 0
                        for par in par_check:
                            if (
                                par
                                not in self.data.technology_data[node][
                                    tec
                                ].performance_data
                            ):
                                raise ValueError(
                                    f"The technology '{tec}' does not have dynamic parameter '{par}'. Add the parameters in the "
                                    f"json files or switch off the dynamics."
                                )

        # check if time horizon is not longer than 1 year (in case of single year analysis)
        # TODO: Do we need multiyear analysis still?
        # if config["optimization"]["multiyear"]["value"] == 0:
        #     nr_timesteps_data = len(self.data.topology.timesteps)
        #     nr_timesteps_year = 8760
        #     if nr_timesteps_data > nr_timesteps_year:
        #         raise ValueError(
        #             f"Time horizon is longer than one year. Enable multiyear analysis if you want to optimize for"
        #             f"a longer time horizon."
        #         )

    def quick_solve(self):
        """
        Quick-solves the model (constructs model and balances and solves model).

        This method lumbs together the following functions for convenience:
        - :func:`~src.energyhub.construct_model`
        - :func:`~src.energyhub.construct_balances`
        - :func:`~src.energyhub.solve`
        - :func:`~src.energyhub.write_results`
        """
        self.construct_model()
        self.construct_balances()
        self.solve()
        if self.data.model_config["optimization"]["monte_carlo"]["N"]["value"] == 0:
            self.write_results()

    def construct_balances(self):
        """
        Todo: document
        Constructs the energy balance, emission balance and calculates costs

        Links all components with the constructing the energybalance (:func:`~add_energybalance`),
        the total cost (:func:`~add_system_costs`) and the emission balance (:func:`~add_emissionbalance`)
        """
        log_event("Constructing balances...")
        start = time.time()

        config = self.data.model_config
        model = self.model["full"]

        model = delete_all_balances(model)

        if not config["energybalance"]["copperplate"]["value"]:
            model = construct_network_constraints(model)
            model = construct_nodal_energybalance(model, config)
        else:
            model = construct_global_energybalance(model, config)

        model = construct_emission_balance(model, config)
        model = construct_system_cost(model, config)
        model = construct_global_balance(model)

        log_event(
            f"Constructing balances completed in {str(round(time.time() - start))}s"
        )

    def solve(self):
        """
        Defines objective and solves model

        The objective is minimized and can be chosen as total annualized costs ('costs'), total annual net emissions
        ('emissions_net'), total positive emissions ('emissions_pos') and annual emissions at minimal cost
        ('emissions_minC'). This needs to be set in the configuration file respectively.
        """
        config = self.data.model_config

        objective = config["optimization"]["objective"]["value"]

        self._define_solver_settings()

        if config["optimization"]["monte_carlo"]["N"]["value"]:
            self._solve_monte_carlo(objective)
        elif objective == "pareto":
            # Todo: does not work yet
            self._solve_pareto()
        else:
            self._optimize(objective)

    def add_technology_to_node(self, nodename, technologies):
        """
        Fixme: This function does not work
        Adds technologies retrospectively to the model.

        After adding a technology to a node, the energy and emission balance need to be re-constructed, as well as the
        costs recalculated. To solve the model, :func:`~construct_balances` and then solve again.

        :param str nodename: name of node for which technology is installed
        :param list technologies: list of technologies that should be added to nodename
        :return None:
        """
        self.data.read_single_technology_data(nodename, technologies)
        add_technology(self, nodename, technologies)

    def _define_solver_settings(self):
        """
        Defines solver and its settings depending on objective and solver
        """
        config = self.data.model_config
        model = self.model["full"]

        objective = config["optimization"]["objective"]["value"]

        # Set solver
        if config["solveroptions"]["solver"]["value"] in [
            "gurobi",
            "gurobi_persistent",
        ]:
            # Gurobi
            if not config["scaling"]["scaling_on"]["value"]:
                if objective in ["emissions_minC", "pareto"]:
                    config["solveroptions"]["solver"]["value"] = "gurobi_persistent"
            self.solver = get_gurobi_parameters(config["solveroptions"])

        elif config["solveroptions"]["solver"]["value"] == "glpk":
            # Todo: put solver parameters of glpk in function
            self.solver = get_glpk_parameters(config["solveroptions"])

        # For persistent solver, set model instance
        if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
            self.solver.set_instance(model)

    def _optimize(self, objective):
        """
        Solves the model with the given objective
        """
        # Define Objective Function
        if objective == "costs":
            self._optimize_cost()
        elif objective == "emissions_net":
            self._optimize_emissions_net()
        elif objective == "emissions_minC":
            self._optimize_costs_minE()
        elif objective == "costs_emissionlimit":
            self._optimize_costs_emissionslimit()
        else:
            raise Exception("objective in Configurations is incorrect")

        # Second stage of time averaging algorithm
        # Fixme: averaging fix
        # if (
        #     self.model_information.averaged_data
        #     and self.model_information.averaged_data_specs.stage == 0
        # ):
        #     self._optimize_time_averaging_second_stage()

    def _optimize_cost(self):
        """
        Minimizes Costs
        """
        model = self.model["full"]

        self._delete_objective()

        def init_cost_objective(obj):
            return model.var_npv

        model.objective = Objective(rule=init_cost_objective, sense=minimize)
        self._call_solver()

    def _optimize_emissions_net(self):
        """
        Minimize net emissions
        """
        model = self.model["full"]

        self._delete_objective()

        def init_emission_net_objective(obj):
            return model.var_emissions_net

        model.objective = Objective(rule=init_emission_net_objective, sense=minimize)
        self._call_solver()

    def _optimize_costs_emissionslimit(self):
        """
        Minimize costs at emission limit
        """
        model = self.model["full"]

        config = self.data.model_config

        emission_limit = config["optimization"]["emission_limit"]["value"]
        if model.find_component("const_emission_limit"):
            if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
                self.solver.remove_constraint(model.const_emission_limit)
            model.del_component(model.const_emission_limit)
        model.const_emission_limit = Constraint(
            expr=model.var_emissions_net <= emission_limit * 1.001
        )
        if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
            self.solver.add_constraint(model.const_emission_limit)
        self._optimize_cost()

    def _optimize_costs_minE(self):
        """
        Minimize costs at minimum emissions
        """
        model = self.model["full"]

        config = self.data.model_config

        self._optimize_emissions_net()
        emission_limit = model.var_emissions_net.value
        if model.find_component("const_emission_limit"):
            if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
                self.solver.remove_constraint(model.const_emission_limit)
            model.del_component(model.const_emission_limit)
        model.const_emission_limit = Constraint(
            expr=model.var_emissions_net <= emission_limit * 1.001
        )
        if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
            self.solver.add_constraint(model.const_emission_limit)
        self._optimize_cost()

    def _solve_pareto(self):
        """
        Optimize the pareto front
        """
        model = self.model["full"]
        config = self.data.model_config
        pareto_points = config["optimization"]["pareto_points"]["value"]

        # Min Cost
        self.info_pareto["pareto_point"] = 0
        self._optimize_cost()
        emissions_max = model.var_emissions_net.value

        # Min Emissions
        self.info_pareto["pareto_point"] = pareto_points + 1
        self._optimize_costs_minE()
        emissions_min = model.var_emissions_net.value

        # Emission limit
        self.info_pareto["pareto_point"] = 0
        emission_limits = np.linspace(emissions_min, emissions_max, num=pareto_points)
        for pareto_point in range(0, pareto_points):
            self.info_pareto["pareto_point"] += 1
            if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
                self.solver.remove_constraint(model.const_emission_limit)
            model.del_component(model.const_emission_limit)
            model.const_emission_limit = Constraint(
                expr=model.var_emissions_net <= emission_limits[pareto_point] * 1.005
            )
            if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
                self.solver.add_constraint(model.const_emission_limit)
            self._optimize_cost()

    def _solve_monte_carlo(self, objective):
        """
        Optimizes multiple runs with monte carlo
        """
        config = self.data.model_config
        self.info_monte_carlo["monte_carlo_run"] = 0

        for run in range(0, config["optimization"]["monte_carlo"]["N"]["value"]):
            self.info_monte_carlo["monte_carlo_run"] += 1
            self._monte_carlo_set_cost_parameters()
            if run == 0:
                self._optimize(objective)
                self.write_results()
            else:
                self._call_solver()
                self.write_results()

    def scale_model(self):
        """
        Creates a scaled model in self.scaled_model using the scale factors specified in the json files for technologies
        and networks as well as the global scaling factors specified. See also the documentation on model scaling.
        """
        config = self.data.model_config

        f_global = config["scaling_factors"]["value"]
        model_full = self.model["full"]

        model_full.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Scale technologies
        for node in model_full.node_blocks:
            for tec in model_full.node_blocks[node].tech_blocks_active:
                b_tec = model_full.node_blocks[node].tech_blocks_active[tec]
                model_full = self.data.technology_data[node][tec].scale_model(
                    b_tec, model_full, config
                )

        # Scale networks
        for netw in model_full.network_block:
            b_netw = model_full.network_block[netw]
            model_full = self.data.network_data[netw].scale_model(
                b_netw, model_full, config
            )

        # Scale objective
        model_full.scaling_factor[model_full.objective] = (
            f_global.objective * f_global.cost_vars
        )

        # Scale globals
        if f_global.energy_vars >= 0:
            model_full.scaling_factor[model_full.const_energybalance] = (
                f_global.energy_vars
            )
            model_full.scaling_factor[model_full.const_cost] = (
                f_global.cost_vars * f_global.energy_vars
            )
            model_full.scaling_factor[model_full.const_node_cost] = (
                f_global.cost_vars * f_global.energy_vars
            )
            model_full.scaling_factor[model_full.const_netw_cost] = (
                f_global.cost_vars * f_global.energy_vars
            )
            model_full.scaling_factor[model_full.const_revenue_carbon] = (
                f_global.cost_vars * f_global.energy_vars
            )
            model_full.scaling_factor[model_full.const_cost_carbon] = (
                f_global.cost_vars * f_global.energy_vars
            )

            model_full.scaling_factor[model_full.var_node_cost] = (
                f_global.cost_vars * f_global.energy_vars
            )
            model_full.scaling_factor[model_full.var_netw_cost] = (
                f_global.cost_vars * f_global.energy_vars
            )
            model_full.scaling_factor[model_full.var_cost_total] = (
                f_global.cost_vars * f_global.energy_vars
            )
            model_full.scaling_factor[model_full.var_carbon_revenue] = (
                f_global.cost_vars * f_global.energy_vars
            )
            model_full.scaling_factor[model_full.var_carbon_cost] = (
                f_global.cost_vars * f_global.energy_vars
            )

            for node in model_full.node_blocks:
                model_full.scaling_factor[
                    model_full.node_blocks[node].var_import_flow
                ] = f_global.energy_vars
                model_full.scaling_factor[
                    model_full.node_blocks[node].var_export_flow
                ] = f_global.energy_vars

                model_full.scaling_factor[
                    model_full.node_blocks[node].var_netw_inflow
                ] = f_global.energy_vars
                model_full.scaling_factor[
                    model_full.node_blocks[node].const_netw_inflow
                ] = f_global.energy_vars

                model_full.scaling_factor[
                    model_full.node_blocks[node].var_netw_outflow
                ] = f_global.energy_vars
                model_full.scaling_factor[
                    model_full.node_blocks[node].const_netw_outflow
                ] = f_global.energy_vars

                model_full.scaling_factor[
                    model_full.node_blocks[node].var_generic_production
                ] = f_global.energy_vars
                model_full.scaling_factor[
                    model_full.node_blocks[node].const_generic_production
                ] = f_global.energy_vars

        self.model["scaled"] = TransformationFactory("core.scale_model").create_using(
            model_full
        )
        # self.scaled_model.pprint()

    def _call_solver(self):
        """
        Calls the solver and solves the model
        """
        print("_" * 60)
        print("Solving Model...")

        start = time.time()
        config = self.data.model_config

        # Create save path and folder
        time_stamp = datetime.datetime.fromtimestamp(start).strftime("%Y%m%d%H%M%S")
        save_path = Path(config["reporting"]["save_path"]["value"])

        if config["reporting"]["case_name"]["value"] == -1:
            folder_name = str(time_stamp)
        else:
            folder_name = (
                str(time_stamp) + "_" + config["reporting"]["case_name"]["value"]
            )

        result_folder_path = create_unique_folder_name(save_path, folder_name)
        create_save_folder(result_folder_path)

        # Scale model
        if config["scaling"]["scaling_on"]["value"] == 1:
            self.scale_model()
            model = self.model["scaled"]
        else:
            model = self.model["full"]

        # Call solver
        if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
            self.solver.set_objective(model.objective)

        if config["solveroptions"]["solver"]["value"] == "glpk":
            self.solution = self.solver.solve(
                model,
                tee=True,
                logfile=str(Path(result_folder_path / "log.txt")),
                keepfiles=True,
            )
        else:
            self.solution = self.solver.solve(
                model,
                tee=True,
                warmstart=True,
                logfile=str(Path(result_folder_path / "log.txt")),
                keepfiles=True,
            )

        if config["scaling"]["scaling_on"]["value"] == 1:
            TransformationFactory("core.scale_model").propagate_solution(
                model, self.model["full"]
            )

        if config["reporting"]["write_solution_diagnostics"]["value"] >= 1:
            self._write_solution_diagnostics(result_folder_path)

        self.solution.write()

        self.last_solve_info["pareto_point"] = self.info_pareto["pareto_point"]
        self.last_solve_info["monte_carlo_run"] = 0
        self.last_solve_info["config"] = config
        self.last_solve_info["result_folder_path"] = result_folder_path

        print("Solving model completed in " + str(round(time.time() - start)) + " s")
        print("_" * 60)

    def write_results(self):
        config = self.data.model_config

        save_summary_path = Path.joinpath(
            Path(config["reporting"]["save_summary_path"]["value"]), "Summary.xlsx"
        )

        model_info = self.last_solve_info
        # Write H5 File
        if (self.solution.solver.status == SolverStatus.ok) or (
            self.solution.solver.status == SolverStatus.warning
        ):

            model = self.model["full"]

            # Fixme: change this for averaging and kmeans

            summary_dict = write_optimization_results_to_h5(
                model, self.solution, model_info, self.data
            )

            # Write Summary
            if not os.path.exists(save_summary_path):
                summary_df = pd.DataFrame(data=summary_dict, index=[0])
                summary_df.to_excel(
                    save_summary_path, index=False, sheet_name="Summary"
                )
            else:
                summary_existing = pd.read_excel(save_summary_path)
                pd.concat(
                    [summary_existing, pd.DataFrame(data=summary_dict, index=[0])]
                ).to_excel(save_summary_path, index=False, sheet_name="Summary")

    def _write_solution_diagnostics(self, save_path):
        config = self.data.model_config
        model = self.solver._solver_model
        constraint_map = self.solver._pyomo_con_to_solver_con_map
        variable_map = self.solver._pyomo_var_to_solver_var_map

        # Write solution quality to txt
        with open(f"{save_path}/diag_solution_quality.txt", "w") as file:
            sys.stdout = file  # Redirect stdout to the file
            model.printQuality()  # Call the function that prints something
            sys.stdout = sys.__stdout__  # Reset stdout to the console

        if config["reporting"]["write_solution_diagnostics"]["value"] >= 2:
            # Write constraint map to txt
            with open(f"{save_path}/diag_constraint_map.txt", "w") as file:
                for key, value in constraint_map.items():
                    file.write(f"{key}: {value}\n")

            # Write var map to txt
            with open(f"{save_path}/diag_variable_map.txt", "w") as file:
                for key, value in variable_map._dict.items():
                    file.write(f"{value[0].name}: {value[1]}\n")

    def _monte_carlo_set_cost_parameters(self):
        """
        Performs monte carlo analysis
        """
        config = self.data.model_config

        # use correct resolution
        model = self.model["full"]

        if config["optimization"]["monte_carlo"]["type"]["value"] == 1:
            if (
                "Technologies"
                in config["optimization"]["monte_carlo"]["on_what"]["value"]
            ):
                for period in model.periods:
                    for node in model.periods[period].node_blocks:
                        for tec in (
                            model.periods[period].node_blocks[node].tech_blocks_active
                        ):
                            self._monte_carlo_technologies(period, node, tec)

            if "Networks" in config["optimization"]["monte_carlo"]["on_what"]["value"]:
                for period in model.periods:
                    for netw in model.periods[period].network_block:
                        self._monte_carlo_networks(period, netw)

            if (
                "ImportPrices"
                in config["optimization"]["monte_carlo"]["on_what"]["value"]
            ):
                for period in model.periods:
                    for node in model.periods[period].node_blocks:
                        for car in model.periods[period].node_blocks[node].set_carriers:
                            self._monte_carlo_import_prices(period, node, car)

            if (
                "ExportPrices"
                in config["optimization"]["monte_carlo"]["on_what"]["value"]
            ):
                for period in model.periods:
                    for node in model.periods[period].node_blocks:
                        for car in model.periods[period].node_blocks[node].set_carriers:
                            self._monte_carlo_export_prices(period, node, car)
        elif config["optimization"]["monte_carlo"]["type"]["value"] == 2:
            MC_parameters = self.data.monte_carlo_specs

            for index, row in MC_parameters.iterrows():
                if row["type"] == "technology":
                    MC_technology_row = row
                    tec = MC_technology_row["name"]

                    for period in model.periods:
                        for node in model.periods[period].node_blocks:
                            if tec in (
                                model.periods[period]
                                .node_blocks[node]
                                .tech_blocks_active
                            ):
                                self._monte_carlo_technologies(
                                    period, node, tec, MC_technology_row
                                )
                elif row["type"] == "network":
                    MC_network_row = row
                    # TODO: implement for networks
                elif row["type"] == "import":
                    MC_import_row = row

                    for period in model.periods:
                        for node in model.periods[period].node_blocks:
                            self._monte_carlo_import_prices(
                                period,
                                node,
                                MC_import_row["name"],
                                MC_import_row,
                            )
                elif row["type"] == "export":
                    MC_export_row = row

                    for period in model.periods:
                        for node in model.periods[period].node_blocks:
                            self._monte_carlo_export_prices(
                                period,
                                node,
                                MC_export_row["name"],
                                MC_export_row,
                            )

    def _monte_carlo_technologies(self, period, node, tec, MC_technology_row=None):
        """
        Changes the capex of technologies
        """
        aggregation_type = "full"
        config = self.data.model_config
        tec_data = self.data.technology_data[aggregation_type][period][node][tec]
        model = self.model[aggregation_type]

        sd = config["optimization"]["monte_carlo"]["sd"]["value"]
        sd_random = np.random.normal(1, sd)

        economics = tec_data.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = self.data.topology["fraction_of_year_modelled"]

        b_tec = model.periods[period].node_blocks[node].tech_blocks_active[tec]

        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )

        # Change parameters
        if tec_data.economics.capex_model == 1:
            # UNIT CAPEX
            # Update parameter
            if MC_technology_row is not None:
                unit_capex = random.uniform(
                    MC_technology_row["min"], MC_technology_row["max"]
                )
            else:
                unit_capex = tec_data.economics.capex_data["unit_capex"] * sd_random

            b_tec.para_unit_capex = unit_capex
            b_tec.para_unit_capex_annual = unit_capex * annualization_factor

        elif tec_data.economics.capex_model == 2:
            warnings.warn(
                "monte carlo on piecewise defined investment costs is not implemented"
            )

        elif tec_data.economics.capex_model == 3:
            unit_capex = tec_data.economics.capex_data["unit_capex"] * sd_random
            b_tec.para_unit_capex = unit_capex
            b_tec.para_unit_capex_annual = unit_capex * annualization_factor

            fix_capex = tec_data.economics.capex_data["fix_capex"] * sd_random
            b_tec.para_fix_capex = fix_capex
            b_tec.para_fix_capex_annual = fix_capex * annualization_factor

        # Change variable bounds
        def calculate_max_capex():
            if economics.capex_model == 1:
                max_capex = b_tec.para_size_max * b_tec.para_unit_capex_annual
                bounds = (0, max_capex)
            elif economics.capex_model == 2:
                # Todo: Calculate
                pass
                # max_capex = (
                #     b_tec.para_size_max
                #     * max(economics.capex_data["piecewise_capex"]["bp_y"])
                #     * annualization_factor
                # )
                # bounds = (0, max_capex)
            elif economics.capex_model == 3:
                max_capex = (
                    b_tec.para_size_max * b_tec.para_unit_capex_annual
                    + b_tec.para_fix_capex_annual
                )
                bounds = (0, max_capex)
            else:
                bounds = None
            return bounds

        bounds = calculate_max_capex()
        b_tec.var_capex_aux.setlb(bounds[0])
        b_tec.var_capex_aux.setub(bounds[1])

        # Delete constraints/conjunctions/relaxations
        if economics.capex_model == 1:
            big_m_transformation_required = 0
            b_tec.del_component(b_tec.const_capex_aux)
        elif economics.capex_model == 2:
            big_m_transformation_required = 1
            b_tec.del_component(b_tec.const_capex_aux)
        elif economics.capex_model == 3:
            big_m_transformation_required = 1
            b_tec.del_component(b_tec.dis_installation)
            b_tec.del_component(b_tec.disjunction_installation)
            b_tec.del_component(b_tec._pyomo_gdp_bigm_reformulation)

        # Reconstruct technology constraints
        data_period = get_data_for_investment_period(
            self.data, period, aggregation_type
        )
        data_node = get_data_for_node(data_period, node)

        b_tec = tec_data._define_capex_constraints(b_tec, data_node)
        if big_m_transformation_required:
            b_tec = perform_disjunct_relaxation(b_tec)

    def _monte_carlo_networks(self, period, netw):
        """
        Changes the capex of networks
        """
        config = self.data.model_config
        aggregation_type = "full"

        sd = config["optimization"]["monte_carlo"]["sd"]["value"]
        sd_random = np.random.normal(1, sd)

        netw_data = self.data.network_data[aggregation_type][period][netw]
        economics = netw_data.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = self.data.topology["fraction_of_year_modelled"]

        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )

        b_netw = self.model[aggregation_type].periods[period].network_block[netw]

        # Update cost parameters
        b_netw.para_capex_gamma1 = (
            economics.capex_data["gamma1"] * annualization_factor * sd_random
        )
        b_netw.para_capex_gamma2 = (
            economics.capex_data["gamma2"] * annualization_factor * sd_random
        )
        b_netw.para_capex_gamma3 = (
            economics.capex_data["gamma3"] * annualization_factor * sd_random
        )
        b_netw.para_capex_gamma4 = (
            economics.capex_data["gamma4"] * annualization_factor * sd_random
        )

        for arc in b_netw.set_arcs:
            b_arc = b_netw.arc_block[arc]

            # Remove constraint (from persistent solver and from model)
            # FIXME: remove disjuctions installation
            b_arc.del_component(b_arc.const_capex_aux)

            # Add constraint again
            def init_capex(const):
                return (
                    b_arc.var_capex_aux
                    == b_netw.para_capex_gamma1
                    + b_netw.para_capex_gamma2 * b_arc.var_size
                    + b_netw.para_capex_gamma3 * b_arc.distance
                    + b_netw.para_capex_gamma4 * b_arc.var_size * b_arc.distance
                )

            b_arc.const_capex_aux = Constraint(rule=init_capex)

            s_indicators = range(0, 2)

            def init_installation(dis, ind):
                if ind == 0:  # network not installed
                    dis.const_capex_aux = Constraint(expr=b_arc.var_capex_aux == 0)
                    dis.const_not_installed = Constraint(expr=b_arc.var_size == 0)
                else:  # network installed
                    dis.const_capex_aux = Constraint(rule=init_capex)

            b_arc.dis_installation = Disjunct(s_indicators, rule=init_installation)

            def bind_disjunctions(dis):
                return [b_arc.dis_installation[i] for i in s_indicators]

            b_arc.disjunction_installation = Disjunction(rule=bind_disjunctions)

            # perform relaxation
            b_netw = perform_disjunct_relaxation(b_netw)

    def _monte_carlo_import_prices(self, period, node, car, MC_import_row=None):
        """
        Changes the import prices
        """
        config = self.data.model_config
        aggregation_type = "full"

        sd = config["optimization"]["monte_carlo"]["sd"]["value"]
        sd_random = np.random.normal(1, sd)

        model = self.model[aggregation_type]
        import_prices = self.data.time_series[aggregation_type][period][node][
            "CarrierData"
        ][car]["Import price"]
        b_period_cost = model.block_costbalance[period]
        # TODO: check for averaging
        set_t = model.periods[period].set_t_full

        # Update parameter
        for t in set_t:
            if MC_import_row is not None:
                model.periods[period].node_blocks[node].para_import_price[t, car] = (
                    random.uniform(MC_import_row["min"], MC_import_row["max"])
                )
            else:
                model.periods[period].node_blocks[node].para_import_price[t, car] = (
                    import_prices.iloc[t - 1] * sd_random
                )

        # delete old constraint
        b_period_cost.del_component(b_period_cost.const_cost_import)

        # add new constraint
        if config["optimization"]["timestaging"]["value"] == 0:
            nr_timesteps_averaged = 1
        else:
            nr_timesteps_averaged = config["optimization"]["timestaging"]["value"]

        def init_cost_import(const):
            return model.periods[period].var_cost_imports == sum(
                sum(
                    sum(
                        model.periods[period].node_blocks[node].var_import_flow[t, car]
                        * model.periods[period]
                        .node_blocks[node]
                        .para_import_price[t, car]
                        * nr_timesteps_averaged
                        for car in model.periods[period].node_blocks[node].set_carriers
                    )
                    for t in set_t
                )
                for node in model.set_nodes
            )

        b_period_cost.const_cost_import = Constraint(rule=init_cost_import)

    def _monte_carlo_export_prices(self, period, node, car, MC_export_row=None):
        """
        Changes the export prices
        """
        config = self.data.model_config
        aggregation_type = "full"

        sd = config["optimization"]["monte_carlo"]["sd"]["value"]
        sd_random = np.random.normal(1, sd)

        model = self.model[aggregation_type]
        export_prices = self.data.time_series[aggregation_type][period][node][
            "CarrierData"
        ][car]["Export price"]
        b_period_cost = model.block_costbalance[period]
        # TODO: check for averaging
        set_t = model.periods[period].set_t_full

        # Update parameter
        for t in set_t:
            if MC_export_row is not None:
                model.periods[period].node_blocks[node].para_export_price[t, car] = (
                    random.uniform(MC_export_row["min"], MC_export_row["max"])
                )
            else:
                model.periods[period].node_blocks[node].para_export_price[t, car] = (
                    export_prices.iloc[t - 1] * sd_random
                )

        # delete old constraint
        b_period_cost.del_component(b_period_cost.const_cost_export)

        # add new constraint
        if config["optimization"]["timestaging"]["value"] == 0:
            nr_timesteps_averaged = 1
        else:
            nr_timesteps_averaged = config["optimization"]["timestaging"]["value"]

        def init_cost_export(const):
            return model.periods[period].var_cost_exports == -sum(
                sum(
                    sum(
                        model.periods[period].node_blocks[node].var_export_flow[t, car]
                        * model.periods[period]
                        .node_blocks[node]
                        .para_export_price[t, car]
                        * nr_timesteps_averaged
                        for car in model.periods[period].node_blocks[node].set_carriers
                    )
                    for t in set_t
                )
                for node in model.set_nodes
            )

        b_period_cost.const_cost_export = Constraint(rule=init_cost_export)

    def _delete_objective(self):
        """
        Delete the objective function
        """
        config = self.data.model_config
        model = self.model["full"]

        if not config["optimization"]["monte_carlo"]["N"]["value"]:
            try:
                model.del_component(model.objective)
            except:
                pass

    def _optimize_time_averaging_second_stage(self):
        """
        Optimizes the second stage of the time_averaging algorithm
        """
        self.model_information.averaged_data_specs.stage += 1
        self.model_information.averaged_data_specs.nr_timesteps_averaged = 1
        bounds_on = "no_storage"
        self.model_first_stage = self.model
        self.solution_first_stage = copy.deepcopy(self.solution)
        self.model = ConcreteModel()
        self.solution = []
        self.data = self.data_storage[0]
        self.construct_model()
        self.construct_balances()
        self._impose_size_constraints(bounds_on)
        self.solve()

    def _impose_size_constraints(self, bounds_on):
        """
        Formulates lower bound on technology and network sizes.

        It is possible to exclude storage technologies or networks by specifying bounds_on. Not this function is called
        from the method solve_model.

        :param bounds_on: can be 'all', 'only_technologies', 'only_networks', 'no_storage'
        """

        m_full = self.model["full"]
        m_avg = self.model["avg_first_stage"]

        # Technologies
        if (
            bounds_on == "all"
            or bounds_on == "only_technologies"
            or bounds_on == "no_storage"
        ):

            def size_constraint_block_tecs_init(block, node):
                def size_constraints_tecs_init(const, tec):
                    if (
                        self.data.technology_data[node][tec].technology_model == "STOR"
                        and bounds_on == "no_storage"
                    ):
                        return Constraint.Skip
                    elif self.data.technology_data[node][tec].existing:
                        return Constraint.Skip
                    else:
                        return (
                            m_avg.node_blocks[node]
                            .tech_blocks_active[tec]
                            .var_size.value
                            <= m_full.node_blocks[node].tech_blocks_active[tec].var_size
                        )

                block.size_constraints_tecs = Constraint(
                    m_full.set_technologies[node], rule=size_constraints_tecs_init
                )

            m_full.size_constraint_tecs = Block(
                m_full.set_nodes, rule=size_constraint_block_tecs_init
            )

        # Networks
        if (
            bounds_on == "all"
            or bounds_on == "only_networks"
            or bounds_on == "no_storage"
        ):

            def size_constraint_block_netw_init(block, netw):
                b_netw_full = m_full.network_block[netw]
                b_netw_avg = m_avg.network_block[netw]

                def size_constraints_netw_init(const, node_from, node_to):
                    return (
                        b_netw_full.arc_block[node_from, node_to].var_size
                        >= b_netw_avg.arc_block[node_from, node_to].var_size.value
                    )

                block.size_constraints_netw = Constraint(
                    b_netw_full.set_arcs, rule=size_constraints_netw_init
                )

            m_full.size_constraints_netw = Block(
                m_full.set_networks, rule=size_constraint_block_netw_init
            )
