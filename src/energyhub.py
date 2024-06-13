import random
from pathlib import Path
import pyomo.environ as pyo
import os
import time
import numpy as np
import pandas as pd
import sys
import datetime

from .utilities import get_set_t
from .data_management import DataHandle, read_tec_data
from .model_construction import *
from .result_management.read_results import add_values_to_summary
from .utilities import get_glpk_parameters, get_gurobi_parameters
from .logger import log_event, logger
from .result_management import *
from .components.utilities import (
    annualize,
    set_discount_rate,
    perform_disjunct_relaxation,
)


class EnergyHub:
    """
    Class to construct and manipulate an energy system model.

    When constructing an instance, it reads data to the instance and initializes all attributes of the EnergyHub
    class:

    - self.data: Data container
    - self.model: Model container
    - self.solution: Solution container
    - self.solver: Solver container
    - self.last_solve_info: Information on last solution that is written to the summary(
      pareto point, time stage,...)
    - self.info_pareto: Current pareto point (if used)
    - self.info_solving_algorithms: Information on time aggregation algorithms
    - self.info_monte_carlo: Information on monte carlo runs
    """

    def __init__(self):
        """
        Constructor
        """
        self.data = DataHandle()
        self.model = {}
        self.solution = {}
        self.solver = None
        self.last_solve_info = {}
        self.info_pareto = {}
        self.info_pareto["pareto_point"] = -1
        self.info_solving_algorithms = {}
        self.info_solving_algorithms["aggregation_model"] = "Full"
        self.info_solving_algorithms["aggregation_data"] = "Full"
        self.info_solving_algorithms["time_stage"] = 1
        self.info_monte_carlo = {}
        self.info_monte_carlo["monte_carlo_run"] = -1

    def read_data(
        self, data_path: Path | str, start_period: int = None, end_period: int = None
    ):
        """
        Reads in data from the specified path. The data is specified as the DataHandle
        class. Specifying the start_period and end_period parameter allows to run a
        time horizon than as specified in the topology (e.g. for testing)

        :param Path, str data_path: Path of folder structure to read data from
        :param int start_period: starting period of the model
        :param int end_period: end period of the model
        """
        log_event("--- Reading in data ---")
        self.data.set_settings(data_path, start_period, end_period)
        self.data.read_data()
        self._perform_preprocessing_checks()

        log_event("--- Reading in data complete ---")

    def _perform_preprocessing_checks(self):
        """
        Checks consistency of input data, before constructing or solving the model

        - Save path must exist
        - monte carlo and pareto cannot be used at the same time
        - dynamics checks
        :return:
        """
        config = self.data.model_config
        topology = self.data.topology

        # Check if save-path exists
        save_path = Path(config["reporting"]["save_path"]["value"])
        if not os.path.exists(save_path) or not os.path.isdir(save_path):
            raise FileNotFoundError(
                f"The folder '{save_path}' does not exist. Create the folder or change the folder "
                f"name in the configuration"
            )

        # Monte carlo and pareto
        if (config["optimization"]["objective"]["value"] == "pareto") and (
            config["optimization"]["monte_carlo"]["N"]["value"] > 0
        ):
            raise Exception("Monte carlo and pareto is not allowed at the same time")

        # Dynamics and time aggregation algorithms
        if config["optimization"]["typicaldays"]["N"]["value"] != 0:
            if config["performance"]["dynamics"]["value"]:
                raise Exception(
                    "Dynamics and clustering with typical days is not " "allowed"
                )
            for period in topology["investment_periods"]:
                for node in topology["nodes"]:
                    for tec_name in self.data.technology_data[period][node]:
                        tec = self.data.technology_data[period][node][tec_name]
                        if ("ramping_const_int" in tec.processed_coeff.dynamics) and (
                            tec.processed_coeff.dynamics["ramping_const_int"] != -1
                        ):
                            raise Exception(
                                f"Ramping constraint with integers (ramping_const_int) for technology {tec_name} "
                                f"needs to be -1 when clustering with typical days"
                            )

        if config["optimization"]["timestaging"]["value"] != 0:
            if config["performance"]["dynamics"]["value"]:
                raise Exception(
                    "Dynamics and two-stage averaging algorithm is not " "allowed"
                )
            for period in topology["investment_periods"]:
                for node in topology["nodes"]:
                    for tec_name in self.data.technology_data[period][node]:
                        tec = self.data.technology_data[period][node][tec_name]
                        if ("ramping_time" in tec.processed_coeff.dynamics) and (
                            tec.processed_coeff.dynamics["ramping_time"] != -1
                        ):
                            raise Exception(
                                f"Ramping Rate for technology {tec_name} "
                                f"needs to be -1 when two-stage averaging algorithm is used"
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
                        for par in par_check:
                            if (
                                par
                                not in self.data.technology_data[node][
                                    tec
                                ].processed_coeff.dynamics
                            ):
                                raise ValueError(
                                    f"The technology '{tec}' does not have dynamic parameter '{par}'. Add the parameters in the "
                                    f"json files or switch off the dynamics."
                                )

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

        # Determine aggregation
        config = self.data.model_config

        # Clustered data
        self.info_solving_algorithms["aggregation_model"] = "full"
        self.info_solving_algorithms["aggregation_data"] = "full"

        if config["optimization"]["typicaldays"]["N"]["value"] != 0:
            if config["optimization"]["typicaldays"]["method"]["value"] == 1:
                self.info_solving_algorithms["aggregation_model"] = "clustered"
                self.info_solving_algorithms["aggregation_data"] = "clustered"
            elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
                self.info_solving_algorithms["aggregation_model"] = "clustered"
                self.info_solving_algorithms["aggregation_data"] = "full"
            else:
                raise Exception("clustering method needs to be 1 or 2")

        # Averaged data
        if config["optimization"]["timestaging"]["value"] != 0:
            if self.info_solving_algorithms["time_stage"] == 1:
                self.info_solving_algorithms["aggregation_model"] = "averaged"
                self.info_solving_algorithms["aggregation_data"] = "averaged"

        # INITIALIZE MODEL
        aggregation_model = self.info_solving_algorithms["aggregation_model"]
        aggregation_data = self.info_solving_algorithms["aggregation_data"]
        self.model[aggregation_model] = pyo.ConcreteModel()

        # GET DATA
        model = self.model[aggregation_model]
        topology = self.data.topology
        config = self.data.model_config

        # DEFINE GLOBAL SETS
        # Nodes, Carriers, Technologies, Networks
        model.set_periods = pyo.Set(initialize=topology["investment_periods"])
        model.set_nodes = pyo.Set(initialize=topology["nodes"])
        model.set_carriers = pyo.Set(initialize=topology["carriers"])

        # DEFINE GLOBAL VARIABLES
        model.var_npv = pyo.Var()
        model.var_emissions_net = pyo.Var()

        # INVESTMENT PERIOD BLOCK
        def init_period_block(b_period):
            """Pyomo rule to initialize a block holding all investment periods"""

            # Get data for investment period
            investment_period = b_period.index()
            data_period = get_data_for_investment_period(
                self.data, investment_period, aggregation_data
            )
            # Add sets, parameters, variables, constraints to block
            b_period = construct_investment_period_block(b_period, data_period)

            # NETWORK BLOCK
            if not config["energybalance"]["copperplate"]["value"]:

                def init_network_block(b_netw, netw):
                    """Pyomo rule to initialize a block holding all networks"""
                    # Add sets, parameters, variables, constraints to block
                    b_netw = construct_network_block(
                        b_netw,
                        data_period,
                        model.set_nodes,
                        b_period.set_t_full,
                        b_period.set_t_clustered,
                    )

                    return b_netw

                b_period.network_block = pyo.Block(
                    b_period.set_networks, rule=init_network_block
                )

            # NODE BLOCK
            def init_node_block(b_node, node):
                """Pyomo rule to initialize a block holding all nodes"""
                # Get data for node
                data_node = get_data_for_node(data_period, node)

                # Add sets, parameters, variables, constraints to block
                b_node = construct_node_block(
                    b_node, data_node, b_period.set_t_full, b_period.set_t_clustered
                )

                # TECHNOLOGY BLOCK
                def init_technology_block(b_tec, tec):
                    b_tec = construct_technology_block(
                        b_tec, data_node, b_period.set_t_full, b_period.set_t_clustered
                    )

                    return b_tec

                b_node.tech_blocks_active = pyo.Block(
                    b_node.set_technologies, rule=init_technology_block
                )

                return b_node

            b_period.node_blocks = pyo.Block(model.set_nodes, rule=init_node_block)

            return b_period

        model.periods = pyo.Block(model.set_periods, rule=init_period_block)

        log_event(f"Constructing model completed in {str(round(time.time() - start))}s")

    def construct_balances(self):
        """
        Constructs the energy balance, emission balance and calculates costs
        """
        log_event("Constructing balances...")
        start = time.time()

        config = self.data.model_config
        data = self.data
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

        model = delete_all_balances(model)

        if not config["energybalance"]["copperplate"]["value"]:
            model = construct_network_constraints(model, config)
            model = construct_nodal_energybalance(model, config)
        else:
            model = construct_global_energybalance(model, config)

        model = construct_emission_balance(model, data)
        model = construct_system_cost(model, data)
        model = construct_global_balance(model)

        log_event(
            f"Constructing balances completed in {str(round(time.time() - start))}s"
        )

    def solve(self):
        """
        Defines objective and solves model
        """
        config = self.data.model_config

        objective = config["optimization"]["objective"]["value"]

        self._define_solver_settings()

        if config["optimization"]["monte_carlo"]["N"]["value"]:
            self._solve_monte_carlo(objective)
        elif objective == "pareto":
            self._solve_pareto()
        else:
            self._optimize(objective)

    def quick_solve(self):
        """
        Quick-solves the model (constructs model and balances and solves model).

        This method lumbs together the following functions for convenience:
        - :func:`~src.energyhub.construct_model`
        - :func:`~src.energyhub.construct_balances`
        - :func:`~src.energyhub.solve`
        """
        self.construct_model()
        self.construct_balances()
        self.solve()

    def write_results(self):
        """
        Writes optimization results of a model run to folder
        """
        # Write H5 File
        config = self.data.model_config

        save_summary_path = Path.joinpath(
            Path(config["reporting"]["save_summary_path"]["value"]), "Summary.xlsx"
        )

        model_info = self.last_solve_info

        model = self.model[self.info_solving_algorithms["aggregation_model"]]

        summary_dict = write_optimization_results_to_h5(
            model, self.solution, model_info, self.data
        )

        # Write Summary
        if not os.path.exists(save_summary_path):
            summary_df = pd.DataFrame(data=summary_dict, index=[0])
            summary_df.to_excel(save_summary_path, index=False, sheet_name="Summary")
        else:
            summary_existing = pd.read_excel(save_summary_path)
            pd.concat(
                [summary_existing, pd.DataFrame(data=summary_dict, index=[0])]
            ).to_excel(save_summary_path, index=False, sheet_name="Summary")

    def add_technology(self, investment_period: str, node: str, technologies: list):
        """
        Adds technologies retrospectively to the model.

        After adding a technology to a node, all balances need to be re-constructed,
        To solve the model again run, :func:`~construct_balances` and then
        :func:`~solve`.

        :param str investment_period: name of investment period for which technology is added
        :param str node: name of node for which technology is added
        :param list technologies: list of technologies that should be added
        :return None:
        """
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

        # Make sure that no aggregation algorithm is used
        config = self.data.model_config
        if (config["optimization"]["typicaldays"]["N"]["value"] != 0) or (
            config["optimization"]["timestaging"]["value"] != 0
        ):
            raise Exception(
                "You cannot add a technolgy retrospectively if using time aggragation algorithms"
            )

        # Read technology data
        data_node = {
            "technology_data": {},
            "config": config,
            "topology": self.data.topology,
        }
        for technology in technologies:
            # read in technology data
            tec_data = read_tec_data(
                technology,
                self.data.data_path
                / investment_period
                / "node_data"
                / node
                / "technology_data",
            )
            # fit technology data
            tec_data.fit_technology_performance(
                self.data.time_series["full"][investment_period][node]["ClimateData"][
                    "global"
                ],
                self.data.node_locations.loc[node, :],
            )
            # add technology data to data handle
            self.data.technology_data[investment_period][node][technology] = tec_data
            data_node["technology_data"][technology] = tec_data

        # Add technology to node
        b_period = model.periods[investment_period]
        b_node = b_period.node_blocks[node]

        # Create new technology block containing all new technologies
        def init_technology_block(b_tec, tec):
            b_tec = construct_technology_block(
                b_tec, data_node, b_period.set_t_full, b_period.set_t_clustered
            )

            return b_tec

        b_node.tech_blocks_new = pyo.Block(technologies, rule=init_technology_block)

        # If it exists, carry over active tech blocks to temporary block
        if b_node.find_component("tech_blocks_active"):
            b_node.tech_blocks_existing = pyo.Block(b_node.set_technologies)
            for tec in b_node.set_technologies:
                b_node.tech_blocks_existing[tec].transfer_attributes_from(
                    b_node.tech_blocks_active[tec]
                )
            b_node.del_component(b_node.tech_blocks_active)
        if b_node.find_component("tech_blocks_active_index"):
            b_node.del_component(b_node.tech_blocks_active_index)

        # Create a block containing all active technologies at node
        if not set(technologies).issubset(b_node.set_technologies):
            b_node.set_technologies.add(technologies)

        def init_active_technology_blocks(bl, tec):
            if tec in technologies:
                bl.transfer_attributes_from(b_node.tech_blocks_new[tec])
            else:
                bl.transfer_attributes_from(b_node.tech_blocks_existing[tec])

        b_node.tech_blocks_active = pyo.Block(
            b_node.set_technologies, rule=init_active_technology_blocks
        )

        # Delete all auxiliary blocks
        if b_node.find_component("tech_blocks_new"):
            b_node.del_component(b_node.tech_blocks_new)
        if b_node.find_component("tech_blocks_new_index"):
            b_node.del_component(b_node.tech_blocks_new_index)
        if b_node.find_component("tech_blocks_existing"):
            b_node.del_component(b_node.tech_blocks_existing)
        if b_node.find_component("tech_blocks_existing_index"):
            b_node.del_component(b_node.tech_blocks_existing_index)

    def _define_solver_settings(self):
        """
        Defines solver and its settings depending on objective and solver
        """
        config = self.data.model_config
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

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
            self.solver = get_glpk_parameters(config["solveroptions"])

        # For persistent solver, set model instance
        if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
            self.solver.set_instance(model)

    def _optimize(self, objective):
        """
        Solves the model with the given objective
        """
        config = self.data.model_config

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
        if config["optimization"]["timestaging"]["value"] != 0:
            self.info_solving_algorithms["time_stage"] = 2
            config["optimization"]["timestaging"]["value"] = 0
            self.info_solving_algorithms["objective"] = objective
            self._optimize_time_averaging_second_stage()

    def _optimize_cost(self):
        """
        Minimizes Costs
        """
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

        self._delete_objective()

        def init_cost_objective(obj):
            return model.var_npv

        model.objective = pyo.Objective(rule=init_cost_objective, sense=pyo.minimize)
        log_event("Set objective on cost")
        self._call_solver()

    def _optimize_emissions_net(self):
        """
        Minimize net emissions
        """
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

        self._delete_objective()

        def init_emission_net_objective(obj):
            return model.var_emissions_net

        model.objective = pyo.Objective(
            rule=init_emission_net_objective, sense=pyo.minimize
        )
        log_event("Set objective on net emissions")
        self._call_solver()

    def _optimize_costs_emissionslimit(self):
        """
        Minimize costs at emission limit
        """
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

        config = self.data.model_config

        emission_limit = config["optimization"]["emission_limit"]["value"]
        if model.find_component("const_emission_limit"):
            if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
                self.solver.remove_constraint(model.const_emission_limit)
            model.del_component(model.const_emission_limit)
        model.const_emission_limit = pyo.Constraint(
            expr=model.var_emissions_net <= emission_limit
        )
        if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
            self.solver.add_constraint(model.const_emission_limit)
        log_event("Defined constraint on net emissions")
        self._optimize_cost()

    def _optimize_costs_minE(self):
        """
        Minimize costs at minimum emissions
        """
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

        config = self.data.model_config

        self._optimize_emissions_net()
        emission_limit = model.var_emissions_net.value
        if model.find_component("const_emission_limit"):
            if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
                self.solver.remove_constraint(model.const_emission_limit)
            model.del_component(model.const_emission_limit)
        model.const_emission_limit = pyo.Constraint(
            expr=model.var_emissions_net <= emission_limit * 1.001
        )
        if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
            self.solver.add_constraint(model.const_emission_limit)
        self._optimize_cost()

    def scale_model(self):
        """
        Creates a scaled model using the scale factors specified in the json files
        for technologies and networks as well as the global scaling factors
        specified. See also the documentation on model scaling.
        """
        config = self.data.model_config

        f_global = config["scaling_factors"]["value"]
        model_full = self.model[self.info_solving_algorithms["aggregation_model"]]

        model_full.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

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

        self.model["scaled"] = pyo.TransformationFactory(
            "core.scale_model"
        ).create_using(model_full)

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
        if self.info_pareto["pareto_point"]:
            folder_name = folder_name + str(self.info_pareto["pareto_point"])

        result_folder_path = create_unique_folder_name(save_path, folder_name)
        create_save_folder(result_folder_path)

        # Scale model
        if config["scaling"]["scaling_on"]["value"] == 1:
            self.scale_model()
            model = self.model["scaled"]
        else:
            model = self.model[self.info_solving_algorithms["aggregation_model"]]

        # Call solver
        if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
            self.solver.set_objective(model.objective)

        if config["solveroptions"]["solver"]["value"] == "glpk":
            self.solution = self.solver.solve(
                model,
                tee=True,
                logfile=str(Path(result_folder_path / "solver_log.txt")),
                keepfiles=True,
            )
        else:
            self.solution = self.solver.solve(
                model,
                tee=True,
                warmstart=True,
                logfile=str(Path(result_folder_path / "solver_log.txt")),
                keepfiles=True,
            )

        if config["scaling"]["scaling_on"]["value"] == 1:
            pyo.TransformationFactory("core.scale_model").propagate_solution(
                model, self.model[self.info_solving_algorithms["aggregation_model"]]
            )

        if config["reporting"]["write_solution_diagnostics"]["value"] >= 1:
            self._write_solution_diagnostics(result_folder_path)

        self.solution.write()

        self.last_solve_info["pareto_point"] = self.info_pareto["pareto_point"]
        self.last_solve_info["monte_carlo_run"] = self.info_monte_carlo[
            "monte_carlo_run"
        ]
        self.last_solve_info["config"] = config
        self.last_solve_info["result_folder_path"] = result_folder_path
        self.last_solve_info["time_stage"] = self.info_solving_algorithms["time_stage"]

        # Write results to path
        # Determine if results should be written
        write_results = False
        if (self.solution.solver.status == pyo.SolverStatus.ok) or (
            self.solution.solver.status == pyo.SolverStatus.warning
        ):
            write_results = True
        if self.solution.solver.termination_condition in [
            pyo.TerminationCondition.infeasibleOrUnbounded,
            pyo.TerminationCondition.infeasible,
            pyo.TerminationCondition.unbounded,
        ]:
            write_results = False

        if write_results:
            self.write_results()

        print("Solving model completed in " + str(round(time.time() - start)) + " s")
        print("_" * 60)

    def _write_solution_diagnostics(self, save_path):
        """
        Can write solution quality, constraint map and variable map to file. Options
        are specified in the configuration.

        :param save_path:
        :return:
        """
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

    def _solve_pareto(self):
        """
        Optimize the pareto front
        """
        model = self.model[self.info_solving_algorithms["aggregation_model"]]
        config = self.data.model_config
        pareto_points = config["optimization"]["pareto_points"]["value"]

        # Min Emissions
        self.info_pareto["pareto_point"] = pareto_points
        self._optimize_costs_minE()
        emissions_min = model.var_emissions_net.value

        # Min Cost
        self.info_pareto["pareto_point"] = 1
        self._optimize_cost()
        emissions_max = model.var_emissions_net.value

        # Emission limit
        emission_limits = np.linspace(emissions_max, emissions_min, num=pareto_points)[
            1:-1
        ]

        for limit in range(0, len(emission_limits)):
            self.info_pareto["pareto_point"] += 1
            log_event(f"Optimizing Pareto point {limit}")
            if limit != 0:
                # If its not the first point, delete constraint
                if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
                    self.solver.remove_constraint(model.const_emission_limit)
                model.del_component(model.const_emission_limit)
            model.const_emission_limit = pyo.Constraint(
                expr=model.var_emissions_net <= emission_limits[limit] * 1.005
            )
            if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
                self.solver.add_constraint(model.const_emission_limit)
            self._optimize("costs")

    def _solve_monte_carlo(self, objective: str):
        """
        Optimizes multiple runs with monte carlo

        :param str objective: objective to optimize
        """
        config = self.data.model_config
        self.info_monte_carlo["monte_carlo_run"] = 0

        for run in range(0, config["optimization"]["monte_carlo"]["N"]["value"]):
            self.info_monte_carlo["monte_carlo_run"] = run
            self._monte_carlo_set_cost_parameters()
            if run == 0:
                # in this case we need to set the objective
                self._optimize(objective)
            else:
                # in this case we can call the solver directly
                self._call_solver()

        summary_path = Path.joinpath(
            Path(config["reporting"]["save_summary_path"]["value"]), "Summary.xlsx"
        )
        if config["optimization"]["monte_carlo"]["type"]["value"] == "normal_dis":
            component_set = config["optimization"]["monte_carlo"]["on_what"]["value"]
        elif (
            config["optimization"]["monte_carlo"]["type"]["value"]
            == "uniform_dis_from_file"
        ):
            component_set = list(set(self.data.monte_carlo_specs["type"]))
        add_values_to_summary(summary_path, component_set=component_set)

    def _monte_carlo_set_cost_parameters(self):
        """
        Changes cost parameters for monte carlo analysis.
        """
        config = self.data.model_config

        # use correct resolution
        model = self.model[self.info_solving_algorithms["aggregation_model"]]
        monte_carlo_type = config["optimization"]["monte_carlo"]["type"]["value"]
        monte_carlo_on = config["optimization"]["monte_carlo"]["on_what"]["value"]

        import_constraint_reconstruction = False
        export_constraint_reconstruction = False

        if monte_carlo_type == "normal_dis":
            if "Technologies" in monte_carlo_on:
                for period in model.periods:
                    for node in model.periods[period].node_blocks:
                        for tec in (
                            model.periods[period].node_blocks[node].tech_blocks_active
                        ):
                            self._monte_carlo_technologies(period, node, tec)

            if "Networks" in monte_carlo_on:
                for period in model.periods:
                    for netw in model.periods[period].network_block:
                        self._monte_carlo_networks(period, netw)

            if "Import" in monte_carlo_on:
                self._monte_carlo_import_parameters()
                import_constraint_reconstruction = True

            if "Export" in monte_carlo_on:
                self._monte_carlo_export_parameters()
                export_constraint_reconstruction = True

        elif monte_carlo_type == "uniform_dis_from_file":
            MC_parameters = self.data.monte_carlo_specs
            processed_names = set()

            for index, row in MC_parameters.iterrows():
                if row["type"] == "Technologies":
                    tec = row["name"]

                    # Iterate through periods and nodes in the model
                    if tec not in processed_names:
                        for period in model.periods:
                            for node in model.periods[period].node_blocks:
                                tech_blocks = (
                                    model.periods[period]
                                    .node_blocks[node]
                                    .tech_blocks_active
                                )

                                # Check if the technology is active in the current node
                                if tec in tech_blocks:
                                    capex_model = self.data.technology_data[period][
                                        node
                                    ][tec].economics.capex_model

                                    if capex_model == 1:
                                        if row["parameter"] == "unit_CAPEX":
                                            self._monte_carlo_technologies(
                                                period, node, tec, row
                                            )
                                        else:
                                            new_row = MC_parameters[
                                                (MC_parameters["name"] == tec)
                                                & (
                                                    MC_parameters["parameter"]
                                                    == "unit_CAPEX"
                                                )
                                            ]
                                            if not new_row.empty:
                                                self._monte_carlo_technologies(
                                                    period, node, tec, new_row
                                                )
                                            else:
                                                log_event(
                                                    f"Parameter unit_CAPEX is not defined for {tec} in MonteCarlo.csv",
                                                    level="warning",
                                                )

                                    else:
                                        # Find all rows with the same technology name
                                        MC_technology_rows = MC_parameters[
                                            (MC_parameters["type"] == "Technologies")
                                            & (MC_parameters["name"] == tec)
                                        ]
                                        self._monte_carlo_technologies(
                                            period, node, tec, MC_technology_rows
                                        )

                                    processed_names.add(tec)
                                    break
                                else:
                                    log_event(
                                        f"Technology {tec} in MonteCarlo.csv is not an active component",
                                        level="warning",
                                    )

                elif row["type"] == "Networks":
                    netw = row["name"]
                    if netw not in processed_names:
                        for period in model.periods:
                            netw_blocks = model.periods[period].network_block
                            if netw in netw_blocks:
                                MC_network_rows = MC_parameters[
                                    (MC_parameters["type"] == "Networks")
                                    & (MC_parameters["name"] == netw)
                                ]

                                self._monte_carlo_networks(
                                    period, netw, MC_network_rows
                                )

                                processed_names.add(netw)
                                break
                            else:
                                log_event(
                                    f"Network {netw} in MonteCarlo.csv is not active component",
                                    level="warning",
                                )

                elif row["type"] == "Import":
                    import_constraint_reconstruction = True
                    car = row["name"]
                    self._monte_carlo_import_parameters(car, row)
                elif row["type"] == "Export":
                    export_constraint_reconstruction = True
                    car = row["name"]
                    self._monte_carlo_export_parameters(car, row)

        if import_constraint_reconstruction:
            self._monte_carlo_import_constraints()

        if export_constraint_reconstruction:
            self._monte_carlo_export_constraints()

    def _monte_carlo_technologies(self, period, node, tec, MC_ranges=None):
        """
        Changes the capex of technologies
        """
        aggregation_model = self.info_solving_algorithms["aggregation_model"]
        aggregation_data = self.info_solving_algorithms["aggregation_data"]

        config = self.data.model_config
        tec_data = self.data.technology_data[period][node][tec]
        model = self.model[aggregation_model]

        if tec_data.economics.capex_model in [1, 3]:
            # Preprocessing
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
                if MC_ranges is not None:
                    if isinstance(MC_ranges, pd.Series):
                        unit_capex = random.uniform(MC_ranges["min"], MC_ranges["max"])
                    elif isinstance(MC_ranges, pd.DataFrame):
                        unit_capex = random.uniform(
                            MC_ranges["min"].iloc[0], MC_ranges["max"].iloc[0]
                        )
                else:
                    unit_capex = tec_data.economics.capex_data["unit_capex"] * sd_random

                b_tec.para_unit_capex = unit_capex
                b_tec.para_unit_capex_annual = unit_capex * annualization_factor

            elif tec_data.economics.capex_model == 3:
                if MC_ranges is not None:
                    for _, row in MC_ranges.iterrows():
                        if row["parameter"] == "unit_CAPEX":
                            unit_capex = random.uniform(row["min"], row["max"])
                        elif row["parameter"] == "fix_CAPEX":
                            fix_capex = random.uniform(row["min"], row["max"])
                else:
                    unit_capex = tec_data.economics.capex_data["unit_capex"] * sd_random
                    fix_capex = tec_data.economics.capex_data["fix_capex"] * sd_random

                b_tec.para_unit_capex = unit_capex
                b_tec.para_unit_capex_annual = unit_capex * annualization_factor
                b_tec.para_fix_capex = fix_capex
                b_tec.para_fix_capex_annual = fix_capex * annualization_factor

            # Change variable bounds
            def calculate_max_capex():
                if economics.capex_model == 1:
                    max_capex = b_tec.para_size_max * b_tec.para_unit_capex_annual
                    bounds = (0, max_capex)
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
                b_tec.del_component(b_tec.const_capex)
            elif economics.capex_model == 3:
                big_m_transformation_required = 1
                b_tec.del_component(b_tec.dis_installation)
                b_tec.del_component(b_tec.disjunction_installation)
                b_tec.del_component(b_tec._pyomo_gdp_bigm_reformulation)

            # Reconstruct technology constraints
            data_period = get_data_for_investment_period(
                self.data, period, aggregation_data
            )
            data_node = get_data_for_node(data_period, node)

            b_tec = tec_data._define_capex_constraints(b_tec, data_node)
            if big_m_transformation_required:
                b_tec = perform_disjunct_relaxation(b_tec)

        else:
            log_event(
                "monte carlo for capex models other than 1 and 3 is not implemented",
                level="warning",
            )

    def _monte_carlo_networks(self, period, netw, MC_ranges=None):
        """
        Changes the capex of networks
        """
        aggregation_model = self.info_solving_algorithms["aggregation_model"]

        config = self.data.model_config
        model = self.model[aggregation_model]

        sd = config["optimization"]["monte_carlo"]["sd"]["value"]
        sd_random = np.random.normal(1, sd)

        netw_data = self.data.network_data[period][netw]
        economics = netw_data.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = self.data.topology["fraction_of_year_modelled"]

        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )

        b_netw = model.periods[period].network_block[netw]

        if MC_ranges is not None:
            for _, row in MC_ranges.iterrows():
                if row["parameter"] == "gamma1":
                    b_netw.para_capex_gamma1 = random.uniform(row["min"], row["max"])
                elif row["parameter"] == "gamma2":
                    b_netw.para_capex_gamma2 = random.uniform(row["min"], row["max"])
                elif row["parameter"] == "gamma3":
                    b_netw.para_capex_gamma3 = random.uniform(row["min"], row["max"])
                elif row["parameter"] == "gamma4":
                    b_netw.para_capex_gamma4 = random.uniform(row["min"], row["max"])
        else:
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

            def calculate_max_capex():
                max_capex = (
                    b_netw.para_capex_gamma1
                    + b_netw.para_capex_gamma2 * b_arc.para_size_max
                    + b_netw.para_capex_gamma3 * b_arc.distance
                    + b_netw.para_capex_gamma4 * b_arc.para_size_max * b_arc.distance
                )
                return (0, max_capex)

            bounds = calculate_max_capex()
            b_arc.var_capex_aux.setlb(bounds[0])
            b_arc.var_capex_aux.setub(bounds[1])
            b_arc.var_capex.setlb(bounds[0])
            b_arc.var_capex.setub(bounds[1])

            # Remove constraint (from persistent solver and from model)
            b_arc.del_component(b_arc._pyomo_gdp_bigm_reformulation)
            b_arc.del_component(b_arc.const_capex)
            b_arc.del_component(b_arc.dis_installation)
            b_arc.del_component(b_arc.disjunction_installation)

            b_arc = netw_data._define_capex_constraints_arc(
                b_arc, b_netw, arc[0], arc[1]
            )

            if b_arc.big_m_transformation_required:
                b_arc = perform_disjunct_relaxation(b_arc)

    def _monte_carlo_import_parameters(self, on_car=None, MC_ranges=None):
        """
        Changes the import prices
        """
        aggregation_model = self.info_solving_algorithms["aggregation_model"]
        aggregation_data = self.info_solving_algorithms["aggregation_data"]

        config = self.data.model_config
        model = self.model[aggregation_model]

        for period in model.periods:
            b_period = model.periods[period]
            set_t = get_set_t(config, b_period)

            for node in b_period.node_blocks:
                if on_car is None:
                    # change for all carriers at node
                    on_car = model.periods[period].node_blocks[node].set_carriers

                for car in model.periods[period].node_blocks[node].set_carriers:
                    if car in on_car:
                        import_prices = self.data.time_series[aggregation_data][
                            period, node, "CarrierData", car, "Import price"
                        ]

                        for t in set_t:
                            if MC_ranges is None:
                                sd = config["optimization"]["monte_carlo"]["sd"][
                                    "value"
                                ]
                                random_factor = np.random.normal(1, sd)
                            else:
                                random_factor = random.uniform(
                                    MC_ranges["min"], MC_ranges["max"]
                                )

                            # Update parameter
                            model.periods[period].node_blocks[node].para_import_price[
                                t, car
                            ] = (import_prices.iloc[t - 1] * random_factor)

    def _monte_carlo_import_constraints(self):
        """
        Reconstructs the import cost constraint
        """
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

        # reconstruct constraint
        for period in model.periods:
            b_period = model.periods[period]
            b_period_cost = model.block_costbalance[period]

            b_period_cost.del_component(b_period_cost.const_cost_import)
            b_period_cost.const_cost_import = construct_import_costs(
                b_period, self.data, period
            )

    def _monte_carlo_export_parameters(self, on_car=None, MC_ranges=None):
        """
        Changes the import prices
        """
        aggregation_model = self.info_solving_algorithms["aggregation_model"]
        aggregation_data = self.info_solving_algorithms["aggregation_data"]

        config = self.data.model_config
        model = self.model[aggregation_model]

        for period in model.periods:
            b_period = model.periods[period]
            set_t = get_set_t(config, b_period)

            for node in b_period.node_blocks:
                if on_car is None:
                    # change for all carriers at node
                    on_car = model.periods[period].node_blocks[node].set_carriers

                for car in model.periods[period].node_blocks[node].set_carriers:
                    if car in on_car:
                        export_prices = self.data.time_series[aggregation_data][
                            period, node, "CarrierData", car, "Export price"
                        ]

                        for t in set_t:
                            if MC_ranges is None:
                                sd = config["optimization"]["monte_carlo"]["sd"][
                                    "value"
                                ]
                                random_factor = np.random.normal(1, sd)
                            else:
                                random_factor = random.uniform(
                                    MC_ranges["min"], MC_ranges["max"]
                                )

                            # Update parameter
                            model.periods[period].node_blocks[node].para_export_price[
                                t, car
                            ] = (export_prices.iloc[t - 1] * random_factor)

    def _monte_carlo_export_constraints(self):
        """
        Reconstructs the import cost constraint
        """
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

        # reconstruct constraint
        for period in model.periods:
            b_period = model.periods[period]
            b_period_cost = model.block_costbalance[period]

            b_period_cost.del_component(b_period_cost.const_cost_export)
            b_period_cost.const_cost_export = construct_export_costs(
                b_period, self.data, period
            )

    def _delete_objective(self):
        """
        Delete the objective function
        """
        config = self.data.model_config
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

        if not config["optimization"]["monte_carlo"]["N"]["value"]:
            try:
                model.del_component(model.objective)
            except:
                pass

    def _optimize_time_averaging_second_stage(self):
        """
        Optimizes the second stage of the time_averaging algorithm
        """
        # Todo: make it possible to chose (config)
        bounds_on = "no_storage"
        self.construct_model()
        self.construct_balances()
        self._impose_size_constraints(bounds_on)
        self._optimize(self.info_solving_algorithms["objective"])

    def _impose_size_constraints(self, bounds_on):
        """
        Formulates lower bound on technology and network sizes.

        It is possible to exclude storage technologies or networks by specifying
        bounds_on.

        :param bounds_on: can be 'all', 'only_technologies', 'only_networks', 'no_storage'
        """

        m_full = self.model["full"]
        m_avg = self.model["averaged"]

        # Technologies
        if (
            bounds_on == "all"
            or bounds_on == "only_technologies"
            or bounds_on == "no_storage"
        ):

            def size_constraint_block_tecs_init(block, period, node):
                def size_constraints_tecs_init(const, tec):
                    if (
                        self.data.technology_data[period][node][
                            tec
                        ].component_options.technology_model
                        == "STOR"
                        and bounds_on == "no_storage"
                    ):
                        return pyo.Constraint.Skip
                    elif self.data.technology_data[period][node][tec].existing:
                        return pyo.Constraint.Skip
                    else:
                        log_event(
                            f"Size constraint imposed on {tec} at {node} in "
                            f"{period}"
                        )
                        return (
                            m_avg.periods[period]
                            .node_blocks[node]
                            .tech_blocks_active[tec]
                            .var_size.value
                            <= m_full.periods[period]
                            .node_blocks[node]
                            .tech_blocks_active[tec]
                            .var_size
                        )

                block.size_constraints_tecs = pyo.Constraint(
                    m_full.periods[period].node_blocks[node].set_technologies,
                    rule=size_constraints_tecs_init,
                )

            m_full.size_constraint_tecs = pyo.Block(
                m_full.set_periods,
                m_full.set_nodes,
                rule=size_constraint_block_tecs_init,
            )

        # Networks
        if (
            bounds_on == "all"
            or bounds_on == "only_networks"
            or bounds_on == "no_storage"
        ):

            def size_constraint_block_netw_init(block, period):

                def size_constraints_netw_init(const, netw):

                    b_netw_full = m_full.periods[period].network_block[netw]
                    b_netw_avg = m_avg.periods[period].network_block[netw]

                    log_event(f"Size constraint imposed on {netw} in " f"{period}")

                    def size_constraints_arcs_init(const, node_from, node_to):
                        return (
                            b_netw_full.arc_block[node_from, node_to].var_size
                            >= b_netw_avg.arc_block[node_from, node_to].var_size.value
                        )

                    block.size_constraints_arcs = pyo.Constraint(
                        b_netw_full.set_arcs, rule=size_constraints_arcs_init
                    )

                block.size_constraints_netw = pyo.Block(
                    m_full.periods[period].set_networks, rule=size_constraints_netw_init
                )

            m_full.size_constraints_netw = pyo.Block(
                m_full.set_periods, rule=size_constraint_block_netw_init
            )
