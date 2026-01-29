import warnings
from pathlib import Path
import pyomo.environ as pyo
import os
import time
import numpy as np
import pandas as pd
import sys
import datetime
import logging
import json

from adopt_net0.plugins.plugin_manager import PluginManager
from adopt_net0.plugins.hooks import Hook

from adopt_net0.core.registries import TechnologyRegistry, NetworkRegistry

from adopt_net0.core.data_management import read_topology, read_config, read_time_series_data, read_network_data, read_technology_data
from adopt_net0.core.data_management.utilities import check_input_data_consistency
from adopt_net0.core.utilities import get_glpk_parameters, get_gurobi_parameters
from adopt_net0.core.result_management import *
from adopt_net0.core.model_construction.construct_components import construct_components
from adopt_net0.core.model_construction.construct_balances import construct_balances


log = logging.getLogger(__name__)


class ModelHub:
    """
    Class to construct and manipulate an energy system model.

    When constructing an instance, it reads data to the instance and initializes all
    attributes of the ModelHub class:

    - self.data: Data container
    - self.model: Model container
    - self.solution: Solution container
    - self.solver: Solver container
    - self.last_solve_info: Information on last solution that is written to the summary(
      pareto point, time stage,...)
    - self.info_pareto: Current pareto point (if used)
    - self.info_solving_algorithms: Information on time aggregation algorithms
    """

    def __init__(self):
        """
        Constructor
        """
        self.data = {}
        self.component_constructors = {}
        self.component_constructors["technology_constructors"] = {}
        self.component_constructors["network_constructors"] = {}

        self.model = {}
        self.solution = {}
        self.solver = None
        self.last_solve_info = {}
        self.info_pareto = {}
        self.info_pareto["pareto_point"] = -1
        self.info_solving_algorithms = {}
        self.info_solving_algorithms["aggregation_model"] = None
        self.info_solving_algorithms["aggregation_data"] = None
        self.info_solving_algorithms["time_stage"] = None

        self.plugin_manager = PluginManager()

        self.technology_registry = TechnologyRegistry()
        self.technology_registry.register_builtin_technologies()

        self.network_registry = NetworkRegistry()
        self.network_registry.register_builtin_networks()

    def read_data(
            self, data_path: Path | str, start_period: int = None, end_period: int = None
    ):
        """
        Reads in data from the specified path.

        Specifying the start_period and end_period parameter allows to run a
        time horizon than as specified in the topology (e.g. for testing)
        The function reads in topology, config and system data and initializes the component constructors.

        :param Path, str data_path: Path of folder structure to read data from
        :param int start_period: starting period of the model
        :param int end_period: end period of the model
        """
        data_path = Path(data_path)

        # Register plugins
        with open(data_path / "Plugins.json", "r") as json_file:
            plugin_list = json.load(json_file)
        self.plugin_manager.register(plugin_list)

        # Perform input data consistency check
        check_input_data_consistency(data_path)

        # Register custom components
        self.plugin_manager.emit(Hook.TECHNOLOGY_REGISTRATION, technology_registry=self.technology_registry)
        self.plugin_manager.emit(Hook.NETWORK_REGISTRATION, network_registry=self.network_registry)

        # Read data
        self.plugin_manager.emit(Hook.DATA_READ_START, data_path=data_path)
        self._read_data(data_path, start_period, end_period)

        # Initialize component constructors
        self._initialize_technology_constructors()
        self._initialize_network_constructors()
        self.plugin_manager.emit(Hook.DATA_READ_END, modelhub=self)

    def _initialize_technology_constructors(self):
        """
        Initialize technology constructors from technology data
        """
        for period, nodes in self.data["technology_data"].items():
            self.component_constructors["technology_constructors"][period] = {}
            for node, technologies in nodes.items():
                self.component_constructors["technology_constructors"][period][node] = {}
                for technology, tec_data in technologies.items():
                    tec_type = tec_data["tec_type"]
                    constructor = self.technology_registry.create(tec_type, tec_data)
                    self.component_constructors["technology_constructors"][period][node][technology] = constructor

    def _initialize_network_constructors(self):
        """
        Initialize network constructors from technology data
        """
        for period, networks in self.data["network_data"].items():
            self.component_constructors["network_constructors"][period] = {}
            for network, network_data in networks.items():
                netw_type = network_data["netw_type"]
                constructor = self.network_registry.create(netw_type, network_data)
                self.component_constructors["network_constructors"][period][network] = constructor

    def _read_data(
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
        if isinstance(data_path, str):
            data_path = Path(data_path)

        log_msg = "--- Reading in data ---"
        print(log_msg)
        log.info(log_msg)

        # Read in data
        topology = read_topology(data_path, start_period, end_period)
        # Trim time series
        time_series = read_time_series_data(data_path, topology)
        if start_period is not None and end_period is not None:
            time_series = time_series.iloc[start_period: end_period]

        # Store full resolution data
        self.data["data_path"] = data_path
        self.data["topology"] = topology
        self.data["config"] = read_config(data_path, topology)
        self.data["technology_data"] = read_technology_data(data_path, topology)
        self.data["network_data"] = read_network_data(data_path, topology)

        self.data["time_series_data"] = {}
        self.data["time_series_data"]["full_resolution"] = time_series

        log_msg = "--- Reading in data complete ---"
        print(log_msg)
        log.info(log_msg)

    def process_data(self):
        """
        Processes the data read in to the ModelHub instance.

        This includes fitting technology performance to climate data
        and processing network data.
        """
        log_msg = "--- Processing data ---"
        print(log_msg)
        log.info(log_msg)

        start_time = time.time()

        for period, nodes in self.component_constructors["technology_constructors"].items():
            for node, technologies in nodes.items():
                for technology, constructor in technologies.items():
                    constructor.fit_performance(self, (period, node, technology))

        for period, networks in self.component_constructors["network_constructors"].items():
            for network, constructor in networks.items():
                constructor.fit_performance(self, (period, network))

        # Todo: process data in plugins

        log_msg = f"--- Data processing complete in {time.time() - start_time:.2f} seconds ---"
        print(log_msg)
        log.info(log_msg)

    def aggregate_data(self):
        """
        Aggregates the data read in to the ModelHub instance based on the chosen aggregation algorithm
        """
        log_msg = "--- Aggregating data ---"
        print(log_msg)
        log.info(log_msg)

        # Standard is full resolution data
        self.data["aggregation_info"] = {}
        self.data["aggregation_info"]["time_series_used"] = "full_resolution"
        self.data["aggregation_info"]["time_indexes"] = {}
        for period in self.data["topology"]["investment_periods"]:
            self.data["aggregation_info"]["time_indexes"]["set_t_full"] = {}
            self.data["aggregation_info"]["time_indexes"]["set_t_clustered"] = {}
            self.data["aggregation_info"]["time_indexes"]["set_t_full"][period] = range(1, len(self.data["topology"]["temporal_information"]["time_index"])+1)
            self.data["aggregation_info"]["time_indexes"]["set_t_clustered"][period] = range(1, len(self.data["topology"]["temporal_information"]["time_index"]))
        self.data["aggregation_info"]["hour_factors"] = [1] * len(self.data["time_series_data"]["full_resolution"])
        self.data["aggregation_info"]["nr_timesteps_averaged"] = 1
        self.info_solving_algorithms["aggregation_model"] = "full_resolution"
        self.info_solving_algorithms["aggregation_data"] = "full_resolution"

    def _perform_preprocessing_checks(self):
        """
        Checks consistency of input data, before constructing or solving the model

        - Save path must exist
        - dynamics checks
        :return:
        """
        config = self.data["config"]

        # Is solver available?
        try:
            mock_model = pyo.ConcreteModel()
            if config["solveroptions"]["solver"]["value"] == "gurobi":
                solver = get_gurobi_parameters(config["solveroptions"])
            elif config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
                solver = get_gurobi_parameters(config["solveroptions"])
                self.solver.set_instance(mock_model)
            elif config["solveroptions"]["solver"]["value"] == "glpk":
                solver = get_glpk_parameters(config["solveroptions"])
            solver.solve(mock_model)
        except:
            raise Exception(
                "The solver you are trying to use is not available. This "
                "could be due to the following two reasons: (1) it is not "
                "implemented (currently AdOpT-NET0 supports gurobi and "
                "glpk. (2) If you are using gurobi, make sure that the "
                "gurobipy version installed matches the gurobi version "
                "installed."
            )

        # Check if save-path exists
        save_path = Path(config["reporting"]["save_path"]["value"])
        if not os.path.exists(save_path) or not os.path.isdir(save_path):
            raise FileNotFoundError(
                f"The folder you want to save your results to ('{save_path}') does not "
                f"exist. Create the folder or change the folder "
                f"name in the ModelConfig"
            )

        # Todo: move to plugins
        # # Dynamics and time aggregation algorithms
        # if config["optimization"]["typicaldays"]["N"]["value"] != 0:
        #     if config["performance"]["dynamics"]["value"]:
        #         raise Exception(
        #             "Dynamics and clustering with typical days is not " "allowed"
        #         )
        #     for period in topology["investment_periods"]:
        #         for node in topology["nodes"]:
        #             for tec_name in self.data.technology_data[period][node]:
        #                 tec = self.data.technology_data[period][node][tec_name]
        #                 if ("ramping_const_int" in tec.processed_coeff.dynamics) and (
        #                     tec.processed_coeff.dynamics["ramping_const_int"] != -1
        #                 ):
        #                     raise Exception(
        #                         f"Ramping constraint with integers (ramping_const_int) for technology {tec_name} "
        #                         f"needs to be -1 when clustering with typical days"
        #                     )
        #
        # if config["optimization"]["timestaging"]["value"] != 0:
        #     if config["performance"]["dynamics"]["value"]:
        #         raise Exception(
        #             "Dynamics and two-stage averaging algorithm is not " "allowed"
        #         )
        #     for period in topology["investment_periods"]:
        #         for node in topology["nodes"]:
        #             for tec_name in self.data.technology_data[period][node]:
        #                 tec = self.data.technology_data[period][node][tec_name]
        #                 if ("ramping_time" in tec.processed_coeff.dynamics) and (
        #                     tec.processed_coeff.dynamics["ramping_time"] != -1
        #                 ):
        #                     raise Exception(
        #                         f"Ramping Rate for technology {tec_name} "
        #                         f"needs to be -1 when two-stage averaging algorithm is used"
        #                     )
        #
        # # check if technologies have dynamic parameters
        # if config["performance"]["dynamics"]["value"]:
        #     for period in topology["investment_periods"]:
        #         for node in topology["nodes"]:
        #             for tec in self.data.technology_data[period][node]:
        #                 if self.data.technology_data[period][node][
        #                     tec
        #                 ].technology_model in [
        #                     "CONV1",
        #                     "CONV2",
        #                     "CONV3",
        #                 ]:
        #                     par_check = [
        #                         "max_startups",
        #                         "min_uptime",
        #                         "min_downtime",
        #                         "SU_load",
        #                         "SD_load",
        #                         "SU_time",
        #                         "SD_time",
        #                     ]
        #                     for par in par_check:
        #                         if (
        #                             par
        #                             not in self.data.technology_data[period][node][
        #                                 tec
        #                             ].processed_coeff.dynamics
        #                         ):
        #                             raise ValueError(
        #                                 f"The technology '{tec}' does not have dynamic parameter '{par}'. Add the parameters in the "
        #                                 f"json files or switch off the dynamics."
        #                             )

    def construct_components(self):
        """
        Constructs the components. The model structure is as follows:

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
        self._perform_preprocessing_checks()

        # Determine aggregation
        config = self.data["config"]

        # Clustered data
        self.info_solving_algorithms["aggregation_model"] = "full_resolution"
        self.info_solving_algorithms["aggregation_data"] = "full_resolution"

        if config["optimization"]["typicaldays"]["N"]["value"] != 0:
            if config["optimization"]["typicaldays"]["method"]["value"] == 1:
                self.info_solving_algorithms["aggregation_model"] = "clustered"
                self.info_solving_algorithms["aggregation_data"] = "clustered"
            elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
                self.info_solving_algorithms["aggregation_model"] = "clustered"
                self.info_solving_algorithms["aggregation_data"] = "full_resolution"
            else:
                raise Exception("clustering method needs to be 1 or 2")

        # Averaged data
        if config["optimization"]["timestaging"]["value"] != 0:
            if self.info_solving_algorithms["time_stage"] == 1:
                self.info_solving_algorithms["aggregation_model"] = "averaged"
                self.info_solving_algorithms["aggregation_data"] = "averaged"

        # INITIALIZE MODEL
        aggregation_model = self.info_solving_algorithms["aggregation_model"]
        self.model[aggregation_model] = pyo.ConcreteModel()

        construct_components(self.model[aggregation_model], self)

    def construct_balances(self):
        """
        Constructs the energy balance, emission balance and calculates costs
        """
        model = self.model[self.info_solving_algorithms["aggregation_model"]]
        construct_balances(model, self)

    def solve(self):
        """
        Defines objective and solves model
        """
        config = self.data["config"]

        objective = config["optimization"]["objective"]["value"]

        self._define_solver_settings()

        if objective == "pareto":
            self._solve_pareto()
        else:
            self._optimize(objective)

    def quick_solve(self):
        """
        Quick-solves the model (constructs model and balances and solves model).

        This method lumbs together the following functions for convenience:
        - :func:`~adopt_net0.modelhub.construct_model`
        - :func:`~adopt_net0.modelhub.construct_balances`
        - :func:`~adopt_net0.modelhub.solve`
        """
        self.process_data()
        self.aggregate_data()
        self.construct_components()
        self.construct_balances()
        self.solve()

    def write_results(self):
        """
        Writes optimization results of a model run to folder
        """
        # Write H5 File

        solution_available = True
        if self.solution.solver.termination_condition in [
            pyo.TerminationCondition.infeasibleOrUnbounded,
            pyo.TerminationCondition.infeasible,
            pyo.TerminationCondition.unbounded,
        ]:
            solution_available = False

        if solution_available:

            config = self.data["config"]

            save_summary_path = Path.joinpath(
                Path(config["reporting"]["save_summary_path"]["value"]), "Summary.xlsx"
            )

            model_info = self.last_solve_info

            model = self.model[self.info_solving_algorithms["aggregation_model"]]

            summary_dict = write_optimization_results_to_h5(
                model, self.solution, model_info, self
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

    # Todo: move to plugin
    # def add_technology(self, investment_period: str, node: str, technologies: list):
    #     """
    #     Adds technologies retrospectively to the model.
    #
    #     After adding a technology to a node, all balances need to be re-constructed,
    #     To solve the model again run, :func:`~construct_balances` and then
    #     :func:`~solve`.
    #
    #     :param str investment_period: name of investment period for which technology is added
    #     :param str node: name of node for which technology is added
    #     :param list technologies: list of technologies that should be added
    #     :return None:
    #     """
    #     model = self.model[self.info_solving_algorithms["aggregation_model"]]
    #
    #     # Make sure that no aggregation algorithm is used
    #     config = self.data["config"]
    #     if (config["optimization"]["typicaldays"]["N"]["value"] != 0) or (
    #         config["optimization"]["timestaging"]["value"] != 0
    #     ):
    #         raise Exception(
    #             "You cannot add a technolgy retrospectively if using time aggragation algorithms"
    #         )
    #
    #     # Read technology data
    #     data_node = get_data_for_node(
    #         get_data_for_investment_period(
    #             self.data,
    #             investment_period,
    #             self.info_solving_algorithms["aggregation_model"],
    #         ),
    #         node,
    #     )
    #
    #     for technology in technologies:
    #         # read in technology data
    #         tec_data = create_technology_class(
    #             technology,
    #             self.data.data_path
    #             / investment_period
    #             / "node_data"
    #             / node
    #             / "technology_data",
    #         )
    #         # fit technology data
    #         tec_data.fit_performance(
    #             self.data.time_series["full_resolution"][investment_period][node]["TechnologyTimeSeries"][
    #                 "global"
    #             ],
    #             self.data.node_locations.loc[node, :],
    #         )
    #         # add technology data to data handle
    #         self.data.technology_data[investment_period][node][technology] = tec_data
    #         data_node["technology_data"][technology] = tec_data
    #
    #     # Add technology to node
    #     b_period = model.periods[investment_period]
    #     b_node = b_period.node_blocks[node]
    #
    #     # Create new technology block containing all new technologies
    #     def init_technology_block(b_tec, tec):
    #         b_tec = construct_technology_block(
    #             b_tec, data_node, b_period.set_t_full, b_period.set_t_clustered
    #         )
    #
    #         return b_tec
    #
    #     b_node.tech_blocks_new = pyo.Block(technologies, rule=init_technology_block)
    #
    #     # If it exists, carry over active tech blocks to temporary block
    #     if b_node.find_component("tech_blocks_active"):
    #         b_node.tech_blocks_existing = pyo.Block(b_node.set_technologies)
    #         for tec in b_node.set_technologies:
    #             b_node.tech_blocks_existing[tec].transfer_attributes_from(
    #                 b_node.tech_blocks_active[tec]
    #             )
    #         b_node.del_component(b_node.tech_blocks_active)
    #     if b_node.find_component("tech_blocks_active_index"):
    #         b_node.del_component(b_node.tech_blocks_active_index)
    #
    #     # Create a block containing all active technologies at node
    #     if not set(technologies).issubset(b_node.set_technologies):
    #         b_node.set_technologies.add(technologies)
    #
    #     def init_active_technology_blocks(bl, tec):
    #         if tec in technologies:
    #             bl.transfer_attributes_from(b_node.tech_blocks_new[tec])
    #         else:
    #             bl.transfer_attributes_from(b_node.tech_blocks_existing[tec])
    #
    #     b_node.tech_blocks_active = pyo.Block(
    #         b_node.set_technologies, rule=init_active_technology_blocks
    #     )
    #
    #     # Delete all auxiliary blocks
    #     if b_node.find_component("tech_blocks_new"):
    #         b_node.del_component(b_node.tech_blocks_new)
    #     if b_node.find_component("tech_blocks_new_index"):
    #         b_node.del_component(b_node.tech_blocks_new_index)
    #     if b_node.find_component("tech_blocks_existing"):
    #         b_node.del_component(b_node.tech_blocks_existing)
    #     if b_node.find_component("tech_blocks_existing_index"):
    #         b_node.del_component(b_node.tech_blocks_existing_index)

    def _define_solver_settings(self):
        """
        Defines solver and its settings depending on objective and solver
        """
        config = self.data["config"]
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
        config = self.data["config"]

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
        log_msg = "Set objective on cost"
        print(log_msg)
        log.info(log_msg)
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
        log_msg = "Set objective on net emissions"
        print(log_msg)
        log.info(log_msg)
        self._call_solver()

    def _optimize_costs_emissionslimit(self):
        """
        Minimize costs at emission limit
        """
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

        config = self.data["config"]

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
        log_msg = "Defined constraint on net emissions"
        print(log_msg)
        log.info(log_msg)
        self._optimize_cost()

    def _optimize_costs_minE(self):
        """
        Minimize costs at minimum emissions
        """
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

        config = self.data["config"]

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
        config = self.data["config"]

        f_global = config["scaling"]["scaling_factors"]
        model_full = self.model[self.info_solving_algorithms["aggregation_model"]]

        model_full.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        # Scale technologies
        for period in model_full.periods:
            b_period = model_full.periods[period]
            # Scale technologies
            for node in b_period.node_blocks:
                for tec in b_period.node_blocks[node].tech_blocks_active:
                    b_tec = b_period.node_blocks[node].tech_blocks_active[tec]
                    model_full = self.data.technology_data[period][node][
                        tec
                    ].scale_model(b_tec, model_full, config)

            # Scale networks
            for netw in b_period.network_block:
                b_netw = b_period.network_block[netw]
                model_full = self.data.network_data[period][netw].scale_model(
                    b_netw, model_full, config
                )

            # Scale period
            if f_global["energy_vars"]["value"] >= 0:
                # Network constraints
                model_full.scaling_factor[
                    model_full.block_network_constraints[period].const_netw_inflow
                ] = f_global["energy_vars"]["value"]
                model_full.scaling_factor[
                    model_full.block_network_constraints[period].const_netw_outflow
                ] = f_global["energy_vars"]["value"]
                model_full.scaling_factor[
                    model_full.block_network_constraints[period].const_netw_consumption
                ] = f_global["energy_vars"]["value"]

                # Energy balance
                model_full.scaling_factor[
                    model_full.block_energybalance[period].const_energybalance
                ] = f_global["energy_vars"]["value"]

            # Costs balance
            model_full.scaling_factor[
                model_full.block_costbalance[period].const_capex_tecs
            ] = (f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"])
            model_full.scaling_factor[
                model_full.block_costbalance[period].const_capex_netw
            ] = (f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"])
            model_full.scaling_factor[
                model_full.block_costbalance[period].const_opex_tecs
            ] = (f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"])
            model_full.scaling_factor[
                model_full.block_costbalance[period].const_opex_netw
            ] = (f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"])
            model_full.scaling_factor[
                model_full.block_costbalance[period].const_cost_tecs
            ] = (f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"])
            model_full.scaling_factor[
                model_full.block_costbalance[period].const_cost_netws
            ] = (f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"])
            model_full.scaling_factor[
                model_full.block_costbalance[period].const_cost_import
            ] = (f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"])
            model_full.scaling_factor[
                model_full.block_costbalance[period].const_cost_export
            ] = (f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"])
            model_full.scaling_factor[
                model_full.block_costbalance[period].const_violation_cost
            ] = (f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"])
            model_full.scaling_factor[
                model_full.block_costbalance[period].const_revenue_carbon
            ] = (f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"])
            model_full.scaling_factor[
                model_full.block_costbalance[period].const_cost_carbon
            ] = (f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"])
            model_full.scaling_factor[
                model_full.block_costbalance[period].const_cost
            ] = (f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"])

            # Period Variables
            model_full.scaling_factor[b_period.var_cost_capex_tecs] = (
                f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
            )
            model_full.scaling_factor[b_period.var_cost_capex_netws] = (
                f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
            )
            model_full.scaling_factor[b_period.var_cost_opex_tecs] = (
                f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
            )
            model_full.scaling_factor[b_period.var_cost_opex_netws] = (
                f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
            )
            model_full.scaling_factor[b_period.var_cost_tecs] = (
                f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
            )
            model_full.scaling_factor[b_period.var_cost_netws] = (
                f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
            )
            model_full.scaling_factor[b_period.var_cost_imports] = (
                f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
            )
            model_full.scaling_factor[b_period.var_cost_exports] = (
                f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
            )
            model_full.scaling_factor[b_period.var_cost_violation] = (
                f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
            )
            model_full.scaling_factor[b_period.var_carbon_revenue] = (
                f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
            )
            model_full.scaling_factor[b_period.var_carbon_cost] = (
                f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
            )
            model_full.scaling_factor[b_period.var_cost_total] = (
                f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
            )

            for node in b_period.node_blocks:
                b_node = b_period.node_blocks[node]
                model_full.scaling_factor[b_node.var_import_flow] = f_global[
                    "energy_vars"
                ]["value"]
                model_full.scaling_factor[b_node.var_export_flow] = f_global[
                    "energy_vars"
                ]["value"]
                model_full.scaling_factor[b_node.var_netw_inflow] = f_global[
                    "energy_vars"
                ]["value"]
                model_full.scaling_factor[b_node.var_netw_outflow] = f_global[
                    "energy_vars"
                ]["value"]
                model_full.scaling_factor[b_node.var_netw_consumption] = f_global[
                    "energy_vars"
                ]["value"]
                model_full.scaling_factor[b_node.var_generic_production] = f_global[
                    "energy_vars"
                ]["value"]
                model_full.scaling_factor[b_node.const_generic_production] = f_global[
                    "energy_vars"
                ]["value"]
                model_full.scaling_factor[b_node.var_import_flow] = f_global[
                    "energy_vars"
                ]["value"]

        # Global cost balance
        model_full.scaling_factor[model_full.const_npv] = (
            f_global["cost_vars"]["value"] * f_global["energy_vars"]["value"]
        )
        # Scale objective
        model_full.scaling_factor[model_full.objective] = (
            f_global["objective"]["value"] * f_global["cost_vars"]["value"]
        )

        self.model["scaled"] = pyo.TransformationFactory(
            "core.scale_model"
        ).create_using(model_full)

    def _call_solver(self):
        """
        Calls the solver and solves the model
        """
        log.info("Solving Model...")

        start = time.time()
        config = self.data["config"]

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

        # Determine if results should be written
        if "write_results" in config["reporting"].keys():
            if config["reporting"]["write_results"]["value"] == 1:
                write_results = True
            else:
                write_results = False
        else:
            warnings.warn(
                "The config file needs to contain config['reporting']"
                "['write_results']. This is mandatory in future versions",
                FutureWarning,
                stacklevel=2,
            )
            write_results = True

        # Check if solution is available
        if write_results:
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
            if config["scaling"]["scaling_on"]["value"] == 1:
                pyo.TransformationFactory("core.scale_model").propagate_solution(
                    model, self.model[self.info_solving_algorithms["aggregation_model"]]
                )

        if config["reporting"]["write_solution_diagnostics"]["value"] >= 1:
            self._write_solution_diagnostics(result_folder_path)

        self.solution.write()

        self.last_solve_info["pareto_point"] = self.info_pareto["pareto_point"]

        self.last_solve_info["config"] = config
        self.last_solve_info["result_folder_path"] = result_folder_path
        self.last_solve_info["time_stage"] = self.info_solving_algorithms["time_stage"]
        self.last_solve_info["aggregation_model"] = self.info_solving_algorithms[
            "aggregation_model"
        ]

        # Write results to path
        if write_results:
            self.write_results()

        log.info("Solving model completed in " + str(round(time.time() - start)) + " s")

    def _write_solution_diagnostics(self, save_path):
        """
        Can write solution quality, constraint map and variable map to file. Options
        are specified in the configuration.

        :param save_path:
        :return:
        """
        config = self.data["config"]
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
        config = self.data["config"]
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
            log_msg = f"Optimizing Pareto point {limit}"
            print(log_msg)
            log.info(log_msg)
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

    def _delete_objective(self):
        """
        Delete the objective function
        """
        config = self.data["config"]
        model = self.model[self.info_solving_algorithms["aggregation_model"]]

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
        self.construct_components()
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

        m_full = self.model["full_resolution"]
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
                        self.data.technology_data[period][node][tec].technology_model
                        == "STOR"
                        and bounds_on == "no_storage"
                    ):
                        return pyo.Constraint.Skip
                    elif self.data.technology_data[period][node][tec].existing:
                        return pyo.Constraint.Skip
                    else:
                        log_msg = (
                            f"Size constraint imposed on {tec} at {node} in {period}"
                        )
                        print(log_msg)
                        log.info(log_msg)

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

                    log_msg = f"Size constraint imposed on {netw} in {period}"
                    print(log_msg)
                    log.info(log_msg)

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
