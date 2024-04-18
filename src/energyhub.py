from pyomo.environ import *
import numpy as np
import pandas as pd
import dill as pickle
import time
import copy
import warnings
import datetime
from pathlib import Path
import os
import sys

from .model_construction import *
from .data_management import *
from .utilities import *
from .components.utilities import annualize, set_discount_rate
from .components.technologies.utilities import set_capex_model
from .result_management import *

class EnergyHub:
    r"""
    Class to construct and manipulate an energy system model.

    When constructing an instance, it reads data to the instance and initializes all attributes of the EnergyHub
    class:
    - self.configuration: Contains options for the optimization and is passed to the constructor
    - self.model: A concrete Pyomo model
    -

    **Set declarations:**

    - Set of nodes :math:`N`
    - Set of carriers :math:`M`
    - Set of time steps :math:`T`
    - Set of weather variables :math:`W`
    - Set of technologies at each node :math:`S_n, n \in N`

    """
    def __init__(self, data, configuration):
        """
        Constructor of the energyhub class.
        """
        print('_' * 60)
        print('Reading in data...')
        start = time.time()

        # READ IN MODEL CONFIGURATION
        self.configuration = configuration

        # INITIALIZE MODEL
        self.model = ConcreteModel()
        self.scaled_model = []

        # INITIALIZE GLOBAL OPTIONS
        self.topology = data.topology
        self.model_information = data.model_information
        self.model_information.clustered_data = 0
        self.model_information.averaged_data = 0

        # INITIALIZE SOLUTION
        self.solution = None

        # INITIALIZE SOLVER
        self.solver = None

        # INITIALIZE DATA
        self.data_storage = []
        if not self.configuration.optimization.typicaldays.N == 0:
            # If clustered
            self.model_information.clustered_data = 1
            self.data_storage.append(ClusteredDataHandle(data, self.configuration.optimization.typicaldays.N))
        else:
            self.data_storage.append(data)

        if self.configuration.optimization.timestaging:
            # Average data
            self.model_information.averaged_data = 1
            self.model_first_stage = None
            self.solution_first_stage = None
            self.data_storage.append(DataHandle_AveragedData(self.data_storage[0], self.configuration.optimization.timestaging))
            self.data = self.data_storage[1]
        else:
            # Write data to self
            self.data = self.data_storage[0]

        print('Reading in data completed in ' + str(round(time.time() - start)) + ' s')
        print('_' * 60)

        self._perform_preprocessing_checks()


    def _perform_preprocessing_checks(self):
        """
        Checks some things, before constructing or solving the model
        :return:
        """
        # Check if save-path exists
        save_path = Path(self.configuration.reporting.save_path)
        if not os.path.exists(save_path) or not \
                os.path.isdir(save_path):
            raise FileNotFoundError(f"The folder '{save_path}' does not exist. Create the folder or change the folder "
                                    f"name in the configuration")

        # check if technologies have dynamic parameters
        if self.configuration.performance.dynamics:
            for node in self.data.topology.nodes:
                for tec in self.data.technology_data[node]:
                    if self.data.technology_data[node][tec].technology_model in ['CONV1', 'CONV2', 'CONV3']:
                        par_check = ['max_startups', 'min_uptime', 'min_downtime', 'SU_load', 'SD_load', 'SU_time', 'SD_time']
                        count = 0
                        for par in par_check:
                            if par not in self.data.technology_data[node][tec].performance_data:
                                raise ValueError(
                                    f"The technology '{tec}' does not have dynamic parameter '{par}'. Add the parameters in the "
                                    f"json files or switch off the dynamics.")

        # check if time horizon is not longer than 1 year (in case of single year analysis)
        if self.configuration.optimization.multiyear == 0:
            nr_timesteps_data = len(self.data.topology.timesteps)
            nr_timesteps_year = 8760
            if nr_timesteps_data > nr_timesteps_year:
                raise ValueError(
                    f"Time horizon is longer than one year. Enable multiyear analysis if you want to optimize for"
                    f"a longer time horizon.")



    def quick_solve(self):
        """
        Quick-solves the model (constructs model and balances and solves model).

        This method lumbs together the following functions for convenience:
        - :func:`~src.energyhub.construct_model`
        - :func:`~src.energyhub.construct_balances`
        - :func:`~src.energyhub.solve_model`
        """
        self.construct_model()
        self.construct_balances()

        self.solve()

    def construct_model(self):
        """
        Constructs model equations, defines objective functions and calculates emissions.

        This function constructs the initial model with all its components as specified in the
        topology. It adds (1) networks (:func:`~add_networks`), and (2) nodes and technologies
        (:func:`~src.model_construction.construct_nodes.add_nodes` including \
        :func:`~add_technologies`)
        """
        print('_' * 60)
        print('Constructing Model...')
        start = time.time()

        # DEFINE SETS
        # Nodes, Carriers, Technologies, Networks
        topology = self.data.topology
        self.model.set_nodes = Set(initialize=topology.nodes)
        self.model.set_carriers = Set(initialize=topology.carriers)

        def tec_node(set, node):
            if self.data.technology_data:
                return self.data.technology_data[node].keys()
            else:
                return Set.Skip

        self.model.set_technologies = Set(self.model.set_nodes, initialize=tec_node)
        self.model.set_networks = Set(initialize=list(self.data.network_data.keys()))

        # Time Frame
        self.model.set_t_full = RangeSet(1,len(self.data.topology.timesteps))

        if self.model_information.clustered_data == 1:
            self.model.set_t_clustered = RangeSet(1,len(self.data.topology.timesteps_clustered))

        # DEFINE VARIABLES
        # Global cost variables
        self.model.var_node_cost = Var()
        self.model.var_netw_cost = Var()
        self.model.var_total_cost = Var()
        self.model.var_carbon_revenue = Var()
        self.model.var_carbon_cost = Var()

        # Global Emission variables
        self.model.var_emissions_pos = Var()
        self.model.var_emissions_neg = Var()
        self.model.var_emissions_net = Var()

        # Parameters
        def init_carbon_subsidy(para, t):
            return self.data.global_data.data['carbon_prices']['subsidy'][t - 1]

        self.model.para_carbon_subsidy = Param(self.model.set_t_full, rule=init_carbon_subsidy, mutable=True)

        def init_carbon_tax(para, t):
            return self.data.global_data.data['carbon_prices']['tax'][t - 1]

        self.model.para_carbon_tax = Param(self.model.set_t_full, rule=init_carbon_tax, mutable=True)

        # Model construction
        if not self.configuration.energybalance.copperplate:
            self.model = add_networks(self)
        self.model = add_nodes(self)

        print('Constructing model completed in ' + str(round(time.time() - start)) + ' s')

    def construct_balances(self):
        """
        Constructs the energy balance, emission balance and calculates costs

        Links all components with the constructing the energybalance (:func:`~add_energybalance`),
        the total cost (:func:`~add_system_costs`) and the emission balance (:func:`~add_emissionbalance`)
        """
        print('_' * 60)
        print('Constructing balances...')
        start = time.time()

        self.model = add_energybalance(self)
        self.model = add_emissionbalance(self)
        self.model = add_system_costs(self)

        print('Constructing balances completed in ' + str(round(time.time() - start)) + ' s')

    def solve(self):
        """
        Defines objective and solves model

        The objective is minimized and can be chosen as total annualized costs ('costs'), total annual net emissions
        ('emissions_net'), total positive emissions ('emissions_pos') and annual emissions at minimal cost
        ('emissions_minC'). This needs to be set in the configuration file respectively.
        """
        objective = self.configuration.optimization.objective

        self._define_solver_settings()

        if self.configuration.optimization.monte_carlo.on:
            self._solve_monte_carlo(objective)
        elif objective == 'pareto':
            self._solve_pareto()
        else:
            self._optimize(objective)


    def add_technology_to_node(self, nodename, technologies):
        """
        Adds technologies retrospectively to the model.

        After adding a technology to a node, the energy and emission balance need to be re-constructed, as well as the
        costs recalculated. To solve the model, :func:`~construct_balances` and then solve again.

        :param str nodename: name of node for which technology is installed
        :param list technologies: list of technologies that should be added to nodename
        :return: None
        """
        self.data.read_single_technology_data(nodename, technologies)
        add_technology(self, nodename, technologies)

    def save_model(self, save_path, file_name):
        """
        Saves an instance of the energyhub instance to the specified path (using pickel/dill).

        The object can later be loaded using into the work space using :func:`~load_energyhub_instance`

        :param str file_path: path to save
        :param str file_name: filename
        :return: None
        """
        with open(Path(save_path) / file_name, mode='wb') as file:
            pickle.dump(self, file)

    def _define_solver_settings(self):
        """
        Defines solver and its settings depending on objective and solver
        """
        objective = self.configuration.optimization.objective

        # Set solver
        if self.configuration.solveroptions.solver in ['gurobi', 'gurobi_persistent']:
            # Gurobi
            if not self.configuration.scaling:
                if objective in ['emissions_minC', 'pareto'] or self.configuration.optimization.monte_carlo.on:
                    self.configuration.solveroptions.solver = 'gurobi_persistent'
            self.solver = get_gurobi_parameters(self.configuration.solveroptions)

        elif self.configuration.solveroptions.solver == 'glpk':
            self.solver =  get_glpk_parameters(self.configuration.solveroptions)

        # For persistent solver, set model instance
        if self.configuration.solveroptions.solver == 'gurobi_persistent':
            self.solver.set_instance(self.model)

    def _optimize(self, objective):
        """
        Solves the model with the given objective
        """
        # Define Objective Function
        if objective == 'costs':
            self._optimize_cost()
        elif objective == 'emissions_pos':
            self._optimize_emissions_pos()
        elif objective == 'emissions_net':
            self._optimize_emissions_net()
        elif objective == 'emissions_minC':
            self._optimize_costs_minE()
        elif objective == 'costs_emissionlimit':
            self._optimize_costs_emissionslimit()
        else:
            raise Exception("objective in Configurations is incorrect")

        # Second stage of time averaging algorithm
        if self.model_information.averaged_data and self.model_information.averaged_data_specs.stage == 0:
            self._optimize_time_averaging_second_stage()


    def _optimize_cost(self):
        """
        Minimizes Costs
        """
        self._delete_objective()

        def init_cost_objective(obj):
            return self.model.var_total_cost
        self.model.objective = Objective(rule=init_cost_objective, sense=minimize)
        self._call_solver()

    def _optimize_emissions_pos(self):
        """
        Minimizes positive emission
        """
        self._delete_objective()

        def init_emission_pos_objective(obj):
            return self.model.var_emissions_pos
        self.model.objective = Objective(rule=init_emission_pos_objective, sense=minimize)
        self._call_solver()

    def _optimize_emissions_net(self):
        """
        Minimize net emissions
        """
        self._delete_objective()

        def init_emission_net_objective(obj):
            return self.model.var_emissions_net
        self.model.objective = Objective(rule=init_emission_net_objective, sense=minimize)
        self._call_solver()


    def _optimize_costs_emissionslimit(self):
        """
        Minimize costs at emission limit
        """
        emission_limit = self.configuration.optimization.emission_limit
        if self.model.find_component('const_emission_limit'):
            if self.configuration.solveroptions.solver == 'gurobi_persistent':
                self.solver.remove_constraint(self.model.const_emission_limit)
            self.model.del_component(self.model.const_emission_limit)
        self.model.const_emission_limit = Constraint(expr=self.model.var_emissions_net <= emission_limit*1.001)
        if self.configuration.solveroptions.solver == 'gurobi_persistent':
            self.solver.add_constraint(self.model.const_emission_limit)
        self._optimize_cost()


    def _optimize_costs_minE(self):
        """
        Minimize costs at minimum emissions
        """
        self._optimize_emissions_net()
        emission_limit = self.model.var_emissions_net.value
        if self.model.find_component('const_emission_limit'):
            if self.configuration.solveroptions.solver == 'gurobi_persistent':
                self.solver.remove_constraint(self.model.const_emission_limit)
            self.model.del_component(self.model.const_emission_limit)
        self.model.const_emission_limit = Constraint(expr=self.model.var_emissions_net <= emission_limit*1.001)
        if self.configuration.solveroptions.solver == 'gurobi_persistent':
            self.solver.add_constraint(self.model.const_emission_limit)
        self._optimize_cost()

    def _solve_pareto(self):
        """
        Optimize the pareto front
        """
        pareto_points = self.configuration.optimization.pareto_points

        # Min Cost
        self.model_information.pareto_point = 0
        self._optimize_cost()
        emissions_max = self.model.var_emissions_net.value

        # Min Emissions
        self.model_information.pareto_point = pareto_points + 1
        self._optimize_costs_minE()
        emissions_min = self.model.var_emissions_net.value

        # Emission limit
        self.model_information.pareto_point = 0
        emission_limits = np.linspace(emissions_min, emissions_max, num=pareto_points)
        for pareto_point in range(0, pareto_points):
            self.model_information.pareto_point += 1
            if self.configuration.solveroptions.solver == 'gurobi_persistent':
                self.solver.remove_constraint(self.model.const_emission_limit)
            self.model.del_component(self.model.const_emission_limit)
            self.model.const_emission_limit = Constraint(
                expr=self.model.var_emissions_net <= emission_limits[pareto_point]*1.005)
            if self.configuration.solveroptions.solver == 'gurobi_persistent':
                self.solver.add_constraint(self.model.const_emission_limit)
            self._optimize_cost()

    def _solve_monte_carlo(self, objective):
        """
        Optimizes multiple runs with monte carlo
        """
        for run in range(0, self.configuration.optimization.monte_carlo.N):
            self.model_information.monte_carlo_run += 1
            self._monte_carlo_set_cost_parameters()
            if run == 0:
                self._optimize(objective)
            else:
                self._call_solver()

    def scale_model(self):
        """
        Creates a scaled model in self.scaled_model using the scale factors specified in the json files for technologies
        and networks as well as the global scaling factors specified. See also the documentation on model scaling.
        """

        f_global = self.configuration.scaling_factors
        self.model.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Scale technologies
        for node in self.model.node_blocks:
            for tec in self.model.node_blocks[node].tech_blocks_active:
                b_tec = self.model.node_blocks[node].tech_blocks_active[tec]
                self.model = self.data.technology_data[node][tec].scale_model(b_tec, self.model, self.configuration)

        # Scale networks
        for netw in self.model.network_block:
            b_netw = self.model.network_block[netw]
            self.model = self.data.network_data[netw].scale_model(b_netw, self.model, self.configuration)

        # Scale objective
        self.model.scaling_factor[self.model.objective] = f_global.objective *  f_global.cost_vars

        # Scale globals
        if f_global.energy_vars >= 0:
            self.model.scaling_factor[self.model.const_energybalance] = f_global.energy_vars
            self.model.scaling_factor[self.model.const_cost] = f_global.cost_vars * f_global.energy_vars
            self.model.scaling_factor[self.model.const_node_cost] = f_global.cost_vars * f_global.energy_vars
            self.model.scaling_factor[self.model.const_netw_cost] = f_global.cost_vars * f_global.energy_vars
            self.model.scaling_factor[self.model.const_revenue_carbon] = f_global.cost_vars * f_global.energy_vars
            self.model.scaling_factor[self.model.const_cost_carbon] = f_global.cost_vars * f_global.energy_vars

            self.model.scaling_factor[self.model.var_node_cost] = f_global.cost_vars * f_global.energy_vars
            self.model.scaling_factor[self.model.var_netw_cost] = f_global.cost_vars * f_global.energy_vars
            self.model.scaling_factor[self.model.var_total_cost] = f_global.cost_vars * f_global.energy_vars
            self.model.scaling_factor[self.model.var_carbon_revenue] = f_global.cost_vars * f_global.energy_vars
            self.model.scaling_factor[self.model.var_carbon_cost] = f_global.cost_vars * f_global.energy_vars

            for node in self.model.node_blocks:
                self.model.scaling_factor[self.model.node_blocks[node].var_import_flow] = f_global.energy_vars
                self.model.scaling_factor[self.model.node_blocks[node].var_export_flow] = f_global.energy_vars

                self.model.scaling_factor[self.model.node_blocks[node].var_netw_inflow] = f_global.energy_vars
                self.model.scaling_factor[self.model.node_blocks[node].const_netw_inflow] = f_global.energy_vars

                self.model.scaling_factor[self.model.node_blocks[node].var_netw_outflow] = f_global.energy_vars
                self.model.scaling_factor[self.model.node_blocks[node].const_netw_outflow] = f_global.energy_vars

                self.model.scaling_factor[self.model.node_blocks[node].var_generic_production] = f_global.energy_vars
                self.model.scaling_factor[self.model.node_blocks[node].const_generic_production] = f_global.energy_vars


        self.scaled_model = TransformationFactory('core.scale_model').create_using(self.model)
        # self.scaled_model.pprint()


    def _call_solver(self):
        """
        Calls the solver and solves the model
        """
        print('_' * 60)
        print('Solving Model...')

        start = time.time()

        # Create save path and folder
        time_stamp = datetime.datetime.fromtimestamp(start).strftime('%Y%m%d%H%M%S')
        save_path = Path(self.configuration.reporting.save_path)

        if self.configuration.reporting.case_name == -1:
            folder_name = str(time_stamp)
        else:
            folder_name = str(time_stamp) + '_' + self.configuration.reporting.case_name

        result_folder_path = create_unique_folder_name(save_path, folder_name)
        create_save_folder(result_folder_path)
        save_summary_path = Path.joinpath(Path(self.configuration.reporting.save_summary_path), 'Summary.xlsx')

        # Scale model
        if self.configuration.scaling == 1:
            self.scale_model()
            model = self.scaled_model
        else:
            model = self.model

        # Call solver
        if self.configuration.solveroptions.solver == 'gurobi_persistent':
            self.solver.set_objective(model.objective)

        if self.configuration.solveroptions.solver == 'glpk':
            self.solution = self.solver.solve(model,
                                              tee=True, logfile=str(Path(result_folder_path / 'log.txt')),
                                              keepfiles=True)
        else:
            self.solution = self.solver.solve(model,
                                              tee=True,
                                              warmstart=True,
                                              logfile=str(Path(result_folder_path / 'log.txt')),
                                              keepfiles=True)

        if self.configuration.scaling == 1:
            TransformationFactory('core.scale_model').propagate_solution(self.scaled_model, self.model)

        if self.configuration.reporting.write_solution_diagnostics >= 1:
            self._write_solution_diagnostics(result_folder_path)

        self.solution.write()

        # Write H5 File
        if self.solution.solver.termination_condition == 'optimal':

            summary_dict = write_optimization_results_to_h5(self, result_folder_path)

            # Write Summary
            if not os.path.exists(save_summary_path):
                summary_df = pd.DataFrame(data=summary_dict, index=[0])
                summary_df.to_excel(save_summary_path, index=False, sheet_name="Summary")
            else:
                summary_existing = pd.read_excel(save_summary_path)
                pd.concat([summary_existing, pd.DataFrame(data=summary_dict, index=[0])]).to_excel(save_summary_path, index=False, sheet_name="Summary")


        print('Solving model completed in ' + str(round(time.time() - start)) + ' s')
        print('_' * 60)

    def _write_solution_diagnostics(self, save_path):

        model = self.solver._solver_model
        constraint_map = self.solver._pyomo_con_to_solver_con_map
        variable_map = self.solver._pyomo_var_to_solver_var_map

        # Write solution quality to txt
        with open(f'{save_path}/diag_solution_quality.txt', 'w') as file:
            sys.stdout = file  # Redirect stdout to the file
            model.printQuality()  # Call the function that prints something
            sys.stdout = sys.__stdout__  # Reset stdout to the console

        if self.configuration.reporting.write_solution_diagnostics >=2:
            # Write constraint map to txt
            with open(f'{save_path}/diag_constraint_map.txt', 'w') as file:
                for key, value in constraint_map.items():
                    file.write(f'{key}: {value}\n')

            # Write var map to txt
            with open(f'{save_path}/diag_variable_map.txt', 'w') as file:
                for key, value in variable_map._dict.items():
                    file.write(f'{value[0].name}: {value[1]}\n')

    def _monte_carlo_set_cost_parameters(self):
        """
        Performs monte carlo analysis
        """

        if 'Technologies' in self.configuration.optimization.monte_carlo.on_what:
            for node in self.model.node_blocks:
                for tec in self.model.node_blocks[node].tech_blocks_active:
                    self._monte_carlo_technologies(node, tec)

        if 'Networks' in self.configuration.optimization.monte_carlo.on_what:
            for netw in self.model.network_block:
                self._monte_carlo_networks(netw)

        if 'ImportPrices' in self.configuration.optimization.monte_carlo.on_what:
            for node in self.model.node_blocks:
                for car in self.model.node_blocks[node].set_carriers:
                    self._monte_carlo_import_prices(node, car)

        if 'ExportPrices' in self.configuration.optimization.monte_carlo.on_what:
            for node in self.model.node_blocks:
                for car in self.model.node_blocks[node].set_carriers:
                    self._monte_carlo_export_prices(node, car)


    def _monte_carlo_technologies(self, node, tec):
        """
        Changes the capex of technologies
        """
        sd = self.configuration.optimization.monte_carlo.sd
        sd_random = np.random.normal(1, sd)

        tec_data = self.data.technology_data[node][tec]
        economics = tec_data.economics
        discount_rate = set_discount_rate(self.configuration, economics)
        fraction_of_year_modelled = self.topology.fraction_of_year_modelled

        capex_model = set_capex_model(self.configuration, economics)
        annualization_factor = annualize(discount_rate, economics.lifetime, fraction_of_year_modelled)

        b_tec = self.model.node_blocks[node].tech_blocks_active[tec]

        if capex_model == 1:
            # UNIT CAPEX
            # Update parameter
            unit_capex = tec_data.economics.capex_data['unit_capex'] * sd_random
            self.model.node_blocks[node].tech_blocks_active[tec].para_unit_capex = unit_capex
            self.model.node_blocks[node].tech_blocks_active[tec].para_unit_capex_annual = unit_capex * annualization_factor

            # Remove constraint (from persistent solver and from model)
            self.solver.remove_constraint(b_tec.const_capex_aux)
            b_tec.del_component(b_tec.const_capex_aux)

            # Add constraint again
            b_tec.const_capex_aux = Constraint(
                expr=b_tec.var_size * b_tec.para_unit_capex_annual == b_tec.var_capex_aux)
            self.solver.add_constraint(b_tec.const_capex_aux)

        elif capex_model == 2:
            warnings.warn("monte carlo on piecewise defined investment costs is not implemented")


    def _monte_carlo_networks(self, netw):
        """
        Changes the capex of networks
        """
        # TODO: This does not work!

        sd = self.configuration.optimization.monte_carlo.sd
        sd_random = np.random.normal(1, sd)

        netw_data = self.data.network_data[netw]
        economics = netw_data.economics
        discount_rate = set_discount_rate(self.configuration, economics)
        fraction_of_year_modelled = self.topology.fraction_of_year_modelled

        capex_model = economics.capex_model
        annualization_factor = annualize(discount_rate, economics.lifetime, fraction_of_year_modelled)

        b_netw =self.model.network_block[netw]

        if capex_model == 1:
            b_netw.para_capex_gamma1 = economics.capex_data['gamma1'] * annualization_factor * sd_random
            b_netw.para_capex_gamma2 = economics.capex_data['gamma2'] * annualization_factor * sd_random

        elif capex_model == 2:
            b_netw.para_capex_gamma1 = economics.capex_data['gamma1'] * annualization_factor * sd_random
            b_netw.para_capex_gamma2 = economics.capex_data['gamma2'] * annualization_factor * sd_random

        elif capex_model == 3:
            b_netw.para_capex_gamma1 = economics.capex_data['gamma1'] * annualization_factor * sd_random
            b_netw.para_capex_gamma2 = economics.capex_data['gamma2'] * annualization_factor * sd_random
            b_netw.para_capex_gamma3 = economics.capex_data['gamma3'] * annualization_factor * sd_random

        for arc in b_netw.set_arcs:
            b_arc = b_netw.arc_block[arc]

            # Remove constraint (from persistent solver and from model)
            self.solver.remove_constraint(b_arc.const_capex_aux)
            b_arc.del_component(b_arc.const_capex_aux)

            # Add constraint again
            def init_capex(const):
                if economics.capex_model == 1:
                    return b_arc.var_capex_aux == b_arc.var_size * \
                           b_netw.para_capex_gamma1 + b_netw.para_capex_gamma2
                elif economics.capex_model == 2:
                    return b_arc.var_capex_aux == b_arc.var_size * \
                           b_arc.distance * b_netw.para_capex_gamma1 + b_netw.para_capex_gamma2
                elif economics.capex_model == 3:
                    return b_arc.var_capex_aux == b_arc.var_size * \
                           b_arc.distance * b_netw.para_capex_gamma1 + \
                           b_arc.var_size * b_netw.para_capex_gamma2 + \
                           b_netw.para_capex_gamma3
            b_arc.const_capex_aux = Constraint(rule=init_capex)
            self.solver.add_constraint(b_arc.const_capex_aux)

    def _monte_carlo_import_prices(self, node, car):
        """
        Changes the import prices
        """
        sd = self.configuration.optimization.monte_carlo.sd
        sd_random = np.random.normal(1, sd)

        model = self.model
        set_t = model.set_t_full

        import_prices = self.data.node_data[node].data['import_prices'][car]
        b_node = self.model.node_blocks[node]

        for t in set_t:
            # Update parameter
            b_node.para_import_price[t, car] = import_prices[t-1] * sd_random

            # Remove constraint (from persistent solver and from model)
            self.solver.remove_constraint(model.const_node_cost)
            self.model.del_component(model.const_node_cost)

            # Add constraint again
            nr_timesteps_averaged = self.model_information.averaged_data_specs.nr_timesteps_averaged

            def init_node_cost(const):
                tec_capex = sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_capex
                                    for tec in model.node_blocks[node].set_tecsAtNode)
                                for node in model.set_nodes)
                tec_opex_variable = sum(sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_opex_variable[t] *
                                                nr_timesteps_averaged
                                                for tec in model.node_blocks[node].set_tecsAtNode)
                                            for t in set_t)
                                        for node in model.set_nodes)
                tec_opex_fixed = sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_opex_fixed
                                         for tec in model.node_blocks[node].set_tecsAtNode)
                                     for node in model.set_nodes)
                import_cost = sum(sum(sum(model.node_blocks[node].var_import_flow[t, car] *
                                          model.node_blocks[node].para_import_price[t, car] *
                                          nr_timesteps_averaged
                                          for car in model.node_blocks[node].set_carriers)
                                      for t in set_t)
                                  for node in model.set_nodes)
                export_revenue = sum(sum(sum(model.node_blocks[node].var_export_flow[t, car] *
                                             model.node_blocks[node].para_export_price[t, car] *
                                             nr_timesteps_averaged
                                             for car in model.node_blocks[node].set_carriers)
                                         for t in set_t)
                                     for node in model.set_nodes)
                return tec_capex + tec_opex_variable + tec_opex_fixed + import_cost - export_revenue == model.var_node_cost

            model.const_node_cost = Constraint(rule=init_node_cost)
            self.solver.add_constraint(model.const_node_cost)


    def _monte_carlo_export_prices(self, node, car):
        """
        Changes the export prices
        """
        sd = self.configuration.optimization.monte_carlo.sd
        sd_random = np.random.normal(1, sd)

        model = self.model
        set_t = model.set_t_full

        export_prices = self.data.node_data[node].data['export_prices'][car]
        b_node = self.model.node_blocks[node]

        for t in set_t:
            # Update parameter
            b_node.para_export_price[t, car] = export_prices[t - 1] * sd_random

            # Remove constraint (from persistent solver and from model)
            self.solver.remove_constraint(model.const_node_cost)
            self.model.del_component(model.const_node_cost)

            # Add constraint again
            nr_timesteps_averaged = self.model_information.averaged_data_specs.nr_timesteps_averaged

            def init_node_cost(const):
                tec_capex = sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_capex
                                    for tec in model.node_blocks[node].set_tecsAtNode)
                                for node in model.set_nodes)
                tec_opex_variable = sum(sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_opex_variable[t] *
                                                nr_timesteps_averaged
                                                for tec in model.node_blocks[node].set_tecsAtNode)
                                            for t in set_t)
                                        for node in model.set_nodes)
                tec_opex_fixed = sum(sum(model.node_blocks[node].tech_blocks_active[tec].var_opex_fixed
                                         for tec in model.node_blocks[node].set_tecsAtNode)
                                     for node in model.set_nodes)
                import_cost = sum(sum(sum(model.node_blocks[node].var_import_flow[t, car] *
                                          model.node_blocks[node].para_import_price[t, car] *
                                          nr_timesteps_averaged
                                          for car in model.node_blocks[node].set_carriers)
                                      for t in set_t)
                                  for node in model.set_nodes)
                export_revenue = sum(sum(sum(model.node_blocks[node].var_export_flow[t, car] *
                                             model.node_blocks[node].para_export_price[t, car] *
                                             nr_timesteps_averaged
                                             for car in model.node_blocks[node].set_carriers)
                                         for t in set_t)
                                     for node in model.set_nodes)
                return tec_capex + tec_opex_variable + tec_opex_fixed + import_cost - export_revenue == model.var_node_cost

            model.const_node_cost = Constraint(rule=init_node_cost)
            self.solver.add_constraint(model.const_node_cost)

    def _delete_objective(self):
        """
        Delete the objective function
        """
        if not self.configuration.optimization.monte_carlo.on:
            try:
                self.model.del_component(self.model.objective)
            except:
                pass

    def _optimize_time_averaging_second_stage(self):
        """
        Optimizes the second stage of the time_averaging algorithm
        """
        self.model_information.averaged_data_specs.stage += 1
        self.model_information.averaged_data_specs.nr_timesteps_averaged = 1
        bounds_on = 'no_storage'
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

        m_full = self.model
        m_avg = self.model_first_stage

        # Technologies
        if bounds_on == 'all' or bounds_on == 'only_technologies' or bounds_on == 'no_storage':
            def size_constraint_block_tecs_init(block, node):
                def size_constraints_tecs_init(const, tec):
                    if self.data.technology_data[node][tec].technology_model == 'STOR' and bounds_on == 'no_storage':
                        return Constraint.Skip
                    elif self.data.technology_data[node][tec].existing:
                        return  Constraint.Skip
                    else:
                        return m_avg.node_blocks[node].tech_blocks_active[tec].var_size.value <= \
                            m_full.node_blocks[node].tech_blocks_active[tec].var_size
                block.size_constraints_tecs = Constraint(m_full.set_technologies[node], rule=size_constraints_tecs_init)
            m_full.size_constraint_tecs = Block(m_full.set_nodes, rule=size_constraint_block_tecs_init)

        # Networks
        if bounds_on == 'all' or bounds_on == 'only_networks' or bounds_on == 'no_storage':
            def size_constraint_block_netw_init(block, netw):
                b_netw_full = m_full.network_block[netw]
                b_netw_avg = m_avg.network_block[netw]
                def size_constraints_netw_init(const, node_from, node_to):
                    return b_netw_full.arc_block[node_from, node_to].var_size >= \
                           b_netw_avg.arc_block[node_from, node_to].var_size.value
                block.size_constraints_netw = Constraint(b_netw_full.set_arcs, rule=size_constraints_netw_init)
            m_full.size_constraints_netw = Block(m_full.set_networks, rule=size_constraint_block_netw_init)


def load_energyhub_instance(load_path):
    """
    Loads an energyhub instance from file.

    :param str file_path: path to previously saved energyhub instance
    :return: energyhub instance
    """
    with open(Path(load_path), mode='rb') as file:
        energyhub = pickle.load(file)
    return energyhub