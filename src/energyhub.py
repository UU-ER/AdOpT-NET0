from pyomo.environ import *
from pyomo.environ import units as u
from pyomo.gdp import *

import src.model_construction as mc
import src.data_management as dm
import pint
import numpy as np
import dill as pickle
import pandas as pd
import src.config_model as m_config
import time


class EnergyHub:
    r"""
    Class to construct and manipulate an energy system model.

    When constructing an instance, it reads data to the instance and defines relevant model sets:

    **Set declarations:**

    - Set of nodes :math:`N`
    - Set of carriers :math:`M`
    - Set of time steps :math:`T`
    - Set of weather variables :math:`W`
    - Set of technologies at each node :math:`S_n, n \in N`

    """
    def __init__(self, data):
        """
        Constructor of the energyhub class.
        """
        print('Reading in data...')
        start = time.time()

        # READ IN DATA
        self.data = data

        # INITIALIZE MODEL
        self.model = ConcreteModel()

        # Define units
        try:
            u.load_definitions_from_strings(['EUR = [currency]'])
        except pint.errors.DefinitionSyntaxError:
            pass

        # Initialize solution
        self.solution = []

        print('Reading in data completed in ' + str(time.time() - start) + ' s')

    def construct_model(self):
        """
        Constructs model equations, defines objective functions and calculates emissions.

        This function constructs the initial model with all its components as specified in the
        topology. It adds (1) networks (:func:`~add_networks`), and (2) nodes and technologies
        (:func:`~src.model_construction.construct_nodes.add_nodes` including \
        :func:`~add_technologies`)
        """
        print('Constructing Model...')
        start = time.time()

        # DEFINE SETS
        sets = self.data.topology
        self.model.set_nodes = Set(initialize=sets['nodes'])
        self.model.set_carriers = Set(initialize=sets['carriers'])
        self.model.set_t = RangeSet(1,len(sets['timesteps']))
        if hasattr(self.data, 'k_means_specs'):
            # If yes, we are working with clustered data
            self.model.set_t_full = RangeSet(1, len(self.data.k_means_specs['keys']['typical_day']))
            m_config.presolve.clustered_data = 1
        else:
            self.model.set_t_full = RangeSet(1,len(sets['timesteps']))
            m_config.presolve.clustered_data = 0

        def tec_node(set, node):
            if node in self.model.set_nodes:
                try:
                    if sets['technologies']:
                        return sets['technologies'][node]
                    else:
                        return Set.Skip
                except (KeyError, ValueError):
                    raise Exception('The nodes in the technology sets do not match the node names. The node \'', node,
                          '\' does not exist.')
        self.model.set_technologies = Set(self.model.set_nodes, initialize=tec_node)
        self.model.set_networks = Set(initialize=sets['networks'].keys())

        # DEFINE VARIABLES
        # Global cost variables
        self.model.var_node_cost = Var()
        self.model.var_netw_cost = Var()
        self.model.var_total_cost = Var()
        # Global Emission variables
        self.model.var_emissions_pos = Var()
        self.model.var_emissions_neg = Var()
        self.model.var_emissions_net = Var()

        # Model construction
        self.model = mc.add_networks(self.model, self.data)
        self.model = mc.add_nodes(self.model, self.data)

        print('Constructing model completed in ' + str(time.time() - start) + ' s')

    def construct_balances(self):
        """
        Constructs the energy balance, emission balance and calculates costs

        Links all components with the constructing the energybalance (:func:`~add_energybalance`),
        the total cost (:func:`~add_system_costs`) and the emission balance (:func:`~add_emissionbalance`)
        """
        print('Constructing balances...')
        start = time.time()

        self.model = mc.add_energybalance(self.model)

        if m_config.presolve.clustered_data == 1:
            occurrence_hour = self.data.k_means_specs['factors']['factor'].to_numpy()
        else:
            occurrence_hour = np.ones(len(self.model.set_t))

        self.model = mc.add_emissionbalance(self.model, occurrence_hour)
        self.model = mc.add_system_costs(self.model, occurrence_hour)

        print('Constructing balances completed in ' + str(time.time() - start) + ' s')

    def solve_model(self, objective = 'cost'):
        """
        Defines objective and solves model

        The objective is minimized and can be chosen as total annualized costs, total annualized emissions \
        multi-objective (emission-cost pareto front).
        """
        # This is a dirty fix as objectives cannot be found with find_component
        try:
            self.model.del_component(self.model.objective)
        except:
            pass

        # Define Objective Function
        if objective == 'cost':
            def init_cost_objective(obj):
                return self.model.var_total_cost
            self.model.objective = Objective(rule=init_cost_objective, sense=minimize)
        elif objective == 'emissions_pos':
            def init_emission_pos_objective(obj):
                return self.model.var_emissions_pos
            self.model.objective = Objective(rule=init_emission_pos_objective, sense=minimize)
        elif objective == 'emissions_net':
            def init_emission_net_objective(obj):
                return self.model.var_emissions_net
            self.model.objective = Objective(rule=init_emission_net_objective, sense=minimize)
        elif objective == 'emissions_minC':
            def init_emission_minC_objective(obj):
                return self.model.var_emissions_pos
            self.model.objective = Objective(rule=init_emission_minC_objective, sense=minimize)
            emission_limit = self.model.var_emissions_pos.value
            self.model.const_emission_limit = Constraint(expr=self.model.var_emissions_pos <= emission_limit)
            def init_cost_objective(obj):
                return self.model.var_total_cost
            self.model.objective = Objective(rule=init_cost_objective, sense=minimize)
        elif objective == 'pareto':
            print('to be implemented')

        # Solve model
        print('Solving Model...')
        start = time.time()
        solver = SolverFactory(m_config.solver.solver)
        self.solution = solver.solve(self.model, tee=True, warmstart=True)
        self.solution.write()

        print('Solving model completed in ' + str(time.time() - start) + ' s')

    def add_technology_to_node(self, nodename, technologies):
        """
        Adds technologies retrospectively to the model.

        After adding a technology to a node, the anergy and emission balance need to be re-constructed, as well as the
        costs recalculated. To solve the model, :func:`~construct_balances` and then solve again.

        :param str nodename: name of node for which technology is installed
        :param list technologies: list of technologies that should be added to nodename
        :return: None
        """
        self.data.read_single_technology_data(nodename, technologies)
        node_block = self.model.node_blocks[nodename]
        mc.add_technologies(nodename, technologies, self.model, self.data, node_block)

    def save_model(self, file_path, file_name):
        """
        Saves an instance of the energyhub instance to the specified path (using pickel/dill).

        The object can later be loaded using into the work space using :func:`~load_energyhub_instance`

        :param file_path: path to save
        :param file_name: filename
        :return: None
        """
        with open(file_path + '/' + file_name, mode='wb') as file:
            pickle.dump(self, file)

    def print_topology(self):
        print('----- SET OF CARRIERS -----')
        for car in self.model.set_carriers:
            print('- ' + car)
        print('----- NODE DATA -----')
        for node in self.model.set_nodes:
            print('\t -----------------------------------------------------')
            print('\t nodename: '+ node)
            print('\t\ttechnologies installed:')
            for tec in self.model.set_technologies[node]:
                print('\t\t - ' + tec)
            print('\t\taverage demand:')
            for car in self.model.set_carriers:
                avg = round(self.data.demand[node][car].mean(), 2)
                print('\t\t - ' + car + ': ' + str(avg))
            print('\t\taverage of climate data:')
            for ser in self.data.climate_data[node]['dataframe']:
                avg = round(self.data.climate_data[node]['dataframe'][ser].mean(),2)
                print('\t\t - ' + ser + ': ' + str(avg))
        print('----- NETWORK DATA -----')
        for car in self.data.topology['networks']:
            print('\t -----------------------------------------------------')
            print('\t carrier: '+ car)
            for netw in self.data.topology['networks'][car]:
                print('\t\t - ' + netw)
                connection = self.data.topology['networks'][car][netw]['connection']
                for from_node in connection:
                    for to_node in connection[from_node].index:
                        if connection.at[from_node, to_node] == 1:
                            print('\t\t\t' + from_node  + '---' +  to_node)

    def write_results(self):
        """
        Exports results to an instance of ResultsHandle to be further exported or viewed
        """
        results = dm.ResultsHandle()
        results.read_results(self)
        return results

def load_energyhub_instance(file_path):
    """
    Loads an energyhub instance from file.

    :param str file_path: path to previously saved energyhub instance
    :return: energyhub instance
    """

    with open(file_path, mode='rb') as file:
        energyhub = pickle.load(file)
    return energyhub