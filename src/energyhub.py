from pyomo.environ import *
from pyomo.environ import units as u

import src.model_construction as mc
import src.data_management as dm
from src.utilities import *
import pint
import numpy as np
import dill as pickle
import src.global_variables as global_variables
import time
import copy


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
    def __init__(self, data, configuration):
        """
        Constructor of the energyhub class.
        """
        print('_' * 20)
        print('Reading in data...')
        start = time.time()

        # READ IN MODEL CONFIGURATION
        self.configuration = configuration

        # INITIALIZE MODEL
        self.model = ConcreteModel()

        # Define units
        try:
            u.load_definitions_from_strings(['EUR = [currency]'])
        except pint.errors.DefinitionSyntaxError:
            pass

        # INITIALIZE SOLUTION
        self.solution = []

        # READ IN DATA
        if not self.configuration.optimization.typicaldays == 0:
            print('Clustering Data...')
            self.data = dm.ClusteredDataHandle(data, self.configuration.optimization.typicaldays)
            global_variables.clustered_data = 1
            global_variables.clustered_data_specs.specs = self.data.k_means_specs
            print('Clustering Data completed')
        else:
            global_variables.clustered_data = 0
            self.data = data

        if self.configuration.optimization.timestaging:
            print('Averaging Data...')
            self.data_full_res = self.data
            self.data = dm.DataHandle_AveragedData(self.data_full_res, self.configuration.optimization.timestaging)
            global_variables.averaged_data = 1
            global_variables.averaged_data_specs.specs = self.data.averaged_specs
            self.model_first_stage = []
            self.solution_first_stage = []
            print('Averaging Data completed')


        print('Reading in data completed in ' + str(time.time() - start) + ' s')
        print('_' * 20)

    def quick_solve_model(self):
        """
        Quick-solves the model (constructs model and balances and solves model).

        This method lumbs together the following functions for convenience:
        - :func:`~src.energyhub.construct_model`
        - :func:`~src.energyhub.construct_balances`
        - :func:`~src.energyhub.solve_model`
        """
        self.construct_model()
        self.construct_balances()
        self.solve_model()

    def construct_model(self):
        """
        Constructs model equations, defines objective functions and calculates emissions.

        This function constructs the initial model with all its components as specified in the
        topology. It adds (1) networks (:func:`~add_networks`), and (2) nodes and technologies
        (:func:`~src.model_construction.construct_nodes.add_nodes` including \
        :func:`~add_technologies`)
        """
        print('_' * 20)
        print('Constructing Model...')
        start = time.time()

        # DEFINE SETS
        topology = self.data.topology
        self.model.set_nodes = Set(initialize=topology.nodes)
        self.model.set_carriers = Set(initialize=topology.carriers)
        self.model.set_t = RangeSet(1,len(topology.timesteps))

        def tec_node(set, node):
            if self.data.technology_data:
                return self.data.technology_data[node].keys()
            else:
                return Set.Skip
        self.model.set_technologies = Set(self.model.set_nodes, initialize=tec_node)
        self.model.set_networks = Set(initialize=self.data.network_data.keys())

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
        self.model = mc.add_networks(self)
        self.model = mc.add_nodes(self)

        print('Constructing model completed in ' + str(time.time() - start) + ' s')
        print('_' * 20)

    def construct_balances(self):
        """
        Constructs the energy balance, emission balance and calculates costs

        Links all components with the constructing the energybalance (:func:`~add_energybalance`),
        the total cost (:func:`~add_system_costs`) and the emission balance (:func:`~add_emissionbalance`)
        """
        print('_' * 20)
        print('Constructing balances...')
        start = time.time()

        self.model = mc.add_energybalance(self)

        self.model = mc.add_emissionbalance(self)
        self.model = mc.add_system_costs(self)

        print('Constructing balances completed in ' + str(time.time() - start) + ' s')
        print('_' * 20)

    def solve_model(self):
        """
        Defines objective and solves model

        The objective is minimized and can be chosen as total annualized costs ('costs'), total annual net emissions
        ('emissions_net'), total positive emissions ('emissions_pos') and annual emissions at minimal cost
        ('emissions_minC'). This needs to be set in the configuration file respectively.
        """
        # This is a dirty fix as objectives cannot be found with find_component
        try:
            self.model.del_component(self.model.objective)
        except:
            pass

        objective = self.configuration.optimization.objective

        # Define Objective Function
        if objective == 'costs':
            def init_cost_objective(obj):
                return self.model.var_total_cost
            self.model.objective = Objective(rule=init_cost_objective, sense=minimize)
            self.__optimize()
        elif objective == 'emissions_pos':
            def init_emission_pos_objective(obj):
                return self.model.var_emissions_pos
            self.model.objective = Objective(rule=init_emission_pos_objective, sense=minimize)
            self.__optimize()
        elif objective == 'emissions_net':
            def init_emission_net_objective(obj):
                return self.model.var_emissions_net
            self.model.objective = Objective(rule=init_emission_net_objective, sense=minimize)
            self.__optimize()
        elif objective == 'emissions_minC':
            def init_emission_minC_objective(obj):
                return self.model.var_emissions_pos
            self.model.objective = Objective(rule=init_emission_minC_objective, sense=minimize)
            self.__optimize()
            emission_limit = self.model.var_emissions_pos.value
            self.model.const_emission_limit = Constraint(expr=self.model.var_emissions_pos <= emission_limit)
            self.model.del_component(self.model.objective)
            def init_cost_objective(obj):
                return self.model.var_total_cost
            self.model.objective = Objective(rule=init_cost_objective, sense=minimize)
            self.__optimize()
        elif objective == 'pareto':
            print('to be implemented')

        if self.configuration.optimization.timestaging and not global_variables.averaged_data_specs.last_stage:
            global_variables.averaged_data = 0
            global_variables.averaged_data_specs.last_stage = 1
            bounds_on = 'all'
            self.model_first_stage = self.model
            self.solution_first_stage = copy.deepcopy(self.solution)
            self.model = ConcreteModel()
            self.solution = []
            self.data = self.data_full_res
            self.construct_model()
            self.construct_balances()
            self.__impose_size_constraints(bounds_on)
            self.solve_model()


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
        mc.add_technologies(self, nodename, technologies)

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

    def calculate_occurance_per_hour(self):
        """
        Calculates how many times an hour in the reduced resolution occurs in the full resolution
        :return np array occurance_hour:
        """
        if global_variables.clustered_data and global_variables.averaged_data:
            occurrence_hour = np.multiply(
                self.data.k_means_specs.reduced_resolution['factor'].to_numpy(),
                self.data.averaged_specs.reduced_resolution['factor'].to_numpy())
        elif global_variables.clustered_data and not global_variables.averaged_data:
            occurrence_hour = self.data.k_means_specs.reduced_resolution['factor'].to_numpy()
        elif not global_variables.clustered_data and global_variables.averaged_data:
            occurrence_hour = self.data.averaged_specs.reduced_resolution['factor'].to_numpy()
        else:
            occurrence_hour = np.ones(len(self.model.set_t))
        return occurrence_hour

    def __optimize(self):
        """
        Solves the model
        :return:
        """

        # Define solver settings
        if self.configuration.solveroptions.solver == 'gurobi':
            solver = get_gurobi_parameters(self.configuration.solveroptions)

        # Solve model
        print('_' * 20)
        print('Solving Model...')

        start = time.time()
        self.solution = solver.solve(self.model, tee=True, warmstart=True)
        self.solution.write()

        print('Solving model completed in ' + str(time.time() - start) + ' s')
        print('_' * 20)

    def __impose_size_constraints(self, bounds_on):
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
                           b_netw_avg.arc_block[node_to, node_from].var_size.value
                block.size_constraints_netw = Constraint(b_netw_full.set_arcs_unique, rule=size_constraints_netw_init)
            m_full.size_constraints_netw = Block(m_full.set_networks, rule=size_constraint_block_netw_init)


def load_energyhub_instance(file_path):
    """
    Loads an energyhub instance from file.

    :param str file_path: path to previously saved energyhub instance
    :return: energyhub instance
    """
    with open(file_path, mode='rb') as file:
        energyhub = pickle.load(file)
    return energyhub