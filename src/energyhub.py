from pyomo.environ import *
from pyomo.environ import units as u

import src.model_construction as mc
import src.data_management as dm
from src.utilities import *
import pint
import numpy as np
import pandas as pd
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

        # READ IN DATA
        self.data = data

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

        # ENABLE TYPICAL DAYS AND/OR TIMESTAGING AND COMPUTE NEW DATASET
        if self.configuration.optimization.typicaldays:
            self.data = ClusteredDataHandle(data, self.configuration.optimization.typicaldays)
        if self.configuration.optimization.timestaging:
            nr_timesteps_averaged = self.configuration.optimization.timestaging
            self.configuration.optimization.timestaging = 0
            self.full_res_ehub = EnergyHub(data, configuration)
            data_averaged = DataHandle_AveragedData(data, nr_timesteps_averaged)
            EnergyHub.__init__(self, data_averaged, configuration)
            global_variables.averaged_data_specs.nr_timesteps_averaged = nr_timesteps_averaged

        # SET GLOBAL VARIABLES
        global_variables.clustered_data = 0
        global_variables.averaged_data = 0
        if hasattr(self.data, 'k_means_specs'):
            # Clustered Data
            global_variables.clustered_data = 1
            global_variables.clustered_data_specs.specs = self.data.k_means_specs
        if hasattr(self, 'full_res_ehub') and hasattr(self.data, 'averaged_specs'):
            # Averaged Data
            global_variables.averaged_data = 1
            global_variables.averaged_data_specs.specs = self.data.averaged_specs

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

        The objective is minimized and can be chosen as total annualized costs ('costs'), total annual emissions
        ('emissions_net'), and total annual emissions at minimal cost ('emissions_minC').
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

class EnergyHubTwoStageTimeAverage(EnergyHub):
    """
    Sub-class of the EnergyHub class to perform two stage time averaging optimization based on
    Weimann, L., & Gazzani, M. (2022). A novel time discretization method for solving complex multi-energy system
    design and operation problems with high penetration of renewable energy. Computers & Chemical Engineering,
    107816. https://doi.org/10.1016/J.COMPCHEMENG.2022.107816

    All methods available in the super-class EnergyHub are also available in this subclass.
    """
    def __init__(self, data, configuration, nr_timesteps_averaged=4):
        self.full_res_ehub = EnergyHub(data, configuration)
        data_averaged = DataHandle_AveragedData(data, nr_timesteps_averaged)
        EnergyHub.__init__(self, data_averaged, configuration)
        global_variables.averaged_data_specs.nr_timesteps_averaged = nr_timesteps_averaged

    def solve_model(self, objective = 'cost', bounds_on = 'all'):
        """
        Solve the model in a two stage time average manner.

        In the first stage, time-steps are averaged and thus the model is optimized with a reduced time frame.
        In the second stage, the sizes of technologies and networks are constrained with a lower bound. It is possible
        to only constrain some technologies with the parameter bounds_on (see below)

        :param objective: objective to minimize
        :param bounds_on: can be 'all', 'only_technologies', 'only_networks', 'no_storage'
        """
        # Solve reduced resolution model
        self.construct_model()
        self.construct_balances()
        super().solve_model()
        global_variables.averaged_data = 0
        global_variables.averaged_data_specs.nr_timesteps_averaged = 1


        # Solve full resolution model
        # Initialize
        self.full_res_ehub.construct_model()
        self.full_res_ehub.construct_balances()
        # Impose additional constraints
        self.impose_size_constraints(bounds_on)
        # Solve with additional constraints
        self.full_res_ehub.solve_model()

    def write_results(self):
        """
        Overwrites method of EnergyHub superclass
        """
        results = dm.ResultsHandle()
        results.read_results(self.full_res_ehub)
        return results

    def impose_size_constraints(self, bounds_on):
        """
        Formulates lower bound on technology and network sizes.

        It is possible to exclude storage technologies or networks by specifying bounds_on. Not this function is called
        from the method solve_model.

        :param bounds_on: can be 'all', 'only_technologies', 'only_networks', 'no_storage'
        """

        m_full = self.full_res_ehub.model
        m_avg = self.model

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

class ClusteredDataHandle(EnergyHub):
    """
    Performs the clustering process

    This function performs the k-means algorithm on the data resulting in a new DataHandle object that can be
    passed to the energhub class for optimization.

    :param DataHandle data_in: DataHandle containing data of the full resolution
    :param int nr_clusters: nr of clusters (tyical days) the data contains after the algorithm
    :param int nr_time_intervals_per_day: nr of time intervals per day in data (full resolution)
    """
    def __init__(self, data_in, nr_clusters, nr_time_intervals_per_day=24):
        """
        Constructor

        :param DataHandle data_in: DataHandle containing data of the full resolution
        :param int nr_clusters: nr of clusters (typical days) the data contains after the algorithm
        :param int nr_time_intervals_per_day: nr of time intervals per day in data (full resolution)
        """
        data = copy.deepcopy(data_in)

        # Copy over data from old object
        self.node_data = {}
        self.node_data_full_resolution = data.node_data
        self.technology_data = data.technology_data
        self.network_data = data.network_data
        self.topology = data.topology

        # k-means specs
        self.k_means_specs = dm.simplification_specs(data.topology.timesteps)

        # flag tecs that contain time-dependent data
        self.tecs_flagged_for_clustering = dm.flag_tecs_for_clustering(self)

        # perform clustering
        nr_days_full_resolution = (max(data.topology.timesteps) -  min(data.topology.timesteps)).days + 1
        self.cluster_data(nr_clusters, nr_days_full_resolution, nr_time_intervals_per_day)

    def cluster_data(self, nr_clusters, nr_days_full_resolution, nr_time_intervals_per_day):
        """
        Performs the clustering process

        This function performs the k-means algorithm on the data resulting in a new DataHandle object that can be passed
        to the energhub class for optimization.

        :param DataHandle data: DataHandle containing data of the full resolution
        :param int nr_clusters: nr of clusters (tyical days) the data contains after the algorithm
        :param int nr_days_full_resolution: nr of days in data (full resolution)
        :param int nr_time_intervals_per_day: nr of time intervalls per day in data (full resolution)
        :return: instance of :class:`~ClusteredDataHandle`
        """
        # adjust timesteps
        self.topology.timesteps = range(0, nr_clusters * nr_time_intervals_per_day)
        # flag tecs that contain time-dependent data
        tecs_flagged_for_clustering = self.tecs_flagged_for_clustering
        # compile full matrix to cluster
        full_resolution = self.compile_full_resolution_matrix(nr_time_intervals_per_day,
                                                              tecs_flagged_for_clustering)
        # Perform clustering
        clustered_data, day_labels = dm.perform_k_means(full_resolution,
                                                        nr_clusters)
        # Get order of typical days
        self.k_means_specs.full_resolution['hourly_order'] = dm.compile_hourly_order(day_labels,
                                         nr_clusters,
                                         nr_days_full_resolution,
                                         nr_time_intervals_per_day)
        # Match typical day to actual day
        self.k_means_specs.full_resolution['typical_day'] = np.repeat(day_labels, nr_time_intervals_per_day)
        # Create factors, indicating how many times an hour occurs
        self.k_means_specs.reduced_resolution = dm.get_day_factors(self.k_means_specs.full_resolution['hourly_order'])
        # Read data back in
        self.read_clustered_data(clustered_data, tecs_flagged_for_clustering)

    def read_clustered_data(self, clustered_data, tecs_flagged_for_clustering):
        """
        Reads clustered data back to self

        :param clustered_data: Clustered data
        :param tecs_flagged_for_clustering: technologies that have time-dependent data
        """
        node_data = self.node_data_full_resolution
        for node in node_data:
            self.node_data[node] = {}
            for series in node_data[node]:
                if not (series == 'climate_data') and not (series == 'production_profile_curtailment'):
                    self.node_data[node][series] = pd.DataFrame()
                    for carrier in node_data[node][series]:
                        self.node_data[node][series][carrier] = \
                            dm.reshape_df(clustered_data[node][series][carrier],
                                       None, 1)
            for tec in tecs_flagged_for_clustering[node]:
                series_data = dm.reshape_df(clustered_data[node][tec][tecs_flagged_for_clustering[node][tec]], None, 1)
                series_data = series_data.to_numpy()
                self.technology_data[node][tec].fitted_performance[tecs_flagged_for_clustering[node][tec]] = \
                    series_data
            self.node_data[node]['production_profile_curtailment'] = node_data[node]['production_profile_curtailment']


    def compile_full_resolution_matrix(self, nr_time_intervals_per_day, tecs_flagged_for_clustering):
        """
        Compiles full resolution matrix to be clustered

        Contains, prices, emission factors, capacity factors,...
        """
        full_resolution = pd.DataFrame()
        node_data = self.node_data_full_resolution
        for node in node_data:
            for series in node_data[node]:
                if not (series == 'climate_data') and not (series == 'production_profile_curtailment'):
                    for carrier in node_data[node][series]:
                        series_names = dm.define_multiindex([
                            [node] * nr_time_intervals_per_day,
                            [series] * nr_time_intervals_per_day,
                            [carrier] * nr_time_intervals_per_day,
                            list(range(1, nr_time_intervals_per_day + 1))
                        ])
                        to_add = dm.reshape_df(node_data[node][series][carrier],
                                            series_names, nr_time_intervals_per_day)
                        full_resolution = pd.concat([full_resolution, to_add], axis=1)
            for tec in tecs_flagged_for_clustering[node]:
                series_names = dm.define_multiindex([
                    [node] * nr_time_intervals_per_day,
                    [tec] * nr_time_intervals_per_day,
                    [tecs_flagged_for_clustering[node][tec]] * nr_time_intervals_per_day,
                    list(range(1, nr_time_intervals_per_day + 1))
                ])
                to_add = dm.reshape_df(self.technology_data[node][tec].fitted_performance[tecs_flagged_for_clustering[node][tec]],
                                    series_names, nr_time_intervals_per_day)
                full_resolution = pd.concat([full_resolution, to_add], axis=1)
        return full_resolution

class DataHandle_AveragedData(EnergyHub):
    """
    DataHandle sub-class for handling averaged data

    This class is used to generate time series of averaged data based on a full resolution
    or clustered input data.
    """
    def __init__(self, data_in, nr_timesteps_averaged):
        """
        Constructor
        """
        data = copy.deepcopy(data_in)
        # Copy over data from old object
        self.node_data_full_resolution = data.node_data
        self.node_data = {}
        self.technology_data = data.technology_data
        self.network_data = data.network_data
        self.topology = data.topology

        if hasattr(data, 'k_means_specs'):
            self.k_means_specs = data.k_means_specs

        # flag tecs that contain time-dependent data
        self.tecs_flagged_for_clustering = dm.flag_tecs_for_clustering(self)

        # averaging specs
        self.averaged_specs = dm.simplification_specs(data.topology.timesteps)

        # perform averaging
        self.average_data(nr_timesteps_averaged)

    def average_data(self, nr_timesteps_averaged):
        # adjust timesteps
        end_interval = max(self.topology.timesteps)
        start_interval = min(self.topology.timesteps)
        time_resolution = str(nr_timesteps_averaged) + 'h'
        self.topology.timestep_length_h = nr_timesteps_averaged
        self.topology.timesteps = pd.date_range(start=start_interval, end=end_interval, freq=time_resolution)
        # flag tecs that contain time-dependent data
        tecs_flagged_for_clustering = self.tecs_flagged_for_clustering
        # Average all time-dependent data and write to self
        self.perform_averaging(nr_timesteps_averaged, tecs_flagged_for_clustering)
        # Write averaged specs
        self.averaged_specs.reduced_resolution = pd.DataFrame(
            data=np.ones(len(self.topology.timesteps)) * nr_timesteps_averaged,
            index=self.topology.timesteps,
            columns=['factor'])

    def perform_averaging(self, nr_timesteps_averaged, tecs_flagged_for_clustering):
        """
        Average all time-dependent data

        :param nr_timesteps_averaged: How many time-steps should be averaged?
        :param tecs_flagged_for_clustering: technologies that have time-dependent data
        """
        node_data = self.node_data_full_resolution
        for node in node_data:
            self.node_data[node] = {}
            for series in node_data[node]:
                self.node_data[node][series] = pd.DataFrame()
                if not (series == 'climate_data') and not (series == 'production_profile_curtailment'):
                    for carrier in node_data[node][series]:
                        series_data = dm.reshape_df(node_data[node][series][carrier],
                                                    None, nr_timesteps_averaged)
                        self.node_data[node][series][carrier] = series_data.mean(axis=1)
            for tec in tecs_flagged_for_clustering[node]:
                series_data = dm.reshape_df(
                    self.technology_data[node][tec].fitted_performance[tecs_flagged_for_clustering[node][tec]],
                    None, nr_timesteps_averaged)
                self.technology_data[node][tec].fitted_performance[
                    tecs_flagged_for_clustering[node][tec]] = series_data.mean(axis=1)
            self.node_data[node]['production_profile_curtailment'] = node_data[node]['production_profile_curtailment']


