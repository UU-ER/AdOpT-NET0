import warnings

from ..component import ModelComponent
from ..utilities import annualize, set_discount_rate, read_dict_value, perform_disjunct_relaxation, determine_variable_scaling, determine_constraint_scaling

import pandas as pd
import copy
from pyomo.environ import *
from pyomo.gdp import *

class Network(ModelComponent):
    """
    Class to read and manage data for networks
    """
    def __init__(self, netw_data):
        """
        Initializes technology class from network name

        The network name needs to correspond to the name of a JSON file in ./data/network_data.

        :param str network: name of technology to read data
        """
        super().__init__(netw_data)

        # General information
        self.connection = []
        self.distance = []
        self.size_max_arcs = []
        self.energy_consumption = {}

        # Technology Performance
        self.performance_data = netw_data['NetworkPerf']
        if self.performance_data['energyconsumption']:
            self.calculate_energy_consumption()

        self.set_nodes = []
        self.set_t = []

        self.scaling_factors = []
        if 'ScalingFactors' in netw_data:
            self.scaling_factors = netw_data['ScalingFactors']


    def calculate_energy_consumption(self):
        """
        Fits the performance parameters for a network, i.e. the consumption at each node.
        :param obj network: Dict read from json files with performance data and options for performance fits
        :param obj climate_data: Climate data
        :return: dict of performance coefficients used in the model
        """
        # Get energy consumption at nodes form file
        energycons = self.performance_data['energyconsumption']
        self.performance_data.pop('energyconsumption')

        for car in energycons:
            self.energy_consumption[car] = {}
            if energycons[car]['cons_model'] == 1:
                self.energy_consumption[car]['send'] = {}
                self.energy_consumption[car]['send'] = energycons[car]
                self.energy_consumption[car]['send'].pop('cons_model')
                self.energy_consumption[car]['receive'] = {}
                self.energy_consumption[car]['receive']['k_flow'] = 0
                self.energy_consumption[car]['receive']['k_flowDistance'] = 0
            elif energycons[car]['cons_model'] == 2:
                temp = energycons[car]
                self.energy_consumption[car]['send'] = {}
                self.energy_consumption[car]['send']['k_flow'] = round(temp['c'] * temp['T'] / temp['eta'] / \
                                                                       temp['LHV'] * ((temp['p'] / 30) **
                                                                                   ((temp['gam'] - 1) / temp[
                                                                                       'gam']) - 1), 4)
                self.energy_consumption[car]['send']['k_flowDistance'] = 0
                self.energy_consumption[car]['receive'] = {}
                self.energy_consumption[car]['receive']['k_flow'] = 0
                self.energy_consumption[car]['receive']['k_flowDistance'] = 0


    def calculate_max_size_arc(self):
        """
        Calculates the maximum size of each arc, if not specified
        :return:
        """
        if self.existing == 0:
            if not isinstance(self.size_max_arcs, pd.DataFrame):
                # Use max size
                self.size_max_arcs = pd.DataFrame(self.size_max, index=self.distance.index, columns=self.distance.columns)
        elif self.existing == 1:
            # Use initial size
            self.size_max_arcs = self.size_initial

    def construct_general_constraints(self, b_netw, energyhub):
        r"""
        Adds a network as model block.

        For each connection between nodes, an arc is created, with its respective cost, flows, losses and \
        consumption at nodes.

        Networks that can be used in two directions (e.g. electricity cables), are called bidirectional and are \
        treated respectively with their size and costs. Other networks, e.g. pipelines, require two installations \
        to be able to transport in two directions. As such their CAPEX is double.

        **Set declarations:**

        - Set of network carrier (i.e. only one carrier, that is transported in the network)
        - Set of all arcs (from_node, to_node)
        - In case the network is bidirectional: Set of unique arcs (i.e. for each pair of arcs, one unique entry)
        - Furthermore for each node:

            * A set of nodes it receives from
            * A set of nodes it sends to

        **Parameter declarations:**

        - Min Size (for each arc)
        - Max Size (for each arc)
        - :math:`{\gamma}_1, {\gamma}_2, {\gamma}_3` for CAPEX calculation  \
          (annualized from given data on up-front CAPEX, lifetime and discount rate)
        - Variable OPEX
        - Fixed OPEX
        - Network losses (in % per km and flow) :math:`{\mu}`
        - Minimum transport (% of rated capacity)
        - Parameters for energy consumption at receiving and sending node

        **Variable declarations:**

        - CAPEX: ``var_capex``
        - Variable OPEX: ``var_opex_variable``
        - Fixed OPEX: ``var_opex_fixed``
        - Furthermore for each node:

            * Inflow to node (as a sum of all inflows from other nodes): ``var_inflow``
            * Outflow from node (as a sum of all outflows toother nodes): ``var_outflow``
            * Consumption of other carriers (e.g. electricity required for compression of a gas): ``var_consumption``

        **Arc Block declaration**

        Each arc represents a connection between two nodes, and is thus indexed by (node_from, node_to). For each arc,
        the following components are defined. Each variable is indexed by the timestep :math:`t` (here left out
        for convinience).

        - Decision Variables:

            * Size :math:`S`
            * Flow :math:`flow`
            * Losses :math:`loss`
            * CAPEX: :math:`CAPEX`
            * Variable :math:`OPEXvariable`
            * Fixed :math:`OPEXfixed`
            * If consumption at nodes exists for network:

                * Consumption at sending node :math:`Consumption_{nodeFrom}`
                * Consumptoin at receiving node :math:`Consumption_{nodeTo}`

        - Constraint definitions

            * Flow losses:

              .. math::
                loss = flow * {\mu}

            * Flow constraints:

              .. math::
                S * minTransport \leq flow \leq S

            * Consumption at sending and receiving node:

              .. math::
                Consumption_{nodeFrom} = flow * k_{1, send} + flow * distance * k_{2, send}

              .. math::
                Consumption_{nodeTo} = flow * k_{1, receive} + flow * distance * k_{2, receive}

            * CAPEX of respective arc. Three different CAPEX models are implemented:
              Model 1:

              .. math::
                CAPEX_{arc} = {\gamma}_1 * S + {\gamma}_2

              Model 2:

              .. math::
                CAPEX_{arc} = {\gamma}_1 * distance * S + {\gamma}_2

              Model 3:

              .. math::
                CAPEX_{arc} = {\gamma}_1 * distance * S + {\gamma}_2 * S + {\gamma}_3

            * Variable OPEX:

              .. math::
                OPEXvariable_{arc} = CAPEX_{arc} * opex_{variable}

        **Constraint declarations**
        This part calculates variables for all respective nodes and enforces constraints for bi-directional networks.

        - If network is bi-directional, the sizes in both directions are equal, and only one direction of flow is
          possible in each time step:

          .. math::
            S_{nodeFrom, nodeTo} = S_{nodeTo, nodeFrom} \forall unique arcs

          .. math::
            flow_{nodeFrom, nodeTo} = 0 \lor flow_{nodeTo, nodeFrom} = 0

        - CAPEX calculation of whole network as a sum of CAPEX of all arcs. For bi-directional networks, each arc
          is only considered once, regardless of the direction of the arc.

        - OPEX fix, as fraction of total CAPEX

        - OPEX variable as a sum of variable OPEX for each arc

        - Total network costs as the sum of OPEX and CAPEX

        - Total inflow and outflow as a sum for each node:

          .. math::
            outflow_{node} = \sum_{nodeTo \in sendsto_{node}} flow_{node, nodeTo}

          .. math::
            inflow_{node} = \sum_{nodeFrom \in receivesFrom_{node}} flow_{nodeFrom, node} - losses_{nodeFrom, node}

        - Energy consumption of other carriers at each node.

        :param EnergyHub energyhub: instance of the energyhub
        :return: network model
        """
        # Data from energyhub
        self.set_nodes = energyhub.model.set_nodes
        self.set_t = energyhub.model.set_t_full

        b_netw = self.__define_possible_arcs(b_netw, energyhub)

        if self.performance_data['bidirectional'] == 1:
            b_netw = self.__define_unique_arcs(b_netw)


        b_netw = self.__define_size(b_netw)
        b_netw = self.__define_capex_parameters(b_netw, energyhub)
        b_netw = self.__define_opex_parameters(b_netw)
        b_netw = self.__define_emission_vars(b_netw)
        b_netw = self.__define_network_characteristics(b_netw)
        b_netw = self.__define_inflow_vars(b_netw)
        b_netw = self.__define_outflow_vars(b_netw)

        if self.energy_consumption:
            b_netw = self.__define_energyconsumption_parameters(b_netw)

        def arc_block_init(b_arc, node_from, node_to):
            """
            Establish each arc as a block

            INDEXED BY: (from, to)
            - size
            - flow
            - losses
            - consumption at from node
            - consumption at to node
            """

            b_arc.big_m_transformation_required = 0
            b_arc = self.__define_size_arc(b_arc, b_netw, node_from, node_to)
            b_arc = self.__define_capex_arc(b_arc, b_netw, node_from, node_to)
            b_arc = self.__define_flow(b_arc, b_netw)
            b_arc = self.__define_opex_arc(b_arc, b_netw)

            if self.energy_consumption:
                b_arc = self.__define_energyconsumption_arc(b_arc, b_netw)

            if b_arc.big_m_transformation_required:
                b_arc = perform_disjunct_relaxation(b_arc)

            return b_arc

        b_netw.arc_block = Block(b_netw.set_arcs, rule=arc_block_init)

        # CONSTRAINTS FOR BIDIRECTIONAL NETWORKS
        if self.performance_data['bidirectional']:
            b_netw = self.__define_bidirectional_constraints(b_netw)

        b_netw = self.__define_capex_total(b_netw)
        b_netw = self.__define_opex_total(b_netw)
        b_netw = self.__define_inflow_constraints(b_netw)
        b_netw = self.__define_outflow_constraints(b_netw)
        b_netw = self.__define_emission_constraints(b_netw)

        if self.energy_consumption:
            b_netw = self.__define_energyconsumption_total(b_netw)

        return b_netw

    def report_results(self, b_netw):
        """
        Function to report results of networks after optimization

        :param b_netw: network model block
        :return: dict results: holds results
        """
        self.results['time_independent'] = pd.DataFrame(columns=['Network',
                                              'fromNode',
                                              'toNode',
                                              'Size',
                                              'capex',
                                              'opex_fixed',
                                              'opex_variable',
                                              'total_flow',
                                              'total_emissions'
                                              ])

        for arc_name in b_netw.set_arcs:
            arc = b_netw.arc_block[arc_name]
            fromNode = arc_name[0]
            toNode = arc_name[1]
            s = arc.var_size.value
            capex = arc.var_capex.value
            # if energyhub.model_information.clustered_data:
            #     sequence = energyhub.k_means_specs.full_resolution['sequence']
            #     opex_var = sum(arc.var_opex_variable[sequence[t - 1]].value
            #                    for t in self.set_t)
            #     total_flow = sum(arc.var_flow[sequence[t - 1]].value
            #                      for t in self.set_t)
            #     total_emissions = ...
            # else:
            opex_var = sum(arc.var_opex_variable[t].value
                           for t in self.set_t)
            total_flow = sum(arc.var_flow[t].value
                             for t in self.set_t)
            opex_fix = b_netw.para_opex_fixed.value * arc.var_capex_aux.value
            total_emissions = sum(arc.var_flow[t].value for t in self.set_t) * b_netw.para_emissionfactor + \
                   sum(arc.var_losses[t].value for t in self.set_t) * b_netw.para_loss2emissions
            self.results['time_independent'].loc[len(self.results['time_independent'].index)] = \
                [self.name, fromNode, toNode, s, capex, opex_fix, opex_var, total_flow, total_emissions]

            index = pd.MultiIndex.from_tuples(zip(['flow'], [fromNode], [toNode]), names=['variable', 'fromNode', 'toNode'])
            data = pd.DataFrame([arc.var_flow[t].value for t in self.set_t], columns=index)
            self.results['time_dependent'] = pd.concat([self.results['time_dependent'], data], axis=1)

            index = pd.MultiIndex.from_tuples(zip(['losses'], [fromNode], [toNode]), names=['variable', 'fromNode', 'toNode'])
            data = pd.DataFrame([arc.var_losses[t].value for t in self.set_t], columns=index)
            self.results['time_dependent'] = pd.concat([self.results['time_dependent'], data], axis=1)

            if arc.find_component('var_consumption_send'):
                for car in b_netw.set_consumed_carriers:
                    # if energyhub.model_information.clustered_data:
                    #     index = pd.MultiIndex.from_tuples(zip(['consumption_send' + car], [fromNode], [toNode]),
                    #                                       names=['variable', 'fromNode', 'toNode'])
                    #     data = pd.DataFrame([arc.var_consumption_send[sequence[t-1], car].value for t in self.set_t], columns=index)
                    #     self.results['time_dependent'] = pd.concat([self.results['time_dependent'], data], axis=1)
                    #
                    #     index = pd.MultiIndex.from_tuples(zip(['consumption_receive' + car], [fromNode], [toNode]),
                    #                                       names=['variable', 'fromNode', 'toNode'])
                    #     data = pd.DataFrame([arc.var_consumption_receive[sequence[t-1], car].value for t in self.set_t], columns=index)
                    #     self.results['time_dependent'] = pd.concat([self.results['time_dependent'], data], axis=1)
                    # else:
                    index = pd.MultiIndex.from_tuples(zip(['consumption_send' + car], [fromNode], [toNode]),
                                                      names=['variable', 'fromNode', 'toNode'])
                    data = pd.DataFrame([arc.var_consumption_send[t, car].value for t in self.set_t], columns=index)
                    self.results['time_dependent'] = pd.concat([self.results['time_dependent'], data], axis=1)

                    index = pd.MultiIndex.from_tuples(zip(['consumption_receive' + car], [fromNode], [toNode]),
                                                      names=['variable', 'fromNode', 'toNode'])
                    data = pd.DataFrame([arc.var_consumption_receive[t, car].value for t in self.set_t], columns=index)
                    self.results['time_dependent'] = pd.concat([self.results['time_dependent'], data], axis=1)

        return self.results


    def scale_model(self, b_netw, model, configuration):
        """
        Scales technology model
        """

        f = self.scaling_factors
        f_global = configuration.scaling_factors

        model = determine_variable_scaling(model, b_netw, f, f_global)
        model = determine_constraint_scaling(model, b_netw, f, f_global)

        for arc in b_netw.arc_block:
            b_arc = b_netw.arc_block[arc]

            model = determine_variable_scaling(model, b_arc, f, f_global)
            model = determine_constraint_scaling(model, b_arc, f, f_global)

        return model

    def __define_possible_arcs(self, b_netw, energyhub):
        """
        Define all possible arcs that have a connection

        Sets defined:
        - set_arcs: Set of all arcs having a connection
        - set_receives_from: Set of nodes for each node specifying receiving from nodes
        - set_sends_to: Set of nodes for each node specifying sending to nodes
        """
        model = energyhub.model
        connection = copy.deepcopy(self.connection[:])

        def init_arcs_set(set):
            for from_node in connection:
                for to_node in connection[from_node].index:
                    if connection.at[from_node, to_node] == 1:
                        yield [from_node, to_node]
        b_netw.set_arcs = Set(initialize=init_arcs_set)

        def init_nodesIn(set, node):
            for i, j in b_netw.set_arcs:
                if j == node:
                    yield i
        b_netw.set_receives_from = Set(model.set_nodes, initialize=init_nodesIn)

        def init_nodesOut(set, node):
            for i, j in b_netw.set_arcs:
                if i == node:
                    yield j
        b_netw.set_sends_to = Set(model.set_nodes, initialize=init_nodesOut)

        return b_netw

    def __define_unique_arcs(self, b_netw):
        """
        Define arcs that are unique (one arc per direction)
        """
        connection = copy.deepcopy(self.connection[:])

        def init_arcs_all(set):
            for from_node in connection:
                for to_node in connection[from_node].index:
                    if connection.at[from_node, to_node] == 1:
                        connection.at[to_node, from_node] = 0
                        yield [from_node, to_node]
        b_netw.set_arcs_unique = Set(initialize=init_arcs_all)

        return b_netw

    def __define_size(self, b_netw):
        """
        Defines parameters related to network size.

        Parameters defined:
        - size min
        - size max
        - size initial (for existing technologies)
        """
        performance_data = self.performance_data

        if 'rated_capacity' in performance_data:
            b_netw.para_rated_capacity = performance_data['rated_capacity']
        else:
            b_netw.para_rated_capacity = 1

        b_netw.para_size_min = Param(domain=NonNegativeReals, initialize=self.size_min, mutable=True)

        if self.existing:
            # Parameters for initial size
            def init_size_initial(param, node_from, node_to):
                return self.size_initial.at[node_from, node_to]

            b_netw.para_size_initial = Param(b_netw.set_arcs, domain=NonNegativeReals, initialize=init_size_initial)
            # Check if sizes in both direction are the same for bidirectional existing networks
            if performance_data['bidirectional'] == 1:
                for from_node in self.size_initial:
                    for to_node in self.size_initial[from_node].index:
                        assert self.size_initial.at[from_node, to_node] == self.size_initial.at[to_node, from_node]
        return b_netw

    def __define_capex_parameters(self, b_netw, energyhub):
        """
        Defines variables and parameters related to technology capex.

        Parameters defined:
        - unit capex

        Variables defined:
        - total capex for network
        """

        configuration = energyhub.configuration

        economics = self.economics

        # CHECK FOR GLOBAL ECONOMIC OPTIONS
        discount_rate = set_discount_rate(configuration, economics)

        # CAPEX
        annualization_factor = annualize(discount_rate, economics.lifetime)

        if economics.capex_model == 1:
            b_netw.para_capex_gamma1 = Param(domain=Reals, mutable=True,
                                             initialize=economics.capex_data['gamma1'] * annualization_factor)
            b_netw.para_capex_gamma2 = Param(domain=Reals, mutable=True,
                                             initialize=economics.capex_data['gamma2'] * annualization_factor)
        elif economics.capex_model == 2:
            b_netw.para_capex_gamma1 = Param(domain=Reals, mutable=True,
                                             initialize=economics.capex_data['gamma1'] * annualization_factor)
            b_netw.para_capex_gamma2 = Param(domain=Reals, mutable=True,
                                             initialize=economics.capex_data['gamma2'] * annualization_factor)
        if economics.capex_model == 3:
            b_netw.para_capex_gamma1 = Param(domain=Reals, mutable=True,
                                             initialize=economics.capex_data['gamma1'] * annualization_factor)
            b_netw.para_capex_gamma2 = Param(domain=Reals, mutable=True,
                                             initialize=economics.capex_data['gamma2'] * annualization_factor)
            b_netw.para_capex_gamma3 = Param(domain=Reals, mutable=True,
                                             initialize=economics.capex_data['gamma3'] * annualization_factor)

        b_netw.var_capex = Var()

        return b_netw

    def __define_opex_parameters(self, b_netw):
        """
        Defines OPEX parameters (fixed and variable)

        Parameters defined:
        - variable OPEX
        - fixed OPEX

        Variables defined:
        - variable OPEX for network
        - fixed OPEX for network
        """
        economics = self.economics

        b_netw.para_opex_variable = Param(domain=Reals, initialize=economics.opex_variable, mutable=True)
        b_netw.para_opex_fixed = Param(domain=Reals, initialize=economics.opex_fixed, mutable=True)

        b_netw.var_opex_variable = Var(self.set_t)
        b_netw.var_opex_fixed = Var()
        if self.existing:
            b_netw.para_decommissioning_cost = Param(domain=Reals, initialize=economics.decommission_cost, mutable=True)

        return b_netw

    def __define_emission_vars(self, b_netw):
        """
        Defines network emissions

        Parameters defined:
        - loss2emissions
        - emissionfactor
        """
        b_netw.para_loss2emissions = self.performance_data['loss2emissions']
        b_netw.para_emissionfactor = self.performance_data['emissionfactor']

        b_netw.var_netw_emissions_pos = Var(self.set_t)

        return b_netw

    def __define_network_characteristics(self, b_netw):
        """
        Defines transported carrier, losses and minimum transport requirements

        Sets defined:
        - carriers transported

        Parameters defined:
        - loss factor
        - minimal transport requirements
        """
        # Define set of transported carrier
        b_netw.set_netw_carrier = Set(initialize=[self.performance_data['carrier']])

        # Network losses (in % per km and flow)
        b_netw.para_loss_factor = self.performance_data['loss']

        # Minimal transport requirements
        b_netw.para_min_transport = self.performance_data['min_transport']

        return b_netw

    def __define_inflow_vars(self, b_netw):
        """
        Defines network inflow (i.e. sum of inflow to one node)
        """
        b_netw.var_inflow = Var(self.set_t, b_netw.set_netw_carrier, self.set_nodes, domain=NonNegativeReals)
        return b_netw

    def __define_outflow_vars(self, b_netw):
        """
        Defines network outflow (i.e. sum of outflow to one node)
        """
        b_netw.var_outflow = Var(self.set_t, b_netw.set_netw_carrier, self.set_nodes, domain=NonNegativeReals)
        return b_netw

    def __define_energyconsumption_parameters(self, b_netw):
        """
        Constructs constraints for network energy consumption
        """
        # Set of consumed carriers
        b_netw.set_consumed_carriers = Set(initialize=list(self.energy_consumption.keys()))

        # Parameters
        def init_cons_send1(para, car):
            return self.energy_consumption[car]['send']['k_flow']

        b_netw.para_send_kflow = Param(b_netw.set_consumed_carriers, domain=Reals, initialize=init_cons_send1)

        def init_cons_send2(para, car):
            return self.energy_consumption[car]['send']['k_flowDistance']

        b_netw.para_send_kflowDistance = Param(b_netw.set_consumed_carriers, domain=Reals, initialize=init_cons_send2)

        def init_cons_receive1(para, car):
            return self.energy_consumption[car]['receive']['k_flow']

        b_netw.para_receive_kflow = Param(b_netw.set_consumed_carriers, domain=Reals, initialize=init_cons_receive1)

        def init_cons_receive2(para, car):
            return self.energy_consumption[car]['receive']['k_flowDistance']

        b_netw.para_receive_kflowDistance = Param(b_netw.set_consumed_carriers, domain=Reals,
                                                  initialize=init_cons_receive2)

        # Consumption at each node
        b_netw.var_consumption = Var(self.set_t, b_netw.set_consumed_carriers, self.set_nodes,
                                     domain=NonNegativeReals)

        return b_netw

    def __define_size_arc(self, b_arc, b_netw, node_from, node_to):
        """
        Defines the size of an arc

        Variables defined:
        - var_size for each arc
        """
        if self.size_is_int:
            size_domain = NonNegativeIntegers
        else:
            size_domain = NonNegativeReals

        b_arc.para_size_max = Param(domain=size_domain,
                                    initialize=self.size_max_arcs.at[node_from, node_to])

        b_arc.distance = self.distance.at[node_from, node_to]

        if self.existing:
            # Existing network
            if not self.decommission:
                # Decommissioning not possible
                b_arc.var_size = Param(domain=size_domain,
                                       initialize=b_netw.para_size_initial[node_from, node_to])
            else:
                # Decommissioning possible
                b_arc.var_size = Var(domain=size_domain,
                                     bounds=(b_netw.para_size_min, b_netw.para_size_initial[node_from, node_to]))
        else:
            # New network
            b_arc.var_size = Var(domain=size_domain,
                                 bounds=(b_netw.para_size_min, b_arc.para_size_max))

        return b_arc

    def __define_capex_arc(self, b_arc, b_netw, node_from, node_to):
        """
        Defines the capex of an arc and corresponding constraints

        Variables defined:
        - var_capex for each arc
        """
        def calculate_max_capex():
            if self.economics.capex_model == 1:
                max_capex = b_arc.para_size_max * \
                            b_netw.para_capex_gamma1 + b_netw.para_capex_gamma2
            elif self.economics.capex_model == 2:
                max_capex = b_arc.para_size_max * \
                            b_arc.distance * b_netw.para_capex_gamma1 + b_netw.para_capex_gamma2
            elif self.economics.capex_model == 3:
                max_capex = b_arc.para_size_max * \
                            b_arc.distance * b_netw.para_capex_gamma1 + \
                            b_arc.para_size_max * b_netw.para_capex_gamma2 + \
                            b_netw.para_capex_gamma3
            return (0, max_capex)

        # CAPEX auxilliary (used to calculate theoretical CAPEX)
        # For new technologies, this is equal to actual CAPEX
        # For existing technologies it is used to calculate fixed OPEX
        b_arc.var_capex_aux = Var(bounds=calculate_max_capex())

        def init_capex(const):
            if self.economics.capex_model == 1:
                return b_arc.var_capex_aux == b_arc.var_size * \
                       b_netw.para_capex_gamma1 + b_netw.para_capex_gamma2
            elif self.economics.capex_model == 2:
                return b_arc.var_capex_aux == b_arc.var_size * \
                       b_arc.distance * b_netw.para_capex_gamma1 + b_netw.para_capex_gamma2
            elif self.economics.capex_model == 3:
                return b_arc.var_capex_aux == b_arc.var_size * \
                       b_arc.distance * b_netw.para_capex_gamma1 + \
                       b_arc.var_size * b_netw.para_capex_gamma2 + \
                       b_netw.para_capex_gamma3

        # CAPEX Variable
        if self.existing and not self.decommission:
            b_arc.var_capex = Param(domain=NonNegativeReals, initialize=0)
        else:
            b_arc.var_capex = Var(bounds=calculate_max_capex())

        # CAPEX aux:
        if self.existing and not self.decommission:
            b_arc.const_capex_aux = Constraint(rule=init_capex)
        else:
            b_arc.big_m_transformation_required = 1
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

        # CAPEX and CAPEX aux
        if self.existing and self.decommission:
            b_arc.const_capex = Constraint(
                expr=b_arc.var_capex == (b_netw.para_size_initial[node_from, node_to] - b_arc.var_size) \
                     * b_netw.para_decommissioning_cost)
        elif not self.existing:
            b_arc.const_capex = Constraint(expr=b_arc.var_capex == b_arc.var_capex_aux)

        return b_arc

    def __define_flow(self, b_arc, b_netw):
        """
        Defines the flow through one arc and respective losses
        """

        b_arc.var_flow = Var(self.set_t, domain=NonNegativeReals,
                             bounds=(b_netw.para_size_min * b_netw.para_rated_capacity,
                                     b_arc.para_size_max * b_netw.para_rated_capacity))
        b_arc.var_losses = Var(self.set_t, domain=NonNegativeReals,
                               bounds=(b_netw.para_size_min * b_netw.para_rated_capacity,
                                       b_arc.para_size_max * b_netw.para_rated_capacity))

        # Losses
        def init_flowlosses(const, t):
            return b_arc.var_losses[t] == b_arc.var_flow[t] * b_netw.para_loss_factor * b_arc.distance

        b_arc.const_flowlosses = Constraint(self.set_t, rule=init_flowlosses)

        # Flow-size-constraint
        def init_size_const_high(const, t):
            return b_arc.var_flow[t] <= b_arc.var_size * b_netw.para_rated_capacity

        b_arc.const_flow_size_high = Constraint(self.set_t, rule=init_size_const_high)

        def init_size_const_low(const, t):
            return b_arc.var_size * b_netw.para_rated_capacity * b_netw.para_min_transport <= \
                   b_arc.var_flow[t]

        b_arc.const_flow_size_low = Constraint(self.set_t, rule=init_size_const_low)
        return b_arc

    def __define_energyconsumption_arc(self, b_arc, b_netw):
        """
        Defines the energyconsumption for an arc
        """
        b_arc.var_consumption_send = Var(self.set_t, b_netw.set_consumed_carriers,
                                         domain=NonNegativeReals,
                                         bounds=(b_netw.para_size_min, b_arc.para_size_max))
        b_arc.var_consumption_receive = Var(self.set_t, b_netw.set_consumed_carriers,
                                            domain=NonNegativeReals,
                                            bounds=(b_netw.para_size_min, b_arc.para_size_max))

        # Sending node
        def init_consumption_send(const, t, car):
            return b_arc.var_consumption_send[t, car] == \
                   b_arc.var_flow[t] * b_netw.para_send_kflow[car] + \
                   b_arc.var_flow[t] * b_netw.para_send_kflowDistance[car] * \
                   b_arc.distance

        b_arc.const_consumption_send = Constraint(self.set_t, b_netw.set_consumed_carriers,
                                                  rule=init_consumption_send)

        # Receiving node
        def init_consumption_receive(const, t, car):
            return b_arc.var_consumption_receive[t, car] == \
                   b_arc.var_flow[t] * b_netw.para_receive_kflow[car] + \
                   b_arc.var_flow[t] * b_netw.para_receive_kflowDistance[car] * \
                   b_arc.distance

        b_arc.const_consumption_receive = Constraint(self.set_t, b_netw.set_consumed_carriers,
                                                     rule=init_consumption_receive)

        return b_arc

    def __define_opex_arc(self, b_arc, b_netw):
        """
        Defines OPEX per Arc
        """
        b_arc.var_opex_variable = Var(self.set_t)

        def init_opex_variable(const, t):
            return b_arc.var_opex_variable[t] == b_arc.var_flow[t] * \
                   b_netw.para_opex_variable

        b_arc.const_opex_variable = Constraint(self.set_t, rule=init_opex_variable)
        return b_arc

    def __define_bidirectional_constraints(self, b_netw):
        """
        Defines constraints necessary, in case one arc can transport in two directions.

        I.e.:
        - size is equal in both directions
        - One directional flow possible only
        """

        self.big_m_transformation_required = 1

        # Size in both direction is the same
        if self.decommission or not self.existing:
            def init_size_bidirectional(const, node_from, node_to):
                return b_netw.arc_block[node_from, node_to].var_size == \
                       b_netw.arc_block[node_to, node_from].var_size

            b_netw.const_size_bidirectional = Constraint(b_netw.set_arcs_unique, rule=init_size_bidirectional)

        s_indicators = range(0, 2)

        # Cut according to Germans work
        def init_cut_bidirectional(const, t, node_from, node_to):
            return b_netw.arc_block[node_from, node_to].var_flow[t] + b_netw.arc_block[node_to, node_from].var_flow[t]\
                   <= b_netw.arc_block[node_from, node_to].var_size
        b_netw.const_cut_bidirectional = Constraint(self.set_t, b_netw.set_arcs_unique, rule=init_cut_bidirectional)

        # Flow only possible in one direction
        def init_bidirectional(dis, t, node_from, node_to, ind):
            if ind == 0:
                def init_bidirectional1(const):
                    return b_netw.arc_block[node_from, node_to].var_flow[t] == 0
                dis.const_flow_zero = Constraint(rule=init_bidirectional1)

            else:
                def init_bidirectional2(const):
                    return b_netw.arc_block[node_to, node_from].var_flow[t] == 0
                dis.const_flow_zero = Constraint(rule=init_bidirectional2)

        b_netw.dis_one_direction_only = Disjunct(self.set_t, b_netw.set_arcs_unique, s_indicators,
                                                 rule=init_bidirectional)

        # Bind disjuncts
        def bind_disjunctions(dis, t, node_from, node_to):
            return [b_netw.dis_one_direction_only[t, node_from, node_to, i] for i in s_indicators]

        b_netw.disjunction_one_direction_only = Disjunction(self.set_t, b_netw.set_arcs_unique,
                                                            rule=bind_disjunctions)

        return b_netw

    def __define_capex_total(self, b_netw):
        """
        Defines total CAPEX of network
        """
        if self.performance_data['bidirectional']:
            arc_set = b_netw.set_arcs_unique
        else:
            arc_set = b_netw.set_arcs

        def init_capex(const):
            return sum(b_netw.arc_block[arc].var_capex for arc in arc_set) == \
                   b_netw.var_capex

        b_netw.const_capex = Constraint(rule=init_capex)

        return b_netw

    def __define_opex_total(self, b_netw):
        """
        Defines total OPEX of network
        """
        if self.performance_data['bidirectional']:
            arc_set = b_netw.set_arcs_unique
        else:
            arc_set = b_netw.set_arcs

        def init_opex_fixed(const):
            return b_netw.para_opex_fixed * sum(b_netw.arc_block[arc].var_capex_aux for arc in arc_set) == \
                   b_netw.var_opex_fixed

        b_netw.const_opex_fixed = Constraint(rule=init_opex_fixed)

        def init_opex_variable(const, t):
            return sum(b_netw.arc_block[arc].var_opex_variable[t] for arc in b_netw.set_arcs) == \
                   b_netw.var_opex_variable[t]

        b_netw.const_opex_var = Constraint(self.set_t, rule=init_opex_variable)
        return b_netw

    def __define_inflow_constraints(self, b_netw):
        """
        Connects the arc flows to inflow at each node
        inflow = sum(arc_block(from,node).flow - arc_block(from,node).losses for from in set_node_receives_from)
        """

        def init_inflow(const, t, car, node):
            return b_netw.var_inflow[t, car, node] == sum(b_netw.arc_block[from_node, node].var_flow[t] - \
                                                          b_netw.arc_block[from_node, node].var_losses[t]
                                                          for from_node in b_netw.set_receives_from[node])

        b_netw.const_inflow = Constraint(self.set_t, b_netw.set_netw_carrier, self.set_nodes, rule=init_inflow)
        return b_netw

    def __define_outflow_constraints(self, b_netw):
        """
        Connects the arc flows to outflow at each node
        outflow = sum(arc_block(node,to).flow for from in set_node_sends_to)
        """

        def init_outflow(const, t, car, node):
            return b_netw.var_outflow[t, car, node] == sum(b_netw.arc_block[node, to_node].var_flow[t] \
                                                           for to_node in b_netw.set_sends_to[node])

        b_netw.const_outflow = Constraint(self.set_t, b_netw.set_netw_carrier, self.set_nodes, rule=init_outflow)
        return b_netw

    def __define_emission_constraints(self, b_netw):
        """
        Defines Emissions from network
        """

        def init_netw_emissions(const, t):
            return sum(b_netw.arc_block[arc].var_flow[t] for arc in b_netw.set_arcs) * \
                   b_netw.para_emissionfactor + \
                   sum(b_netw.arc_block[arc].var_losses[t] for arc in b_netw.set_arcs) * \
                   b_netw.para_loss2emissions \
                   == b_netw.var_netw_emissions_pos[t]

        b_netw.const_netw_emissions = Constraint(self.set_t, rule=init_netw_emissions)
        return b_netw

    def __define_energyconsumption_total(self, b_netw):
        """
        Defines network consumption at each node
        """

        def init_network_consumption(const, t, car, node):
            return b_netw.var_consumption[t, car, node] == \
                   sum(b_netw.arc_block[node, to_node].var_consumption_send[t, car]
                       for to_node in b_netw.set_sends_to[node]) + \
                   sum(b_netw.arc_block[from_node, node].var_consumption_receive[t, car]
                       for from_node in b_netw.set_receives_from[node])

        b_netw.const_netw_consumption = Constraint(self.set_t, b_netw.set_consumed_carriers, self.set_nodes,
                                                   rule=init_network_consumption)

        return b_netw

