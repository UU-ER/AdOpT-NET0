from pyomo.environ import *
from pyomo.environ import units as u
from pyomo.gdp import *
import copy

import src.config_model as m_config
import src.model_construction as mc

def add_networks(model, data):
    r"""
        Adds all networks as model blocks to respective node.

        This function initializes parameters and decision variables for all networks.
        For each network, it adds one block indexed by the set of all networks at the node :math:`Netw`.
        This function adds Sets, Parameters, Variables and Constraints.

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

        - CAPEX: ``var_CAPEX``
        - Variable OPEX: ``var_OPEX_variable``
        - Fixed OPEX: ``var_OPEX_fixed``
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

        :param object model: pyomo model
        :param DataHandle data:  instance of a DataHandle
        :return: model
        """
    def init_network(b_netw, netw):

        # NETWORK DATA
        netw_data = data.network_data[netw]
        existing = netw_data.existing
        size_is_int = netw_data.size_is_int
        size_min = netw_data.size_min
        size_max = netw_data.size_max
        decommission = netw_data.decommission
        economics = netw_data.economics
        performance_data = netw_data.performance_data
        energy_consumption = netw_data.energy_consumption
        connection = copy.deepcopy(netw_data.connection[:])
        distance = netw_data.distance
        if existing:
            size_initial = netw_data.size_initial

        # ARCS
        # Define sets of possible arcs
        def init_arcs_set(set):
            for from_node in connection:
                for to_node in connection[from_node].index:
                    if connection.at[from_node, to_node] == 1:
                        yield [from_node, to_node]
        b_netw.set_arcs = Set(initialize=init_arcs_set)

        # Define unique arcs (if bidirectional is possible)
        if performance_data['bidirectional'] == 1:
            def init_arcs_all(set):
                for from_node in connection:
                    for to_node in connection[from_node].index:
                        if connection.at[from_node, to_node] == 1:
                            connection.at[to_node, from_node] = 0
                            yield [from_node, to_node]
            b_netw.set_arcs_unique = Set(initialize=init_arcs_all)

        # SIZE
        if size_is_int:
            unit_size = u.dimensionless
            b_netw.para_rated_capacity =  Param(domain=NonNegativeReals,
                                                initialize=performance_data['rated_capacity'],
                                                units=unit_size)
        else:
            unit_size = u.MW
            b_netw.para_rated_capacity = Param(domain=NonNegativeReals, initialize=1, units=unit_size)

        b_netw.para_size_min = Param(domain=NonNegativeReals, initialize=size_min,
                                     units=unit_size)
        b_netw.para_size_max = Param(domain=NonNegativeReals, initialize=size_max,
                                     units=unit_size)

        if existing:
            # Parameters for initial size
            def init_size_initial(param, node_from, node_to):
                return size_initial.at[node_from, node_to]
            b_netw.para_size_initial = Param(b_netw.set_arcs, domain=NonNegativeReals, initialize=init_size_initial,
                                         units=unit_size)
            # Check if sizes in both direction are the same for bidirectional existing networks
            if performance_data['bidirectional'] == 1:
                for from_node in size_initial:
                    for to_node in size_initial[from_node].index:
                        assert size_initial.at[from_node, to_node] == size_initial.at[to_node, from_node]

        # CAPEX
        annualization_factor = mc.annualize(economics.discount_rate, economics.lifetime)

        if economics.capex_model == 1:
            b_netw.para_CAPEX_gamma1 = Param(domain=Reals,
                                             initialize=economics.capex_data['gamma1'] * annualization_factor,
                                             units=u.EUR / unit_size)
            b_netw.para_CAPEX_gamma2 = Param(domain=Reals,
                                             initialize=economics.capex_data['gamma2'] * annualization_factor,
                                             units=u.EUR)
        elif economics.capex_model == 2:
            b_netw.para_CAPEX_gamma1 = Param(domain=Reals,
                                             initialize=economics.capex_data['gamma1'] * annualization_factor,
                                             units=u.EUR / unit_size / u.km)
            b_netw.para_CAPEX_gamma2 = Param(domain=Reals,
                                             initialize=economics.capex_data['gamma2'] * annualization_factor,
                                             units=u.EUR)
        if economics.capex_model == 3:
            b_netw.para_CAPEX_gamma1 = Param(domain=Reals,
                                             initialize=economics.capex_data['gamma1'] * annualization_factor,
                                             units=u.EUR / unit_size / u.km)
            b_netw.para_CAPEX_gamma2 = Param(domain=Reals,
                                             initialize=economics.capex_data['gamma2'] * annualization_factor,
                                             units=u.EUR / unit_size)
            b_netw.para_CAPEX_gamma3 = Param(domain=Reals,
                                             initialize=economics.capex_data['gamma3'] * annualization_factor,
                                             units=u.EUR)

        # OPEX
        b_netw.para_OPEX_variable = Param(domain=Reals, initialize=economics.opex_variable,
                                          units=u.EUR / u.MWh)
        b_netw.para_OPEX_fixed = Param(domain=Reals, initialize=economics.opex_fixed,
                                       units=u.EUR / u.EUR)

        # Network losses (in % per km and flow)
        b_netw.para_loss_factor = Param(domain=Reals, initialize=performance_data['loss'],
                                       units=u.dimensionless)

        # Energy consumption at sending and receiving nodes
        b_netw.set_consumed_carriers = Set(initialize=energy_consumption.keys())
        if energy_consumption:
            def init_cons_send1(para, car):
                return energy_consumption[car]['send']['k_flow']
            b_netw.para_send_kflow = Param(b_netw.set_consumed_carriers, domain=Reals, initialize=init_cons_send1,
                                            units=u.dimensionless)
            def init_cons_send2(para, car):
                return energy_consumption[car]['send']['k_flowDistance']
            b_netw.para_send_kflowDistance = Param(b_netw.set_consumed_carriers, domain=Reals, initialize=init_cons_send2,
                                            units=u.dimensionless)
            def init_cons_receive1(para, car):
                return energy_consumption[car]['receive']['k_flow']
            b_netw.para_receive_kflow = Param(b_netw.set_consumed_carriers, domain=Reals, initialize=init_cons_receive1,
                                            units=u.dimensionless)
            def init_cons_receive2(para, car):
                return energy_consumption[car]['receive']['k_flowDistance']
            b_netw.para_receive_kflowDistance = Param(b_netw.set_consumed_carriers, domain=Reals, initialize=init_cons_receive2,
                                            units=u.dimensionless)

        # Network emissions
        b_netw.para_loss2emissions = Param(domain=NonNegativeReals, initialize=performance_data['loss2emissions'],
                                     units=u.t/u.dimensionless)
        b_netw.para_emissionfactor = Param(domain=NonNegativeReals, initialize=performance_data['emissionfactor'],
                                           units=u.t / u.MWh)

        # Minimal transport requirements
        b_netw.para_min_transport =  Param(domain=NonNegativeReals,
                                           initialize= performance_data['min_transport'],
                                           units=u.dimensionless)

        # Define set of transported carrier
        b_netw.set_netw_carrier = Set(initialize=[performance_data['carrier']])

        # Define inflows and outflows for each node
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

        # DECISION VARIABLES
        b_netw.var_inflow = Var(model.set_t, b_netw.set_netw_carrier, model.set_nodes, domain=NonNegativeReals)
        b_netw.var_outflow = Var(model.set_t, b_netw.set_netw_carrier, model.set_nodes, domain=NonNegativeReals)
        b_netw.var_consumption = Var(model.set_t, model.set_carriers, model.set_nodes,
                                         domain=NonNegativeReals)

        # Capex/Opex
        b_netw.var_CAPEX = Var(units=u.EUR)
        b_netw.var_OPEX_variable = Var(model.set_t, units=u.EUR)
        b_netw.var_OPEX_fixed = Var(units=u.EUR)
        if existing:
            b_netw.para_decommissioning_cost = Param(domain=Reals, initialize=economics.decommission_cost,
                                                    units=u.EUR / unit_size)

        # Emissions
        b_netw.var_netw_emissions_pos = Var(model.set_t, units=u.t)

        # Arcs
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
            # SIZE
            if existing:
                # Existing network
                if not decommission:
                    # Decommissioning not possible
                    b_arc.var_size = Param(domain=NonNegativeReals,
                                           initialize=b_netw.para_size_initial[node_from, node_to])
                else:
                    # Decommissioning possible
                    b_arc.var_size = Var(domain=NonNegativeReals,
                                         bounds=(b_netw.para_size_min, b_netw.para_size_initial[node_from, node_to]))
            else:
                # New network
                b_arc.var_size = Var(domain=NonNegativeReals,
                                     bounds=(b_netw.para_size_min, b_netw.para_size_max))

            # CAPEX auxilliary (used to calculate theoretical CAPEX)
            # For new technologies, this is equal to actual CAPEX
            # For existing technologies it is used to calculate fixed OPEX
            b_arc.var_CAPEX_aux = Var(units=u.EUR)
            def init_capex(const):
                if economics.capex_model == 1:
                    return b_arc.var_CAPEX_aux == b_arc.var_size * \
                           b_netw.para_CAPEX_gamma1 + b_netw.para_CAPEX_gamma2
                elif economics.capex_model == 2:
                    return b_arc.var_CAPEX_aux == b_arc.var_size * \
                           distance.at[node_from, node_to] * b_netw.para_CAPEX_gamma1 + b_netw.para_CAPEX_gamma2
                elif economics.capex_model == 3:
                    return b_arc.var_CAPEX_aux == b_arc.var_size * \
                           distance.at[node_from, node_to] * b_netw.para_CAPEX_gamma1 + \
                           b_arc.var_size * b_netw.para_CAPEX_gamma2 + \
                           b_netw.para_CAPEX_gamma3
            b_arc.const_capex_aux = Constraint(rule=init_capex)

            # CAPEX
            if existing:
                if not decommission:
                    b_arc.var_CAPEX = Param(domain=NonNegativeReals, initialize=0, units=u.EUR)
                else:
                    b_arc.var_CAPEX = Var(units=u.EUR)
                    b_arc.const_capex = Constraint(expr= b_arc.var_CAPEX == (b_netw.para_size_initial[node_from, node_to] - b_arc.var_size) \
                                                         * b_netw.para_decommissioning_cost)
            else:
                b_arc.var_CAPEX = Var(units=u.EUR)
                b_arc.const_CAPEX = Constraint(expr=b_arc.var_CAPEX == b_arc.var_CAPEX_aux)

            # OPEX VARIABLE
            b_arc.var_OPEX_variable = Var(model.set_t, units=u.EUR)

            # FLOW
            b_arc.var_flow = Var(model.set_t, domain=NonNegativeReals,
                                    bounds=(b_netw.para_size_min * b_netw.para_rated_capacity,
                                            b_netw.para_size_max * b_netw.para_rated_capacity))
            b_arc.var_losses = Var(model.set_t, domain=NonNegativeReals,
                                   bounds=(b_netw.para_size_min * b_netw.para_rated_capacity,
                                           b_netw.para_size_max * b_netw.para_rated_capacity))

            # Losses
            def init_flowlosses(const, t):
                return b_arc.var_losses[t] == b_arc.var_flow[t] * b_netw.para_loss_factor
            b_arc.const_flowlosses = Constraint(model.set_t, rule=init_flowlosses)

            # Flow-size-constraint
            def init_size_const_high(const, t):
                return b_arc.var_flow[t] <= b_arc.var_size * b_netw.para_rated_capacity
            b_arc.const_flow_size_high = Constraint(model.set_t, rule=init_size_const_high)

            def init_size_const_low(const, t):
                return b_arc.var_size * b_netw.para_rated_capacity * b_netw.para_min_transport <= \
                       b_arc.var_flow[t]
            b_arc.const_flow_size_low = Constraint(model.set_t, rule=init_size_const_low)

            # CONSUMPTION AT NODES
            if energy_consumption:
                b_arc.var_consumption_send = Var(model.set_t, b_netw.set_consumed_carriers,
                                                 domain=NonNegativeReals,
                                                 bounds=(b_netw.para_size_min, b_netw.para_size_max))
                b_arc.var_consumption_receive = Var(model.set_t, b_netw.set_consumed_carriers,
                                                    domain=NonNegativeReals,
                                                    bounds=(b_netw.para_size_min, b_netw.para_size_max))

                # Sending node
                def init_consumption_send(const, t, car):
                    return b_arc.var_consumption_send[t, car] == \
                           b_arc.var_flow[t] * b_netw.para_send_kflow[car] + \
                           b_arc.var_flow[t] * b_netw.para_send_kflowDistance[car] * \
                           distance.at[node_from, node_to]
                b_arc.const_consumption_send = Constraint(model.set_t, b_netw.set_consumed_carriers,
                                                         rule=init_consumption_send)

                # Receiving node
                def init_consumption_receive(const, t, car):
                    return b_arc.var_consumption_receive[t, car] == \
                           b_arc.var_flow[t] * b_netw.para_receive_kflow[car] + \
                           b_arc.var_flow[t] * b_netw.para_receive_kflowDistance[car] * \
                           distance.at[node_from, node_to]
                b_arc.const_consumption_receive = Constraint(model.set_t, b_netw.set_consumed_carriers,
                                                         rule=init_consumption_receive)

            # OPEX
            def init_OPEX_variable(const, t):
                return b_arc.var_OPEX_variable[t] == b_arc.var_flow[t] * \
                       b_netw.para_OPEX_variable
            b_arc.const_OPEX_variable = Constraint(model.set_t, rule=init_OPEX_variable)

        b_netw.arc_block = Block(b_netw.set_arcs, rule=arc_block_init)

        if performance_data['bidirectional'] == 1:
            m_config.presolve.big_m_transformation_required = 1

            if decommission or not existing:
                """
                bi-directional
                    size(from, to) = size(to, from)
                    disjunction for each segment allowing only one direction
                """

                # Size in both direction is the same
                def init_size_bidirectional(const, node_from, node_to):
                    return b_netw.arc_block[node_from, node_to].var_size == \
                           b_netw.arc_block[node_to, node_from].var_size
                b_netw.const_size_bidirectional = Constraint(b_netw.set_arcs_unique, rule=init_size_bidirectional)

            s_indicators = range(0, 2)

            def init_bidirectional(dis, t, node_from, node_to, ind):
                if ind == 0:
                    def init_bidirectional1(const):
                        return b_netw.arc_block[node_from, node_to].var_flow[t] == 0
                    dis.const_flow_zero = Constraint(rule=init_bidirectional1)
                else:
                    def init_bidirectional2(const):
                        return b_netw.arc_block[node_to, node_from].var_flow[t] == 0
                    dis.const_flow_zero = Constraint(rule=init_bidirectional2)
            b_netw.dis_one_direction_only = Disjunct(model.set_t, b_netw.set_arcs_unique, s_indicators,
                                                     rule=init_bidirectional)

            # Bind disjuncts
            def bind_disjunctions(dis, t, node_from, node_to):
                return [b_netw.dis_one_direction_only[t, node_from, node_to, i] for i in s_indicators]
            b_netw.disjunction_one_direction_only = Disjunction(model.set_t, b_netw.set_arcs_unique,
                                                                rule=bind_disjunctions)

        # Cost of network
        if performance_data['bidirectional'] == 1:
            arc_set = b_netw.set_arcs_unique
        else:
            arc_set = b_netw.set_arcs

        def init_capex(const):
            return sum(b_netw.arc_block[arc].var_CAPEX for arc in arc_set) == \
                   b_netw.var_CAPEX
        b_netw.const_CAPEX = Constraint(rule=init_capex)

        def init_opex_fixed(const):
            return b_netw.para_OPEX_fixed * sum(b_netw.arc_block[arc].var_CAPEX_aux for arc in arc_set) == \
                   b_netw.var_OPEX_fixed
        b_netw.const_OPEX_fixed = Constraint(rule=init_opex_fixed)

        def init_opex_variable(const, t):
            return sum(b_netw.arc_block[arc].var_OPEX_variable[t] for arc in b_netw.set_arcs) == \
                   b_netw.var_OPEX_variable[t]
        b_netw.const_OPEX_var = Constraint(model.set_t, rule=init_opex_variable)

        # Establish inflow and outflow for each node and this network
        """
        INDEXED BY: (node)
        inflow = sum(arc_block(from,node).flow - arc_block(from,node).losses for from in set_node_receives_from)
        outflow = sum(arc_block(node,to).flow for from in set_node_sends_to) 
        """

        def init_inflow(const, t, car, node):
            return b_netw.var_inflow[t, car, node] == sum(b_netw.arc_block[from_node, node].var_flow[t] - \
                                                     b_netw.arc_block[from_node, node].var_losses[t]
                                                     for from_node in b_netw.set_receives_from[node])
        b_netw.const_inflow = Constraint(model.set_t, b_netw.set_netw_carrier, model.set_nodes, rule=init_inflow)

        def init_outflow(const, t, car, node):
            return b_netw.var_outflow[t, car, node] == sum(b_netw.arc_block[node, from_node].var_flow[t] \
                                                      for from_node in b_netw.set_receives_from[node])
        b_netw.const_outflow = Constraint(model.set_t, b_netw.set_netw_carrier, model.set_nodes, rule=init_outflow)

        # Network emissions as sum over inflow
        def init_netw_emissions(const, t):
            return sum(b_netw.arc_block[arc].var_flow[t] for arc in b_netw.set_arcs) * \
                   b_netw.para_emissionfactor + \
                   sum(b_netw.arc_block[arc].var_losses[t] for arc in b_netw.set_arcs) * \
                   b_netw.para_loss2emissions \
                   == b_netw.var_netw_emissions_pos[t]
        b_netw.const_netw_emissions = Constraint(model.set_t, rule=init_netw_emissions)

        # Establish energy consumption for each node and this network
        def init_network_consumption(const, t, car, node):
            if energy_consumption:
                if car in b_netw.set_consumed_carriers:
                    return b_netw.var_consumption[t, car, node] == \
                           sum(b_netw.arc_block[node, to_node].var_consumption_send[t, car]
                               for to_node in b_netw.set_sends_to[node]) + \
                           sum(b_netw.arc_block[from_node, node].var_consumption_receive[t, car]
                               for from_node in b_netw.set_receives_from[node])
                else:
                    return b_netw.var_consumption[t, car, node] == 0
            else:
                return b_netw.var_consumption[t, car, node] == 0

        b_netw.const_netw_consumption = Constraint(model.set_t, model.set_carriers, model.set_nodes,
                                         rule=init_network_consumption)

        if m_config.presolve.big_m_transformation_required:
            mc.perform_disjunct_relaxation(b_netw)

        return b_netw
    model.network_block = Block(model.set_networks, rule=init_network)
    return model