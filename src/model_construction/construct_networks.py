from pyomo.environ import *
from pyomo.environ import units as u
from pyomo.gdp import *
import numpy as np
import numbers


def add_networks(model, data):
    model.set_network_carriers = Set(initialize = data.network_data.keys())

    def network_carrier_rule(b_netw_car, car):
        b_netw_car.set_networks = Set(initialize=data.network_data[car].keys())
        b_netw_car.var_inflow = Var(model.set_t, model.set_nodes, domain=NonNegativeReals)
        b_netw_car.var_outflow = Var(model.set_t, model.set_nodes, domain=NonNegativeReals)

        # Capex/Opex
        b_netw_car.var_cost = Var(units=u.EUR)  # capex


        def init_network(b_netw, netw):

            # Get network data
            netw_data = data.network_data[car][netw]
            connection = netw_data['connection'][:]
            distance = netw_data['distance']

            # Define parameters
            eta = 0.99
            cons_send = 0
            cons_receive = 0
            bi_directional = 0
            netw_data['Economics'] = {}
            netw_data['TechnologyPerf'] = {}

            netw_data['Economics']['unit_CAPEX_annual']  = 2
            netw_data['Economics']['OPEX_variable'] =0
            netw_data['Economics']['OPEX_fixed'] =0
            netw_data['TechnologyPerf']['size_min'] = 0
            netw_data['TechnologyPerf']['size_max'] = 1000
            size_is_integer = 0
            capex_model = 1
                
            # Define Parameters
            if isinstance(netw_data['TechnologyPerf']['size_min'], numbers.Number):
                size_min = netw_data['TechnologyPerf']['size_min']
            else:
                size_min = min(netw_data['TechnologyPerf']['size_min'])
            if isinstance(netw_data['TechnologyPerf']['size_max'], numbers.Number):
                size_max = netw_data['TechnologyPerf']['size_max']
            else:
                size_max = max(netw_data['TechnologyPerf']['size_max'])

            if size_is_integer:
                unit_size = []
            else:
                unit_size = u.MW
            b_netw.para_size_min = Param(domain=NonNegativeReals, initialize=size_min, units=unit_size)
            b_netw.para_size_max = Param(domain=NonNegativeReals, initialize=size_max, units=unit_size)
            b_netw.para_unit_CAPEX = Param(domain=Reals, initialize=netw_data['Economics']['unit_CAPEX_annual'],
                                          units=u.EUR / unit_size/u.km )
            b_netw.para_OPEX_variable = Param(domain=Reals, initialize=netw_data['Economics']['OPEX_variable'],
                                             units=u.EUR / u.MWh)
            b_netw.para_OPEX_fixed = Param(domain=Reals, initialize=netw_data['Economics']['OPEX_fixed'],
                                          units=u.EUR / u.EUR)

            # Define possible arcs
            def init_arcs_set(set):
                for from_node in connection:
                    for to_node in connection[from_node].index:
                        if connection.at[from_node, to_node] == 1:
                            yield [from_node, to_node]
            b_netw.set_arcs = Set(initialize=init_arcs_set)


            if bi_directional == 1:
                def init_arcs_all(set):
                    for from_node in connection:
                        for to_node in connection[from_node].index:
                            if connection.at[from_node, to_node] == 1:
                                connection.at[to_node, from_node] = 0
                                yield [from_node, to_node]
                b_netw.set_arcs_unique = Set(initialize=init_arcs_all)

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

            b_netw.var_inflow = Var(model.set_t, model.set_nodes, domain=NonNegativeReals)
            b_netw.var_outflow = Var(model.set_t, model.set_nodes, domain=NonNegativeReals)

            # Capex/Opex
            b_netw.var_CAPEX = Var(units=u.EUR)  # capex
            b_netw.var_OPEX_variable = Var(units=u.EUR)  # variable opex
            b_netw.var_OPEX_fixed = Var(units=u.EUR)  # fixed opex


            # Establish each arc as a block with
            """
            INDEXED BY: (from, to)
            - size
            - flow
            - losses
            - consumption at from node
            - consumption at to node
            """
            def arc_block_init(b_arc, node_from, node_to):
                b_arc.var_size = Var(domain=NonNegativeReals,
                                     bounds=(b_netw.para_size_min, b_netw.para_size_max))
                b_arc.var_flow = Var(model.set_t, domain=NonNegativeReals,
                                     bounds=(b_netw.para_size_min, b_netw.para_size_max))
                b_arc.var_losses = Var(model.set_t, domain=NonNegativeReals,
                                       bounds=(b_netw.para_size_min, b_netw.para_size_max))
                b_arc.var_consumption_n1 = Var(model.set_t, domain=NonNegativeReals,
                                               bounds=(b_netw.para_size_min, b_netw.para_size_max))
                b_arc.var_consumption_n2 = Var(model.set_t, domain=NonNegativeReals,
                                               bounds=(b_netw.para_size_min, b_netw.para_size_max))
                b_arc.var_CAPEX = Var(units=u.EUR)
                b_arc.var_OPEX_variable = Var(units=u.EUR)
                b_arc.var_OPEX_fixed = Var(units=u.EUR)
                def init_flowlosses(const, t):
                    return b_arc.var_losses[t] == b_arc.var_flow[t] * (1 - eta)
                b_arc.const_flowlosses = Constraint(model.set_t, rule=init_flowlosses)
                def init_consumption_n1(const, t):
                    return b_arc.var_consumption_n1[t] == b_arc.var_flow[t] * cons_send
                b_arc.const_consumption_n1 = Constraint(model.set_t, rule=init_consumption_n1)
                def init_consumption_n2(const, t):
                    return b_arc.var_consumption_n2[t] == b_arc.var_flow[t] * cons_receive
                b_arc.const_consumption_n2 = Constraint(model.set_t, rule=init_consumption_n2)
                def init_size_const(const, t):
                    return b_arc.var_flow[t] <= b_arc.var_size
                b_arc.const_flow_size = Constraint(model.set_t, rule=init_size_const)
                def init_capex(const):
                    return b_arc.var_CAPEX == b_arc.var_size * distance.at[node_from, node_to] * b_netw.para_unit_CAPEX
                b_arc.const_capex = Constraint(rule=init_capex)
                b_arc.const_OPEX_fixed =Constraint(expr=b_arc.var_CAPEX * b_netw.para_OPEX_fixed == b_arc.var_OPEX_fixed)
                def init_OPEX_variable(const):
                    return b_arc.var_OPEX_variable == sum(b_arc.var_flow[t] for t in model.set_t) * \
                           b_netw.para_OPEX_variable
                b_arc.const_OPEX_variable = Constraint(rule=init_OPEX_variable)
            b_netw.arc_block = Block(b_netw.set_arcs, rule=arc_block_init)

            if bi_directional == 1:
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
                def init_bi_directional(dis, t, node_from, node_to, ind):
                    #TODO: set not as set
                    if ind == 0:
                        def init_bi_directional1(const):
                            return b_netw.arc_block[node_from, node_to].var_flow[t] == 0
                        dis.const_set_flow_zero = Constraint(rule=init_bi_directional1)
                    else:
                        def init_bi_directional2(const):
                            return b_netw.arc_block[node_to, node_from].var_flow[t] == 0
                        dis.const_set_flow_zero = Constraint(rule=init_bi_directional2)
                b_netw.dis_one_direction_only = Disjunct(model.set_t, b_netw.set_arcs_unique, s_indicators,
                                                         rule=init_bi_directional)

                # Bind disjuncts
                def bind_disjunctions(dis, t, node_from, node_to):
                    return [b_netw.dis_one_direction_only[t, node_from, node_to, i] for i in s_indicators]
                b_netw.disjunction_one_direction_only = Disjunction(model.set_t, b_netw.set_arcs_unique,
                                                                    rule=bind_disjunctions)

            # Cost of network
            if bi_directional == 1:
                arc_set = b_netw.set_arcs_unique
            else:
                arc_set = b_netw.set_arcs

            def init_capex(const):
                return sum(b_netw.arc_block[arc].var_CAPEX for arc in arc_set) == \
                        b_netw.var_CAPEX
            b_netw.const_CAPEX_arc = Constraint(rule=init_capex)
            def init_opex_fixed(const):
                return sum(b_netw.arc_block[arc].var_OPEX_fixed for arc in arc_set) == \
                        b_netw.var_OPEX_fixed
            b_netw.const_OPEX_fixed_arc = Constraint(rule=init_opex_fixed)
            def init_opex_variable(const):
                return sum(b_netw.arc_block[arc].var_OPEX_variable for arc in arc_set) == \
                        b_netw.var_OPEX_variable
            b_netw.const_OPEX_var_arc = Constraint(rule=init_opex_variable)

            # Establish inflow and outflow for each node and this network
            """
            INDEXED BY: (node)
            inflow = sum(arc_block(from,node).flow - arc_block(from,node).losses for from in set_node_receives_from)
            outflow = sum(arc_block(node,to).flow for from in set_node_sends_to) 
            """
            def init_inflow(const, t, node):
                return b_netw.var_inflow[t, node] == sum(b_netw.arc_block[from_node,node].var_flow[t] - \
                                                         b_netw.arc_block[from_node,node].var_losses[t]
                                                         for from_node in b_netw.set_receives_from[node])
            b_netw.const_inflow = Constraint(model.set_t, model.set_nodes, rule=init_inflow)

            def init_outflow(const, t, node):
                return b_netw.var_outflow[t, node] == sum(b_netw.arc_block[node, from_node].var_flow[t] \
                                                         for from_node in b_netw.set_receives_from[node])
            b_netw.const_outflow = Constraint(model.set_t, model.set_nodes, rule=init_outflow)
            return b_netw

        # sum up costs and inflows/outflows for all networks for respective carrier
        b_netw_car.network_block = Block(b_netw_car.set_networks, rule=init_network)

        def init_netw_car_cost(const):
            return b_netw_car.var_cost == sum(b_netw_car.network_block[netw].var_CAPEX +
                                       b_netw_car.network_block[netw].var_OPEX_fixed +
                                       b_netw_car.network_block[netw].var_OPEX_variable
                                       for netw in b_netw_car.set_networks)
        b_netw_car.const_cost = Constraint(rule=init_netw_car_cost)

        def init_netw_car_totalInflowAtNode(const, t, node):
            return b_netw_car.var_inflow[t, node] == \
                   sum(b_netw_car.network_block[netw].var_inflow[t, node] for netw in b_netw_car.set_networks)
        b_netw_car.const_totalInflowAtNode = Constraint(model.set_t, model.set_nodes, rule=init_netw_car_totalInflowAtNode)

        def init_netw_car_totalOutflowAtNode(const, t, node):
            return b_netw_car.var_outflow[t, node] == \
                   sum(b_netw_car.network_block[netw].var_outflow[t, node] for netw in b_netw_car.set_networks)
        b_netw_car.const_totalOutflowAtNode = Constraint(model.set_t, model.set_nodes, rule=init_netw_car_totalOutflowAtNode)

        return b_netw_car

    model.network_carrier_blocks = Block(model.set_network_carriers, rule=network_carrier_rule)
    return model









    #
    #

    #
    #         # Define Variables
    #         b_netw.var_inflow = Var(model.set_t, model.set_nodes, domain=NonNegativeReals)
    #         b_netw.var_outflow = Var(model.set_t, model.set_nodes, domain=NonNegativeReals)
    #
    #         # Define each arc
    #         def arc_rule(b_arc):
    #             b_arc.var_flow = Var(model.set_t, domain=NonNegativeReals)
    #             b_arc.var_losses = Var(model.set_t, domain=NonNegativeReals)
    #             def flowlosses_init(const,t):
    #                 return b_arc.var_losses[t] == b_arc.var_flow[t] * (1-eta)
    #             b_arc.const_flowlosses = Constraint(model.set_t, rule=flowlosses_init)
    #         b_netw.arc_block = Block(b_netw.set_arcs, rule=arc_rule)
    #
    #         # Define flow balance at each node
    #         def node_inflow_init(const, t, node):
    #             if b_netw.set_receives_from[node] == {}:
    #                 return b_netw.var_inflow[t, node] == 0
    #             else:
    #                 return b_netw.var_inflow[t, node] == \
    #                         sum(b_netw.arc_block[i, node].var_flow[t] for i in b_netw.set_receives_from[node]) - \
    #                         sum(b_netw.arc_block[i, node].var_losses[t] for i in b_netw.set_receives_from[node])
    #         b_netw.cons_node_inflow = Constraint(model.set_t, model.set_nodes, rule=node_inflow_init)
    #
    #         def node_outflow_init(const, t, node):
    #             if b_netw.set_sends_to[node] == {}:
    #                 return b_netw.var_outflow[t, node] == 0
    #             else:
    #                 return b_netw.var_outflow[t, node] == \
    #                         sum(b_netw.arc_block[node, i].var_flow[t] for i in b_netw.set_sends_to[node])
    #         b_netw.cons_node_outflow = Constraint(model.set_t, model.set_nodes, rule=node_outflow_init)
    #
    #         b_netw.var_size = Var(domain=NonNegativeReals, units=u.MW)
    #         b_netw.var_CAPEX = Var(units=u.EUR)
    #
    #
    #         b_netw = Block(b_netw_car.set_networks, rule=network_rule)
    #         b_netw.pprint()
    #         return b_netw
    #
    #     b_netw_car.network_block = Block(b_netw_car.set_networks, rule=network_rule)
    #     return b_netw_car
    # model.network_carrier_blocks = Block(model.set_network_carriers, rule=network_carrier_rule)
    #
    # return model
            #
            #
            # # Get bidirectional network flows:
            # # for from_node in connection:  # loops through table headers
            # #     for to_node in connection[from_node].index:
            # #         if connection.at[from_node, to_node] == connection.at[to_node, from_node]:
            # #             connection.at[from_node, to_node] = 2
            # #             connection.at[to_node, from_node] = 0
            #
            # # # Generate Sets for bidirectional arcs:
            # # def arcs_set_init(set):
            # #     for from_node in connection: # loops through table headers
            # #         for to_node in connection[from_node].index:
            # #             if connection.at[from_node, to_node] ==2:
            # #                 yield [from_node, to_node]
            # # model.set_bi_arcs = Set(initialize=arcs_set_init)
            #
            # # Generate Sets for unidirectional arcs:
            # def arcs_set_init(set):
            #     for from_node in connection: # loops through table headers
            #         for to_node in connection[from_node].index:
            #             if connection.at[from_node, to_node] == 1:
            #                 yield [from_node, to_node]
            # b_netw.set_arcs = Set(initialize=arcs_set_init)
            #
            # def nodesIn_init(m, node):
            #     for i, j in b_netw.set_arcs:
            #         if j == node:
            #             yield i
            # b_netw.set_nodesIn = Set(model.Nodes, initialize=nodesIn_init)


            #
            # # Create a set for each node, that specifies connection to other nodes (for inflow and outflow
            # def node_inflow_set(set, node):
            #     for i in connection:
            #         print(connection[i])
            #
            #     a=1
            # b_netw.node_inflow = Set(model.set_nodes, initialize=node_inflow_set)




    # def network_block_rule(b_netw):
    #
    # # region Get options from data
    # netw_data = data.technology_data[nodename][tec]
    # tec_type = netw_data['TechnologyPerf']['tec_type']
    # capex_model = netw_data['Economics']['CAPEX_model']
    # size_is_integer = netw_data['TechnologyPerf']['size_is_int']
    # # endregion
    #
    #
    # # define subsets of arcs
    # def initialize_arc_set(set):
    #
    # model.netw_blocks = Block(model.set_nodes, rule=node_block_rule)
    #
    #
    # a = 1