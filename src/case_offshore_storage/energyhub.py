from ..energyhub import EnergyHub
from pyomo.environ import *


class EnergyHubAdapted(EnergyHub):
    def change_network_size(self, network:str, size:float):
        """
        Changes the network size of an existing network to the specified size. Note: this changes
        the size of all arcs of the network to the respective size.
        Warning: This does not work with the gurobi persistent solver!

        :param str network: network name to change
        :param float size: size to change to
        """
        if self.configuration.solveroptions.solver == 'gurobi_persistent':
            raise Exception("This does not work with persistent solvers!")

        b_netw = self.model.network_block[network]

        for arc in b_netw.set_arcs:
            b_arc = b_netw.arc_block[arc]
            b_arc.var_flow.setub(size * b_netw.para_rated_capacity)  # Set upper bound

            b_arc.del_component('const_flow_size_high')

            def init_size_const_high(const, t):
                return b_arc.var_flow[t] <= size * b_netw.para_rated_capacity
            b_arc.const_flow_size_high = Constraint(self.model.set_t_full, rule=init_size_const_high)

            b_netw.del_component('const_cut_bidirectional')
            b_netw.del_component('const_cut_bidirectional_index')

            def init_cut_bidirectional(const, t, node_from, node_to):
                return b_netw.arc_block[node_from, node_to].var_flow[t] + b_netw.arc_block[node_to, node_from].var_flow[
                    t] \
                       <= size

            b_netw.const_cut_bidirectional = Constraint(self.model.set_t_full, b_netw.set_arcs_unique,
                                                        rule=init_cut_bidirectional)

    def change_generic_production(self, node_name:str, carrier:str, profile:list):
        """
        Changes the generic production profile at a node for a carrier
        Warning: This does not work with the gurobi persistent solver!

        :param str node_name: node name
        :param str carrier: carrier to change
        :param list profile: profile to use. Must have same number of time steps
        """
        if self.configuration.solveroptions.solver == 'gurobi_persistent':
            raise Exception("This does not work with persistent solvers!")

        self.data.read_production_profile(node_name, carrier, profile, 1)

        b_node = self.model.node_blocks[node_name]
        b_node.del_component('const_generic_production_index')
        b_node.del_component('const_generic_production')
        b_node.del_component('para_production_profile')
        b_node.del_component('para_production_profile_index')
        node_data = self.data.node_data[node_name]

        def init_production_profile(para, t, car):
            return node_data.data['production_profile'][car][t - 1]
        b_node.para_production_profile = Param(self.model.set_t_full, b_node.set_carriers, rule=init_production_profile, mutable=True)

        def init_generic_production(const, t, car):
            return b_node.para_production_profile[t, car] >= b_node.var_generic_production[t, car]
        b_node.const_generic_production = Constraint(self.model.set_t_full, b_node.set_carriers, rule=init_generic_production)


class EnergyhubCapexOptimization(EnergyHubAdapted):
    def __init__(self, data, configuration, technology_to_optimize:tuple, total_cost_limit:float):
        super().__init__(data, configuration)
        self.technology_to_optimize = technology_to_optimize
        self.total_cost_limit = total_cost_limit

    def _optimize(self, objective):
        self._delete_objective()
        self.model.del_component('const_cost_limit')

        self.model.const_cost_limit = Constraint(expr=self.model.var_total_cost <= self.total_cost_limit)

        def init_max_capex(obj):
            return self.model.node_blocks[self.technology_to_optimize[0]].tech_blocks_active[self.technology_to_optimize[1]].var_capex
        self.model.objective = Objective(rule=init_max_capex, sense=maximize)
        if self.configuration.solveroptions.solver == 'gurobi_persistent':
            self.solver.add_constraint(self.model.const_cost_limit)
            self.solver.set_objective(self.model.objective)

        self._call_solver()


class EnergyhubEmissionOptimization(EnergyHubAdapted):
    def __init__(self, data, configuration, technology_to_optimize:tuple, emission_limit:float):
        super().__init__(data, configuration)
        self.technology_to_optimize = technology_to_optimize
        self.emission_limit = emission_limit

    def _optimize(self, objective):
        self._delete_objective()
        self.model.del_component('const_cost_limit')

        self.model.const_cost_limit = Constraint(expr=self.model.var_emissions_net <= self.emission_limit)

        def init_max_capex(obj):
            return self.model.node_blocks[self.technology_to_optimize[0]].tech_blocks_active[self.technology_to_optimize[1]].var_size
        self.model.objective = Objective(rule=init_max_capex, sense=minimize)

        self._call_solver()
