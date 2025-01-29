import pandas as pd
import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
from warnings import warn

# from ..genericNetworks import fitting_classes as f
from ..network import Network
from ...utilities import link_full_resolution_to_clustered


class Electricity(Network):

    def __init__(self, netw_data: dict):
        """
        Constructor

        :param dict netw_data: network data
        """
        super().__init__(netw_data)

    def fit_network_performance(self):
        super(Electricity, self).fit_network_performance()

    def _define_emissions_arc(self, b_arc, b_netw):
        """
        Defines emission per arc equal to zero for Electricity networks

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """
        super(Electricity, self)._define_emissions_arc(b_arc, b_netw)

        def init_arc_emissions(const, t):
            return b_arc.var_emissions[t] == 0

        b_arc.const_arc_emissions = pyo.Constraint(self.set_t, rule=init_arc_emissions)

        return b_arc

    def _define_emission_constraints(self, b_netw):
        """
        Defines emissions from electricity network equal to zero

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        super(Electricity, self)._define_emission_constraints(b_netw)

        def init_netw_emissions(const, t, node):
            return b_netw.var_netw_emissions_pos[t, node] == 0

        b_netw.const_netw_emissions = pyo.Constraint(
            self.set_t, self.set_nodes, rule=init_netw_emissions
        )
        return b_netw

    def _calculate_energy_consumption(self):
        """
        Fits the performance parameters for a network, i.e. the consumption at each node.
        Overwritten in child class
        """
        # Get energy consumption at nodes form file

        self.energy_consumption = {}

    def _define_energyconsumption_arc(self, b_arc, b_netw):
        """
        Defines the energy consumption for an electricity arc

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :return: pyomo arc block
        """
        super(Electricity, self)._define_energyconsumption_arc(b_arc, b_netw)
        rated_capacity = self.input_parameters.rated_power

        b_arc.var_consumption_send = pyo.Var(
            self.set_t,
            b_netw.set_consumed_carriers,
            domain=pyo.NonNegativeReals,
            bounds=(
                b_netw.para_size_min * rated_capacity,
                b_arc.para_size_max * rated_capacity,
            ),
        )
        b_arc.var_consumption_receive = pyo.Var(
            self.set_t,
            b_netw.set_consumed_carriers,
            domain=pyo.NonNegativeReals,
            bounds=(
                b_netw.para_size_min * rated_capacity,
                b_arc.para_size_max * rated_capacity,
            ),
        )

        # Sending node
        def init_consumption_send(const, t, car):
            return b_arc.var_consumption_send[t, car] == 0

        b_arc.const_consumption_send = pyo.Constraint(
            self.set_t, b_netw.set_consumed_carriers, rule=init_consumption_send
        )

        # Receiving node
        def init_consumption_receive(const, t, car):
            return b_arc.var_consumption_receive[t, car] == 0

        b_arc.const_consumption_receive = pyo.Constraint(
            self.set_t, b_netw.set_consumed_carriers, rule=init_consumption_receive
        )

        return b_arc

    def _define_energyconsumption_total(self, b_netw):
        """
        Defines network consumption at each node for electricity network

        :param b_netw: pyomo network block
        :return: pyomo network block
        """

        super(Electricity, self)._define_energyconsumption_total(b_netw)

        def init_network_consumption(const, t, car, node):
            return b_netw.var_consumption[t, car, node] == 0

        b_netw.const_netw_consumption = pyo.Constraint(
            self.set_t,
            b_netw.set_consumed_carriers,
            self.set_nodes,
            rule=init_network_consumption,
        )

        return b_netw

    def write_results_netw_design(self, h5_group, model_block):
        """
        Function to report electricity network design

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super(Electricity, self).write_results_netw_design(h5_group, model_block)

        for arc_name in model_block.set_arcs:
            arc = model_block.arc_block[arc_name]
            str = "".join(arc_name)
            arc_group = h5_group[str]

            total_emissions = 0
            arc_group.create_dataset("total_emissions", data=total_emissions)
