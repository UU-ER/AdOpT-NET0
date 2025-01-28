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
