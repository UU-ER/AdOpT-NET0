import pandas as pd
import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
from warnings import warn

from ..network import Network
from ...utilities import get_attribute_from_dict


class Simple(Network):
    """
    Network with no specific carrier

    This network type resembles a network in which the carrier is not specified.

    - Constraint definitions:

    **Network constraint declarations**
    This part calculates variables for all respective nodes.

    -  netw_emissions equal to zero
    -  netw_consumption equal to zero

    """

    def __init__(self, netw_data: dict):
        """
        Constructor

        :param dict netw_data: network data
        """
        super().__init__(netw_data)

        # we keep simple networks are always bi-directional
        self.bidirectional_network = 1
        self.bidirectional_network_precise = get_attribute_from_dict(
            netw_data["Performance"], "bidirectional_network_precise", 1
        )

    def fit_network_performance(self):
        super(Simple, self).fit_network_performance()

    def _define_emission_constraints(self, b_netw):
        """
        Defines emissions from simple network equal to zero

        :param b_netw: pyomo network block
        :return: pyomo network block
        """
        super(Simple, self)._define_emission_constraints(b_netw)

        def init_netw_emissions(const, t, node):
            return b_netw.var_netw_emissions_pos[t, node] == 0

        b_netw.const_netw_emissions = pyo.Constraint(
            self.set_t, self.set_nodes, rule=init_netw_emissions
        )
        return b_netw

    def _define_energyconsumption_total(self, b_netw):
        """
        Defines network consumption at each node for simple network

        :param b_netw: pyomo network block
        :return: pyomo network block
        """

        super(Simple, self)._define_energyconsumption_total(b_netw)

        def init_network_consumption(const, t, car, node):
            return b_netw.var_consumption[t, car, node] == 0

        b_netw.const_netw_consumption = pyo.Constraint(
            self.set_t,
            b_netw.set_consumed_carriers,
            self.set_nodes,
            rule=init_network_consumption,
        )

        return b_netw
