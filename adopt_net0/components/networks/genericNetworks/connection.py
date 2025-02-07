import pandas as pd
import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
from warnings import warn

# from ..genericNetworks import fitting_classes as f
from ..network import Network
from ...utilities import link_full_resolution_to_clustered


class Connection(Network):
    """
    Network with no specific carrier

    This network type resembles a network in which the carrier is not specified
    """

    def __init__(self, netw_data: dict):
        """
        Constructor

        :param dict netw_data: network data
        """
        super().__init__(netw_data)

        # we keep simple networks are always bi-directional
        self.component_options.bidirectional_network = 1
