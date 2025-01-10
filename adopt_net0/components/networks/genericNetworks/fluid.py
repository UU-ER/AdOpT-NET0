import pandas as pd
import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
from warnings import warn

# from ..genericNetworks import fitting_classes as f
from ..network import Network
from ...utilities import link_full_resolution_to_clustered


class Fluid(Network):

    def fit_network_performance_performance(self):
        super(Fluid, self).fit_network_performance(self)
