import pandas as pd
from pyomo.gdp import *
from warnings import warn
from pyomo.environ import *

from ..components.technologies.technology import Technology
from ..components.utilities import annualize, set_discount_rate, link_full_resolution_to_clustered, determine_variable_scaling, determine_constraint_scaling

class TechnologyCapexOptimization(Technology):
    """
    Adapts the technology class to fix the size and have capex as a variable
    """
    def __init__(self, tec_data):
        """
        Initializes technology class from technology name

        The technology name needs to correspond to the name of a JSON file in ./data/technology_data.

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)
        self.size = tec_data['size']

    def _define_size(self, b_tec):
        if self.existing:
            size_max = self.size_initial
        else:
            size_max = self.size_max

        if self.size_is_int:
            size_domain = NonNegativeIntegers
        else:
            size_domain = NonNegativeReals

        b_tec.para_size_min = Param(domain=NonNegativeReals, initialize=self.size_min, mutable=True)
        b_tec.para_size_max = Param(domain=NonNegativeReals, initialize=size_max, mutable=True)

        b_tec.var_size = Param(within=NonNegativeReals, initialize=self.size)
        return b_tec

    def _define_capex(self, b_tec, energyhub):
        configuration = energyhub.configuration

        economics = self.economics
        discount_rate = set_discount_rate(configuration, economics)
        fraction_of_year_modelled = energyhub.topology.fraction_of_year_modelled
        annualization_factor = annualize(discount_rate, economics.lifetime, fraction_of_year_modelled)

        b_tec.var_unit_capex_annual = Var(within=Reals, bounds=(0, 1e10))
        b_tec.var_unit_capex = Var(within=Reals, bounds=(0, 1e10), initialize=0)
        b_tec.var_capex = Var(within=Reals, bounds=(0, 1e10))
        b_tec.var_capex_aux = Var(within=Reals, bounds=(0, 1e10))

        b_tec.const_capex_aux = Constraint(expr=b_tec.var_size * b_tec.var_unit_capex_annual == b_tec.var_capex_aux)
        b_tec.const_unit_capex = Constraint(
            expr=b_tec.var_unit_capex_annual == b_tec.var_unit_capex * annualization_factor)

        b_tec.const_capex = Constraint(expr=b_tec.var_capex == b_tec.var_capex_aux)

        return b_tec
