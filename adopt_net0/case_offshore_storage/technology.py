import pandas as pd
from pyomo.gdp import *
from warnings import warn
import pyomo.environ as pyo

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

        coeff_ti = self.processed_coeff.time_independent
        charge_rate = coeff_ti["charge_rate"]
        discharge_rate = coeff_ti["discharge_rate"]

        b_tec.para_size_min = pyo.Param(
            domain=pyo.NonNegativeReals, initialize=coeff_ti["size_min"], mutable=True
        )
        b_tec.para_size_max = pyo.Param(
            domain=pyo.NonNegativeReals, initialize=coeff_ti["size_max"], mutable=True
        )

        b_tec.var_size = pyo.Param(within=pyo.NonNegativeReals, initialize=self.size)

        b_tec.var_capacity_charge = pyo.Var(
            domain=pyo.NonNegativeReals, bounds=(0, b_tec.para_size_max * charge_rate)
        )
        b_tec.var_capacity_discharge = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(0, b_tec.para_size_max * discharge_rate),
        )
        return b_tec

    def _define_capex_variables(self, b_tec, data):
        config = data["config"]

        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(discount_rate, economics.lifetime, fraction_of_year_modelled)

        b_tec.var_unit_capex_annual = pyo.Var(within=pyo.Reals, bounds=(0, 1e8))
        b_tec.var_unit_capex = pyo.Var(within=pyo.Reals, bounds=(0, 1e8))
        b_tec.var_capex = pyo.Var(within=pyo.Reals, bounds=(0, 1e8))
        b_tec.var_capex_aux = pyo.Var(within=pyo.Reals, bounds=(0, 1e8))

        return b_tec

    def _define_capex_constraints(self, b_tec, data):

        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )

        b_tec.const_capex_aux = pyo.Constraint(expr=b_tec.var_size * b_tec.var_unit_capex_annual == b_tec.var_capex_aux)
        b_tec.const_unit_capex = pyo.Constraint(
            expr=b_tec.var_unit_capex_annual == b_tec.var_unit_capex * annualization_factor)

        b_tec.const_capex = pyo.Constraint(expr=b_tec.var_capex == b_tec.var_capex_aux)

        return b_tec
