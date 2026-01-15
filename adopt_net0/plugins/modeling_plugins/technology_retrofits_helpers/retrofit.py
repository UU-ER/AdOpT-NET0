"""
Retrofit base class

Retrofits implement the following variables of a technology that change the base variable respectively:

**Variable declarations:**

- ``var_delta_emissions``: Change in emissions :math:`var\_delta\_emissions_{t}`
- ``var_delta_capex``: Change in capex :math:`var\_delta\_capex`
- ``var_delta_opex_fix``: Change in capex :math:`var\_delta\_opex\_fix`
- ``var_delta_opex_var``: Change in capex :math:`var\_delta\_opex\_var`

Implemented in subclasses:

- ``var_delta_input``: Change in input :math:`var\_delta\_input_{t}`
- ``var_delta_output``: Change in output :math:`var\_delta\_output_{t}`

Also, constraints are implemented in the respective retrofit classes.
"""

import pyomo.environ as pyo

class Retrofit():

    def define_retrofit_variables(self, technology_constructor, modelhub, b_retrofit):
        """
        Defines retrofit variables

        :param technology_constructor: Technology constructor
        :param b_retrofit: pyomo block for retrofit
        :return:
        """
        b_retrofit.var_delta_emissions = pyo.Var(
            technology_constructor.set_t_global,
            within=pyo.Reals,
        )
        b_retrofit.var_delta_capex = pyo.Var(
            within=pyo.Reals,
        )
        b_retrofit.var_delta_opex_fix = pyo.Var(
            within=pyo.Reals,
        )
        b_retrofit.var_delta_opex_var = pyo.Var(
            within=pyo.Reals,
        )

    def define_retrofit_constraints(self, technology_constructor, modelhub, b_retrofit, b_tec):
        """
        Defines retrofit constraints

        Implemented in subclasses.
        """
        pass

    def define_expression_entering_energybalance(self, technology_constructor, b_retrofit):
        """
        Defines expressions entering energy balance

        Implemented in subclasses.
        """
        pass

    def define_expression_entering_costbalance_capex(self, technology_constructor, b_retrofit):
        """
        Defines expressions entering cost balance capex

        Implemented in subclasses.
        """
        pass

    def define_expression_entering_emissionbalance(self, technology_constructor, b_retrofit):
        """
        Defines expressions entering emission balance

        Implemented in subclasses.
        """
        pass

    def write_results_design(self, model_block, h5_group):
        """
        Writes retrofit design to h5 file

        Implemented in subclasses
        .
        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        h5_group.create_dataset(
            "capex",
            data=[
                model_block.var_delta_capex.value
            ],
        )
        h5_group.create_dataset(
            "opex_fix",
            data=[
                model_block.var_delta_opex_fix.value
            ],
        )
        h5_group.create_dataset(
            "opex_var",
            data=[
                model_block.var_delta_opex_var.value
            ],
        )

    def write_results_operation(self, technology_constructor, model_block, h5_group):
        """
        Writes retrofit operation to h5 file

        Implemented in subclasses
        .
        :param technology_constructor: Technology constructor
        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        h5_group.create_dataset(
            "delta_emissions",
            data=[
                model_block.var_delta_emissions[t].value for t in technology_constructor.set_t_global
            ],
        )
