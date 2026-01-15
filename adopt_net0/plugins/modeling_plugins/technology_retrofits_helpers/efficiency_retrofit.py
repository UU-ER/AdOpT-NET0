"""
Efficiency improvement retrofit.

Efficiency improvements adds additional outputs a fraction of the output of the base technology.
The efficiency improvement is either fully installed or not at all (resulting in a disjunctive
constraint). The efficiency improvement has an additional CAPEX and variable OPEX, but no
additional fixed OPEX costs. Emissions are unchanged by the efficiency improvement (they are
implicitly included in the change in output).

The following settings can be added to a technology json (example for a technolgoy with heat and electric output):

.. code-block:: json

    "settings_technology_retrofits": {
        "enable_efficiency_retrofit": True,
        "efficiency_retrofit": {
            "unit_capex": 10,
            "coefficients":[
                {
                    "output_carrier": "electricity",
                    "coefficient": 0.0,
                    "opex_variable": 1,
                },
                {
                    "output_carrier": "heat",
                    "coefficient": 0.0,
                    "opex_variable": 1,
                },
        ]
        }
    }

**Variable declarations:**

- ``var_delta_output``: Change in output :math:`var\_delta\_output_{t}`
- ``var_delta_emissions``: Change in emissions :math:`var\_delta\_emissions_{t}`
- ``var_delta_capex``: Change in capex :math:`var\_delta\_capex`
- ``var_delta_opex_fix``: Change in capex :math:`var\_delta\_opex\_fix`
- ``var_delta_opex_var``: Change in capex :math:`var\_delta\_opex\_var`

**Constraint declarations:**

**Constraint declarations:**

- Install / not-install (disjunctive) constraints:

  - If retrofit is not installed:

    .. math::
       var\_delta\_output_{t,car} = 0

    .. math::
       var\_delta\_capex = 0

  - If retrofit is installed:

    .. math::
       var\_delta\_output_{t,car} = coefficient_{car} \\cdot var\_output^{tec}_{t,car}

    .. math::
       var\_delta\_capex = annualization\_factor \\cdot unit\_capex

- Variable OPEX (aggregated over timesteps, defined in \_define\_opex\_variable):

  .. math::
     var\_delta\_opex\_var = \\sum_{car}\\sum_{t} \Bigl( var\_delta\_output_{t,car} opex\_variable_{car} \\Bigr)

- Emissions (no change):

  .. math::
     var\_delta\_emissions_{t} = 0

"""
import pyomo.environ as pyo
import pyomo.gdp as gdp

from adopt_net0.plugins.modeling_plugins.technology_retrofits_helpers.retrofit import Retrofit
from adopt_net0.core.components.utilities import annualize, set_discount_rate

import logging

log = logging.getLogger(__name__)


class EfficiencyRetrofit(Retrofit):

    def _define_bounds(self, technology_constructor, modelhub, b_retrofit):
        """
        Define upper and lower bounds for the efficiency retrofit variables.

        This method sets numerical bounds for:

        - ``b_retrofit.var_delta_capex``: bounded between 0 and
          ``unit_capex`` * annualization_factor (annualized unit CAPEX)
        - ``b_retrofit.var_delta_output[t, car]``: bounded between 0 and
          ``max_output_base_tec * efficiency_improvement``

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """
        parameters = technology_constructor.data["settings_technology_retrofits"]["efficiency_retrofit"]

        economics = technology_constructor.economics
        config = modelhub.data["config"]

        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = modelhub.data["topology"]["temporal_information"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        b_retrofit.var_delta_capex.setlb(0)
        b_retrofit.var_delta_capex.setub(parameters["unit_capex"]*annualization_factor)
        b_retrofit.var_delta_opex_fix.setlb(0)
        b_retrofit.var_delta_opex_fix.setub(0)
        for t in technology_constructor.set_t_performance:
            for car in b_retrofit.set_output_carriers:
                coef = next(
                    (item["coefficient"]
                     for item in parameters["coefficients"]
                     if item["output_carrier"] == car),
                    None
                )
                max_output_base_tec = technology_constructor.output[t, car].ub
                b_retrofit.var_delta_output[t, car].setlb(0)
                b_retrofit.var_delta_output[t, car].setub(max_output_base_tec * coef)


    def define_retrofit_variables(self, technology_constructor, modelhub, b_retrofit):
        """
        Declare Pyomo Sets and Variables required by the efficiency retrofit.

        After declaration the method calls ``_define_bounds`` to apply numeric
        bounds based on preprocessed parameters and economics.

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """
        super().define_retrofit_variables(technology_constructor, modelhub, b_retrofit)

        coefficients = technology_constructor.data["settings_technology_retrofits"]["efficiency_retrofit"]["coefficients"]
        output_carriers = [param["output_carrier"] for param in coefficients]
        if len(output_carriers) != len(set(output_carriers)):
            raise ValueError("Efficiency improvement coefficients must have unique output carriers.")

        b_retrofit.set_output_carriers = pyo.Set(initialize=output_carriers)
        b_retrofit.set_efficiency_retrofits = pyo.RangeSet(len(coefficients))

        b_retrofit.var_delta_output = pyo.Var(
    technology_constructor.set_t_performance,
            b_retrofit.set_output_carriers,
            within=pyo.Reals,
        )

        self._define_bounds(technology_constructor, modelhub, b_retrofit)

    def define_retrofit_constraints(self, technology_constructor, modelhub, b_retrofit, b_tec):
        """
        Define constraints that couple the retrofit with the parent
        technology model block (``b_tec``).

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        :param b_tec: Pyomo block
        """
        super().define_retrofit_constraints(technology_constructor, modelhub, b_retrofit, b_tec)
        technology_constructor.big_m_transformation_required = 1

        self._define_opex_fix(b_retrofit)
        self._define_opex_variable(technology_constructor, modelhub, b_retrofit)
        self._define_performance(technology_constructor, modelhub, b_retrofit, b_tec)
        self._define_emissions(technology_constructor, b_retrofit, b_tec)


    def _define_opex_fix(self, b_retrofit):
        """
        Sets fixed OPEX to zero.

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """

        # Opex fix is always zero
        b_retrofit.const_opex_fix = pyo.Constraint(expr=b_retrofit.var_delta_opex_fix == 0)

    def _define_opex_variable(self, technology_constructor, modelhub, b_retrofit):
        """
        Define variable OPEX constraint for each output.

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """
        parameters = technology_constructor.data["settings_technology_retrofits"]["efficiency_retrofit"]

        hour_factors = modelhub.data["aggregation_info"]["hour_factors"]
        nr_timesteps_averaged = modelhub.data["aggregation_info"]["nr_timesteps_averaged"]

        # Opex variable
        def init_opex_variable_on(const):

            return b_retrofit.var_delta_opex_var == \
                sum(
                    sum(
                        b_retrofit.var_delta_output[
                            t, parameters["coefficients"][improvement_idx - 1]["output_carrier"]]
                        * nr_timesteps_averaged
                        * hour_factors[t - 1]
                        * parameters["coefficients"][improvement_idx - 1]["opex_variable"]
                        for t in technology_constructor.set_t_performance
                    )
                    for improvement_idx in b_retrofit.set_efficiency_retrofits
                )

        b_retrofit.const_opex_var_on = pyo.Constraint(rule=init_opex_variable_on)

    def _define_performance(self, technology_constructor, modelhub, b_retrofit, b_tec):
        """
        Define performance constraints linking the output of the paretn technology to the additional
        output possible with efficiency improvement..

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """
        parameters = technology_constructor.data["settings_technology_retrofits"]["efficiency_retrofit"]

        economics = technology_constructor.economics
        config = modelhub.data["config"]

        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = modelhub.data["topology"]["temporal_information"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        s_indicators = range(0, 2)

        def init_input_output(dis, ind):
            if ind == 0:  # retrofit off

                def init_output_off(const, t, car):
                    return b_retrofit.var_delta_output[t, car] == 0
                dis.const_input_output_off = pyo.Constraint(
            technology_constructor.set_t_performance, b_retrofit.set_output_carriers, rule=init_output_off
                )

                dis.const_capex_off = pyo.Constraint(expr=b_retrofit.var_delta_capex == 0)

            else:  # retrofit on

                def init_output_on(const, t, improvement_idx):
                    parameter = parameters["coefficients"][improvement_idx - 1]
                    return (b_retrofit.var_delta_output[t, parameter["output_carrier"]] ==
                            parameter["coefficient"] * b_tec.var_output[t, parameter["output_carrier"]])

                dis.const_input_output_on = pyo.Constraint(
                    technology_constructor.set_t_performance, b_retrofit.set_efficiency_retrofits,
                    rule=init_output_on
                )

                dis.const_capex_on = pyo.Constraint(expr=b_retrofit.var_delta_capex == parameters["unit_capex"] * annualization_factor)

        b_retrofit.dis_input_output = gdp.Disjunct(s_indicators, rule=init_input_output)

        # Bind disjuncts
        def bind_disjunctions(dis):
            return [b_retrofit.dis_input_output[i] for i in s_indicators]

        b_retrofit.disjunction_input_output = gdp.Disjunction(rule=bind_disjunctions)

    def _define_emissions(self, technology_constructor, b_retrofit, b_tec):
        """
        Emissions are unchanged by efficiency retrofit (they are implicitly included in the
        change in output).

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """
        def init_tec_delta_emissions(const, t):
            return b_retrofit.var_delta_emissions[t] == 0

        b_retrofit.const_delta_tec_emissions = pyo.Constraint(
            technology_constructor.set_t_global, rule=init_tec_delta_emissions
        )

    def write_results_operation(self, technology_constructor, model_block, h5_group):
        """
        Writes retrofit operation to h5 file

        :param technology_constructor: Technology constructor
        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super().write_results_operation(technology_constructor, model_block, h5_group)

        for car in model_block.set_output_carriers:
            h5_group.create_dataset(
                f"{car}_output",
                data=[model_block.var_delta_output[t, car].value for t in technology_constructor.set_t_global],
            )