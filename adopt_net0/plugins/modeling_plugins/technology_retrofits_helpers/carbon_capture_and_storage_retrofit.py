"""
Carbon capture and storage (CCS) retrofit

This module implements a CCS retrofit as a technology retrofit plugin. The
CCS retrofit augments an existing technology by capturing a fraction of its
CO2 emissions and modelling the additional size-dependent CAPEX, OPEX,
energy requirements and captured CO2. Captured CO2 is treated as another carrier.

Parameter and behaviour summary:

- capture_rate: fraction of CO2 emissions (or CO2 per input/output depending
  on technology) that can be captured by CCS.
- input_ratios: for each input carrier, ratio to convert captured CO2 to required
  input amount (e.g. electricity per tCO2 captured).
- size bounds: min/max sizing converted to t_CO2/h in preprocessing.
- CAPEX/OPEX: unit_capex, fix_capex, opex_variable and opex_fixed are handled
  and annualized using the model's discount rate and lifetime settings.

**Set declarations:**

- ``set_output_carriers``: output carriers of the CCS retrofit (e.g. CO2captured)
- ``set_input_carriers``: input carriers of the CCS retrofit (e.g. electricity,
  heat)

**Variable declarations:**
Additional variables (compare to parent Retrofit class):

- ``var_delta_input``: Change in input :math:`var\_delta\_input_{t}`
- ``var_delta_output``: Change in output :math:`var\_delta\_output_{t}`
- ``var_size``: total CCS capture capacity (t_CO2/h) :math:`var\_size`
  captured CO2) :math:`var\_delta\_emissions_{t}`

**Constraint declarations**



- Input carriers are given by:

.. math::
    input_{car, t} <= inputRatio_{car} * output_{t}/captureRate

- CO2 captured output is constrained by (depending on if emissions are based on input or output):

.. math::
    output_{t} <= input^{tec}_{t} / output^{tec}_{t} * emissionFactor * captureRate

- Emissions of the ccs retrofit are given by:

.. math::
    emissions_{t} = - output_{t}

- CAPEX is given by

.. math::
    CAPEX = Size * UnitCost + FixCost

- Fixed OPEX: defined as a fraction of annual CAPEX:

.. math::
    OPEXfix = CAPEX * opex

"""
import pyomo.environ as pyo
import pyomo.gdp as gdp

from adopt_net0.plugins.modeling_plugins.technology_retrofits_helpers.retrofit import Retrofit
from adopt_net0.core.components.utilities import annualize, set_discount_rate

import logging

log = logging.getLogger(__name__)


class CcsRetrofit(Retrofit):
    # TODO: Make sure this works with aggregation
    def preprocess_parameters(self, technology_constructor):
        """
        Preprocess CCS parameters from the technology JSON and store derived
        values back into the technology data structure.

        This method calculates parameters and performs unit conversion
        and adds the recaculated values to ``technology_constructor.data["settings_technology_retrofits"]["ccs_retrofit"]``.
        The following actions are performed:

        - Convert molar flow units of CO2 to mass flow (t/h CO2_out) using
          the provided CO2 concentration and the molar mass of CO2.
        - Recalculate ``Economics.unit_capex`` to be expressed in EUR per
          (t_CO2_out / h) using capex kappa/lambda/zeta inputs and the
          conversion factors derived from CO2 concentration.
        - Populate ``size_min_t_h`` and ``size_max_t_h`` with size bounds in
          t_CO2/h based on the configured capture rate and original size
          settings.
        - Compute ``Performance.input_ratios`` mapping each CCS input carrier
          (e.g. electricity, heat) to the required input per tCO2 captured.

        :param technology_constructor: Technology constructor
        """
        parameters = technology_constructor.data["settings_technology_retrofits"]["ccs_retrofit"]

        co2_concentration = parameters["co2_concentration"]
        molar_mass_CO2 = 44.01
        # convert kmol/s of fluegas to ton/h of molar_mass_CO2 = 44.01
        convert2t_per_h = molar_mass_CO2 * co2_concentration * 3.6
        capture_rate = parameters["Performance"]["capture_rate"]
        # Recalculate unit_capex in EUR/(t_CO2out/h)
        parameters["Economics"]["unit_capex"] = (
                                                      (
                                                              parameters["Economics"]["capex_kappa"] / convert2t_per_h
                                                              + parameters["Economics"]["capex_lambda"]
                                                      )
                                                      * co2_concentration
                                              ) / convert2t_per_h

        parameters["Economics"]["fix_capex"] = parameters["Economics"]["capex_zeta"]

        # Recalculate min/max size to have it in t/hCO2_out
        parameters["size_min_t_h"] = parameters["size_min"] * co2_concentration * capture_rate
        parameters["size_max_t_h"] = parameters["size_max"] * co2_concentration * capture_rate

        # Calculate input ratios
        if "MEA" in parameters["ccs_type"]:
            input_ratios = {}
            for car in parameters["Performance"]["input_carrier"]:
                input_ratios[car] = (
                                            parameters["Performance"]["eta"][car]
                                            + parameters["Performance"]["omega"][car] * co2_concentration
                                    ) / (co2_concentration * molar_mass_CO2 * 3.6)
            parameters["Performance"]["input_ratios"] = input_ratios
        else:
            raise Exception(
                "Only CCS type MEA is modelled so far. ccs_type in the json file of the "
                "technology must include MEA"
            )

    def _define_bounds(self, technology_constructor, modelhub, b_retrofit):
        """
        Define upper and lower bounds for CCS retrofit variables.

        This method sets numerical bounds for:

        - b_retrofit.var_size: upper bound by ``size_max_t_h``
        - b_retrofit.var_delta_output[t, car]: 0..capture_rate * size_max_t_h
        - b_retrofit.var_delta_input[t, car]: 0..input_ratio[car] * size_max_t_h
        - b_retrofit.var_delta_capex: 0..unit_capex * size_max_t_h * annualization + fix_capex_annual

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """
        parameters = technology_constructor.data["settings_technology_retrofits"]["ccs_retrofit"]

        economics = technology_constructor.economics
        config = modelhub.data["config"]

        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = modelhub.data["topology"]["temporal_information"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )
        # Size bounds
        b_retrofit.var_size.setub(parameters["size_max_t_h"])

        # Input/output bounds
        for t in technology_constructor.set_t_global:

            # Output bounds
            for car in b_retrofit.set_output_carriers:
                b_retrofit.var_delta_output[t, car].setlb(0)
                b_retrofit.var_delta_output[t, car].setub(parameters["Performance"]["capture_rate"] * parameters["size_max_t_h"])

            # Input bounds
            for car in b_retrofit.set_input_carriers:
                b_retrofit.var_delta_input[t, car].setlb(0)
                b_retrofit.var_delta_input[t, car].setub(
                    parameters["Performance"]["input_ratios"][car] * parameters["size_max_t_h"]
                )

        # CAPEX
        b_retrofit.var_delta_capex.setlb(0)
        b_retrofit.var_delta_capex.setub(parameters["Economics"]["unit_capex"] * parameters["size_max_t_h"] * annualization_factor + annualization_factor * parameters["Economics"]["fix_capex"])

    def define_retrofit_variables(self, technology_constructor, modelhub, b_retrofit):
        """
        Declare Pyomo Sets and Variables required by the CCS retrofit.

        After declaration the method calls ``_define_bounds`` to apply numeric
        bounds based on preprocessed parameters and economics.

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """
        super().define_retrofit_variables(technology_constructor, modelhub, b_retrofit)

        parameters = technology_constructor.data["settings_technology_retrofits"]["ccs_retrofit"]

        b_retrofit.var_size = pyo.Var(
            within=pyo.NonNegativeReals,
        )

        output_carriers = [car for car in parameters["Performance"]["output_carrier"]]
        b_retrofit.set_output_carriers = pyo.Set(initialize=output_carriers)
        b_retrofit.var_delta_output = pyo.Var(
    technology_constructor.set_t_global,
            b_retrofit.set_output_carriers,
            within=pyo.Reals,
        )

        input_carrier = [car for car in parameters["Performance"]["input_carrier"]]
        b_retrofit.set_input_carriers = pyo.Set(initialize=input_carrier)
        b_retrofit.var_delta_input = pyo.Var(
            technology_constructor.set_t_global,
            b_retrofit.set_input_carriers,
            within=pyo.Reals,
        )

        self._define_bounds(technology_constructor, modelhub, b_retrofit)

    def define_retrofit_constraints(self, technology_constructor, modelhub, b_retrofit, b_tec):
        """
        Define constraints that couple the CCS retrofit with the parent
        technology model block (``b_tec``).

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        :param b_tec: Pyomo block
        """
        super().define_retrofit_constraints(technology_constructor, modelhub, b_retrofit, b_tec)
        technology_constructor.big_m_transformation_required = 1

        self._define_opex_fix(technology_constructor, modelhub, b_retrofit)
        self._define_opex_variable(technology_constructor, modelhub, b_retrofit)
        self._define_installment(technology_constructor, modelhub, b_retrofit)
        self._define_performance(technology_constructor, b_retrofit, b_tec)
        self._define_emissions(technology_constructor, b_retrofit, b_tec)


    def _define_opex_fix(self, technology_constructor, modelhub, b_retrofit):
        """
        Define fixed OPEX constraint for the CCS retrofit.

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """
        economics = technology_constructor.economics
        config = modelhub.data["config"]

        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = modelhub.data["topology"]["temporal_information"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        b_retrofit.const_opex_fixed_ccs = pyo.Constraint(
            expr=(b_retrofit.var_delta_capex / annualization_factor)
                 * economics["opex_fixed"]
                 == b_retrofit.var_delta_opex_fix
        )

    def _define_opex_variable(self, technology_constructor, modelhub, b_retrofit):
        """
        Define variable OPEX constraint for CCS operation in terms of captured CO2.

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """
        parameters = technology_constructor.data["settings_technology_retrofits"]["ccs_retrofit"]

        hour_factors = modelhub.data["aggregation_info"]["hour_factors"]
        nr_timesteps_averaged = modelhub.data["aggregation_info"]["nr_timesteps_averaged"]

        # Opex variable
        def init_opex_variable(const):

            return b_retrofit.var_delta_opex_var == \
                sum(
                    b_retrofit.var_delta_output[
                        t, b_retrofit.set_output_carriers.at(1)]
                    * nr_timesteps_averaged
                    * hour_factors[t - 1]
                    * parameters["Economics"]["opex_variable"]
                    for t in technology_constructor.set_t_performance
                )


        b_retrofit.const_opex_var = pyo.Constraint(rule=init_opex_variable)

    def _define_installment(self, technology_constructor, modelhub, b_retrofit):
        """
        Define install / not-install disjuncts, sizing and capex constraints.

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """
        parameters = technology_constructor.data["settings_technology_retrofits"]["ccs_retrofit"]
        economics = technology_constructor.economics
        config = modelhub.data["config"]

        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = modelhub.data["topology"]["temporal_information"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )

        s_indicators = range(0, 2)

        def init_installed(dis, ind):
            if ind == 0:  # retrofit off

                dis.const_capex_off = pyo.Constraint(expr=b_retrofit.var_delta_capex == 0)
                dis.const_size_off = pyo.Constraint(expr=b_retrofit.var_size == 0)

            else:  # retrofit on

                dis.const_capex_on = pyo.Constraint(
                    expr=
                        annualization_factor * (
                                parameters["Economics"]["unit_capex"] * b_retrofit.var_size +
                                parameters["Economics"]["fix_capex"])
                         == b_retrofit.var_delta_capex
                )
                dis.const_installed_sizelim_min = pyo.Constraint(
                    expr=b_retrofit.var_size >= parameters["size_min_t_h"]
                )
                dis.const_installed_sizelim_max = pyo.Constraint(
                    expr=b_retrofit.var_size <= parameters["size_max_t_h"]
                )

        b_retrofit.dis_installed = gdp.Disjunct(s_indicators, rule=init_installed)

        # Bind disjuncts
        def bind_disjunctions(dis):
            return [b_retrofit.dis_installed[i] for i in s_indicators]

        b_retrofit.disjunction_installed = gdp.Disjunction(rule=bind_disjunctions)

    def _define_performance(self, technology_constructor, b_retrofit, b_tec):
        """
        Define performance constraints linking CCS captured CO2 to the parent
        technology and the retrofit inputs.

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """
        parameters = technology_constructor.data["settings_technology_retrofits"]["ccs_retrofit"]
        capture_rate = parameters["Performance"]["capture_rate"]

        def init_input_output_ccs(const, t):
            if technology_constructor.emissions_based_on == "output":
                return (
                    b_retrofit.var_delta_output[t, "CO2captured"]
                    <= capture_rate
                    * b_tec.para_tec_emissionfactor
                    * b_tec.var_delta_output[t, technology_constructor.main_output_carrier]
                )
            else:
                return (
                    b_retrofit.var_delta_output[t, "CO2captured"]
                    <= capture_rate
                    * b_tec.para_tec_emissionfactor
                    * b_tec.var_input[t, technology_constructor.main_input_carrier]
                )

        b_retrofit.const_input_output_ccs = pyo.Constraint(
            technology_constructor.set_t_global, rule=init_input_output_ccs
        )

        def init_size_output_ccs(const, t):
            return b_retrofit.var_delta_output[t, "CO2captured"] <= b_retrofit.var_size

        b_retrofit.const_size_output_ccs = pyo.Constraint(
            technology_constructor.set_t_global, rule=init_size_output_ccs
        )

        # Electricity and heat demand CCS
        def init_input_ccs(const, t, car):
            return (
                b_retrofit.var_delta_input[t, car]
                == parameters["Performance"]["input_ratios"][car]
                * b_retrofit.var_delta_output[t, "CO2captured"]
                / capture_rate
            )

        b_retrofit.const_input_el = pyo.Constraint(
            technology_constructor.set_t_global, b_retrofit.set_input_carriers, rule=init_input_ccs
        )


    def _define_emissions(self, technology_constructor, b_retrofit, b_tec):
        """
        Define emission accounting for the CCS retrofit. Note that delta_emissions
        are negative since the retrofit captures CO2.

        :param technology_constructor: Technology constructor
        :param modelhub: ModelHub
        :param b_retrofit: Pyomo block
        """
        def init_tec_emissions(const, t):
            return (
                - b_retrofit.var_delta_output[t, "CO2captured"]
                == b_retrofit.var_delta_emissions[t]
            )

        b_retrofit.const_ccs_emissions = pyo.Constraint(
            technology_constructor.set_t_global, rule=init_tec_emissions
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

        for car in model_block.set_input_carriers:
            h5_group.create_dataset(
                f"{car}_input",
                data=[model_block.var_delta_input[t, car].value for t in technology_constructor.set_t_global],
            )

    def write_results_design(self, model_block, h5_group):
        """
        Writes retrofit design to h5 file

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super().write_results_design(model_block, h5_group)

        h5_group.create_dataset(
            "size",
            data=[
                model_block.var_size.value
            ],
        )
