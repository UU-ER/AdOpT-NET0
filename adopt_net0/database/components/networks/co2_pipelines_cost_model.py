import numpy as np
import pandas as pd
from statsmodels import api as sm

from .utilities import *
from adopt_net0.database.utilities import convert_currency
from adopt_net0.database.data_component import DataComponent_CostModel


class CO2_Pipeline_CostModel(DataComponent_CostModel):
    """
    CO2 Pipeline

    Calculates CO2 transport costs and compression energy.

    Possible options are:

    If source = "Oeuvray"

    - cost and energy consumption model is based on Oeuvray, P., Burger, J., Roussanaly, S., Mazzotti, M., Becattini, V. (2024):
      Multi-criteria assessment of inland and offshore carbon dioxide transport options, Journal of Cleaner Production, https://doi.org/10.1016/j.jclepro.2024.140781.
    - length_km: Length of pipeline in km
    - timeframe: determines which steel grades are available, can be 'short-term', 'mid-term', or 'long-term'
    - massflow_min_kg_per_s: minimal mass flow rate of CO2 in kg/s to evaluate for costs
    - massflow_max_kg_per_s: maximal mass flow rate of CO2 in kg/s to evaluate for costs
    - massflow_evaluation_points: for how many points should costs be calculated between massflow_min_kg_per_s, massflow_max_kg_per_s (includes min and max)
    - terrain: 'Offshore' or 'Onshore', determines right of way cost and if recompression is possible (not possible for "Offshore")
    - electricity_price_eur_per_mw: used to minimize levelized cost (EUR/MWh)
    - operating_hours_per_a: number of operating hours per year
    - p_inlet_bar: inlet pressure in bar (beginning of pipeline)
    - p_outlet_bar: outlet pressure in bar (end of pipeline)

    Financial indicators are:

    - gamma1, gamma2, gamma3, gamma4 in [currency] (equivalent to the cost parameters of a network)
    - fixed opex as fraction of up-front capex
    - variable opex in [currency]/ton
    - lifetime in years
    - levelized_cost in [currency]/t

    Technical indicators are:

    - energyconsumption in MWh/t compressed
    """

    def __init__(self, tec_name):
        super().__init__(tec_name)
        # Default options:
        self.default_options["source"] = "Oeuvray"
        self.default_options["timeframe"] = "mid-term"
        self.default_options["massflow_min_kg_per_s"] = 5
        self.default_options["massflow_max_kg_per_s"] = 10
        self.default_options["massflow_evaluation_points"] = 2
        self.default_options["terrain"] = "Offshore"
        self.default_options["electricity_price_eur_per_mw"] = 60
        self.default_options["operating_hours_per_a"] = 8000
        self.default_options["p_inlet_bar"] = 10
        self.default_options["p_outlet_bar"] = 70

    def _set_options(self, options: dict):
        """
        Sets all provided options
        """
        super()._set_options(options)

        try:
            self.options["length_km"] = options["length_km"]
        except KeyError:
            raise KeyError(
                "You need to at least specify the pipeline length (length_km)"
            )

        # Set options
        self._set_option_value("source", options)
        self.options["discount_rate"] = self.discount_rate

        if self.options["source"] == "Oeuvray":
            # Input units
            self.currency_in = "EUR"
            self.financial_year_in = 2024

            # Options
            for o in self.default_options.keys():
                self._set_option_value(o, options)
        else:
            raise ValueError("This source is not available")

    def calculate_indicators(self, options: dict):
        """
        Calculates financial indicators
        """
        super().calculate_indicators(options)

        if self.options["source"] == "Oeuvray":
            if (
                self.options["massflow_min_kg_per_s"]
                == self.options["massflow_max_kg_per_s"]
            ):
                range_massflow_kg_per_s = [self.options["massflow_min_kg_per_s"]]
            else:
                range_massflow_kg_per_s = np.linspace(
                    self.options["massflow_min_kg_per_s"],
                    self.options["massflow_max_kg_per_s"],
                    self.options["massflow_evaluation_points"],
                )

            calculation_module = CO2Chain_Oeuvray()
            self.financial_indicators["lifetime"] = calculation_module.universal_data[
                "z_pumpcomp"
            ]

            # Calculate costs for different mass flow rates
            costs = pd.DataFrame()
            for massflow_kg_per_s in range_massflow_kg_per_s:
                print(massflow_kg_per_s)
                massflow_t_per_h = massflow_kg_per_s / 1000 * 3600
                self.options["massflow_kg_per_s"] = massflow_kg_per_s
                cost = calculation_module.calculate_cost(self.options)

                # Correct for compression lifetime
                cr_pipeline = (
                    self.discount_rate
                    * (1 + self.discount_rate)
                    ** calculation_module.universal_data["z_pipe"]
                    / (
                        (1 + self.discount_rate)
                        ** calculation_module.universal_data["z_pipe"]
                        - 1
                    )
                )
                cr_compressor = (
                    self.discount_rate
                    * (1 + self.discount_rate)
                    ** calculation_module.universal_data["z_pumpcomp"]
                    / (
                        (1 + self.discount_rate)
                        ** calculation_module.universal_data["z_pumpcomp"]
                        - 1
                    )
                )
                correction_factor = cr_pipeline / cr_compressor

                costs.loc[massflow_t_per_h, "capex_pipeline"] = (
                    cost["cost_pipeline"]["unit_capex"] * correction_factor
                )
                costs.loc[massflow_t_per_h, "capex_compression"] = cost[
                    "cost_compression"
                ]["unit_capex"]
                costs.loc[massflow_t_per_h, "capex_total"] = (
                    cost["cost_pipeline"]["unit_capex"] * correction_factor
                    + cost["cost_compression"]["unit_capex"]
                    + cost["cost_compression"]["unit_capex"]
                )
                costs.loc[massflow_t_per_h, "opex_var"] = (
                    cost["cost_pipeline"]["opex_var"] * correction_factor
                    + cost["cost_compression"]["opex_var"]
                )
                costs.loc[massflow_t_per_h, "opex_fix"] = (
                    cost["cost_pipeline"]["opex_fix_abs"]
                    + cost["cost_compression"]["opex_fix_abs"]
                ) / costs.loc[massflow_t_per_h, "capex_total"]
                costs.loc[massflow_t_per_h, "specific_compression_energy"] = cost[
                    "energy_requirements"
                ]["specific_compression_energy"]
                costs.loc[massflow_t_per_h, "levelized_cost"] = cost["levelized_cost"]

            # Fit linear cost function to results
            costs["intercept"] = 1
            d = costs.reset_index(names="massflow_t_per_h")
            x = d[["massflow_t_per_h", "intercept"]]
            y = d["capex_total"]

            linmodel = sm.OLS(y, x)
            linfit = linmodel.fit()
            coeff = linfit.params
            #
            # d["fitted"] = linfit.predict(x)
            #
            # plt.plot(d["massflow_t_per_h"],d["capex_total"])
            # plt.plot(d["massflow_t_per_h"], d["fitted"])
            #
            #
            # plt.show()

            self.financial_indicators["gamma1"] = convert_currency(
                coeff["intercept"],
                self.financial_year_in,
                self.financial_year_out,
                self.currency_in,
                self.currency_out,
            )
            self.financial_indicators["gamma2"] = convert_currency(
                coeff["massflow_t_per_h"],
                self.financial_year_in,
                self.financial_year_out,
                self.currency_in,
                self.currency_out,
            )
            self.financial_indicators["gamma3"] = 0
            self.financial_indicators["gamma4"] = 0
            self.financial_indicators["opex_fixed"] = costs["opex_fix"].mean()
            self.financial_indicators["opex_variable"] = convert_currency(
                costs["opex_var"].mean(),
                self.financial_year_in,
                self.financial_year_out,
                self.currency_in,
                self.currency_out,
            )
            self.financial_indicators["levelized_cost"] = convert_currency(
                costs["levelized_cost"].mean(),
                self.financial_year_in,
                self.financial_year_out,
                self.currency_in,
                self.currency_out,
            )

            self.technical_indicators["energyconsumption"] = {
                "CO2captured": {"cons_model": 1, "k_flow": 0, "k_flowDistance": 0},
                "electricity": {
                    "cons_model": 1,
                    "k_flow": costs["specific_compression_energy"].mean(),
                    "k_flowDistance": 0,
                },
            }

            # Write to json template
            self.json_data["Economics"]["gamma1"] = self.financial_indicators["gamma1"]
            self.json_data["Economics"]["gamma2"] = self.financial_indicators["gamma2"]
            self.json_data["Economics"]["gamma3"] = self.financial_indicators["gamma3"]
            self.json_data["Economics"]["gamma4"] = self.financial_indicators["gamma4"]
            self.json_data["Economics"]["OPEX_variable"] = self.financial_indicators[
                "opex_variable"
            ]
            self.json_data["Economics"]["opex_fixed"] = self.financial_indicators[
                "opex_fixed"
            ]
            self.json_data["Economics"]["lifetime"] = self.financial_indicators[
                "lifetime"
            ]
            self.json_data["Performance"]["energyconsumption"] = (
                self.technical_indicators["energyconsumption"]
            )

            return {
                "financial_indicators": self.financial_indicators,
                "technical_indicators": self.technical_indicators,
            }
