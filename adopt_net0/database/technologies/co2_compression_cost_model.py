import pandas as pd
import numpy as np
from statsmodels import api as sm

# import matplotlib.pyplot as plt

from ..networks.utilities import CO2Compression_Oeuvray
from ..utilities import convert_currency
from ..data_component import DataComponent_CostModel


class CO2_Compression_CostModel(DataComponent_CostModel):
    """
    CO2 Compression

    Possible options are:

    If source = "Oeuvray"

    - cost and energy consumption model is based on Oeuvray, P., Burger, J., Roussanaly, S., Mazzotti, M., Becattini, V. (2024):
      Multi-criteria assessment of inland and offshore carbon dioxide transport options, Journal of Cleaner Production, https://doi.org/10.1016/j.jclepro.2024.140781.    - cumulative_capacity_installed_t_per_a: total global installed capturing capacity in t/a. Determines the cost reduction due to learning.
    - massflow_min_kg_per_s: minimal mass flow rate of CO2 in kg/s to evaluate for costs
    - massflow_max_kg_per_s: maximal mass flow rate of CO2 in kg/s to evaluate for costs
    - massflow_evaluation_points: for how many points should costs be calculated between massflow_min_kg_per_s, massflow_max_kg_per_s (includes min and max)
    - p_inlet_bar: inlet pressure in bar (beginning of pipeline)
    - p_outlet_bar: outlet pressure in bar (end of pipeline)
    - capex_model: for 1 linear cost through origin, for 3 linear with intercept

    Financial indicators are:

    - unit_capex in [currency]/t/a
    - fixed capex as fraction of annualized capex
    - variable opex in [currency]/ton
    - lifetime in years

    Technical indicators are:

    - energyconsumption in MWh/t compressed

    -
    """

    def __init__(self, tec_name):
        super().__init__(tec_name)
        # Default options:
        self.default_options["source"] = "Oeuvray"
        self.default_options["massflow_min_kg_per_s"] = 5
        self.default_options["massflow_max_kg_per_s"] = 10
        self.default_options["massflow_evaluation_points"] = 2
        self.default_options["p_inlet_bar"] = 10
        self.default_options["p_outlet_bar"] = 70
        self.default_options["capex_model"] = 1

    def _set_options(self, options: dict):
        """
        Sets all provided options
        """
        super()._set_options(options)

        # Set options
        self._set_option_value("source", options)
        self.options["discount_rate"] = self.discount_rate

        if self.options["source"] == "Oeuvray":
            # Input units
            self.currency_in = "EUR"
            self.financial_year_in = 2024
            self.options["terrain"] = "Onshore"

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
                range_m_kg_per_s = [self.options["massflow_min_kg_per_s"]]
            else:
                range_m_kg_per_s = np.linspace(
                    self.options["massflow_min_kg_per_s"],
                    self.options["massflow_max_kg_per_s"],
                    self.options["massflow_evaluation_points"],
                )
            calculation_module = CO2Compression_Oeuvray()
            self.financial_indicators["lifetime"] = calculation_module.universal_data[
                "z_pumpcomp"
            ]

            # Calculate costs for different mass flow rates
            costs = pd.DataFrame()
            for m_kg_per_s in range_m_kg_per_s:
                m_t_per_h = m_kg_per_s / 1000 * 3600
                self.options["m_kg_per_s"] = m_kg_per_s
                cost = calculation_module.calculate_cost(self.options)

                costs.loc[m_t_per_h, "unit_capex"] = cost["unit_capex"]
                costs.loc[m_t_per_h, "opex_fix"] = cost["opex_fix"]
                costs.loc[m_t_per_h, "opex_var"] = cost["opex_var"]
                costs.loc[m_t_per_h, "specific_compression_energy"] = cost[
                    "specific_compression_energy_mwh_per_t"
                ]

            # Fit linear cost function to results
            if self.options["capex_model"] == 1:
                costs["intercept"] = 0
            else:
                costs["intercept"] = 1

            d = costs.reset_index(names="m_t_per_h")
            x = d[["m_t_per_h", "intercept"]]
            y = d["unit_capex"]

            linmodel = sm.OLS(y, x)
            linfit = linmodel.fit()
            coeff = linfit.params

            #
            # d["fitted"] = linfit.predict(x)
            #
            # plt.plot(d["m_t_per_h"],d["unit_capex"])
            # plt.plot(d["m_t_per_h"], d["fitted"])
            #
            #
            # plt.show()

            self.financial_indicators["unit_capex"] = convert_currency(
                coeff["m_t_per_h"],
                self.financial_year_in,
                self.financial_year_out,
                self.currency_in,
                self.currency_out,
            )
            self.financial_indicators["fix_capex"] = convert_currency(
                coeff["intercept"],
                self.financial_year_in,
                self.financial_year_out,
                self.currency_in,
                self.currency_out,
            )
            self.financial_indicators["opex_variable"] = convert_currency(
                costs["opex_var"].mean(),
                self.financial_year_in,
                self.financial_year_out,
                self.currency_in,
                self.currency_out,
            )
            self.financial_indicators["opex_fix"] = costs["opex_fix"].mean()

            self.technical_indicators["energyconsumption"] = costs[
                "specific_compression_energy"
            ].mean()

        # Write to json template
        self.json_data["Economics"]["CAPEX_model"] = self.options["capex_model"]
        if self.options["capex_model"] == 3:
            self.json_data["Economics"]["fix_CAPEX"] = self.financial_indicators[
                "fix_capex"
            ]
        else:
            self.json_data["Economics"]["fix_CAPEX"] = 0

        self.json_data["Economics"]["unit_capex"] = self.financial_indicators[
            "unit_capex"
        ]
        self.json_data["Economics"]["OPEX_fixed"] = self.financial_indicators[
            "opex_fix"
        ]
        self.json_data["Economics"]["OPEX_variable"] = self.financial_indicators[
            "opex_variable"
        ]
        self.json_data["Economics"]["lifetime"] = self.financial_indicators["lifetime"]

        self.json_data["Performance"]["input_ratios"]["electricity"] = (
            self.technical_indicators["energyconsumption"]
        )

        return {
            "financial_indicators": self.financial_indicators,
            "technical_indicators": self.technical_indicators,
        }
