from .utilities import Dac_sievert
from ..utilities import convert_currency
from ..data_component import DataComponent_CostModel


class Dac_SolidSorbent_CostModel(DataComponent_CostModel):
    """
    DAC (Adsorption)

    Possible options are:

    If source = "Sievert"

    - cost model is based on Sievert, K., Schmidt, T. S., & Steffen, B. (2024). Considering technology characteristics to project future costs of direct air capture. Joule, 8(4), 979-999,  https://doi.org/10.1016/j.joule.2024.02.005.
    - cumulative_capacity_installed_t_per_a: total global installed capturing capacity in t/a. Determines the cost reduction due to learning.
    - average_productivity_per_module_kg_per_h: average productivity of a DAC module in kg/h (default is at 20 degree, 43% humidity)
    - capacity_factor: used to calculate levelized cost of removal

    Financial indicators are:

    - module_capex in [currency]/module
    - fixed capex as fraction of annualized capex
    - variable opex in [currency]/ton
    - levelized cost in [currency]/ton without energy costs
    - lifetime in years
    """

    def __init__(self, tec_name):
        super().__init__(tec_name)
        # Default options:
        self.default_options["source"] = "Sievert"
        self.default_options["cumulative_capacity_installed_t_per_a"] = 0
        self.default_options["average_productivity_per_module_kg_per_h"] = 8.6
        self.default_options["capacity_factor"] = 0.9

    def _set_options(self, options: dict):
        """
        Sets all provided options
        """
        super()._set_options(options)

        # Set options
        self._set_option_value("source", options)

        if self.options["source"] == "Sievert":
            # Input units
            self.currency_in = "USD"
            self.financial_year_in = 2022

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

        if self.options["source"] == "Sievert":
            module_capacity_t_per_a = (
                self.options["average_productivity_per_module_kg_per_h"] * 8760 / 1000
            )

            calculation_module = Dac_sievert("SS", 2030)
            cost = calculation_module.calculate_cost(
                self.discount_rate,
                self.options["cumulative_capacity_installed_t_per_a"],
                self.options["capacity_factor"],
            )

            self.financial_indicators["module_capex"] = convert_currency(
                cost["unit_capex"] * module_capacity_t_per_a,
                self.financial_year_in,
                self.financial_year_out,
                self.currency_in,
                self.currency_out,
            )
            self.financial_indicators["opex_variable"] = convert_currency(
                cost["opex_var"],
                self.financial_year_in,
                self.financial_year_out,
                self.currency_in,
                self.currency_out,
            )
            self.financial_indicators["opex_fix"] = cost["opex_fix"]
            self.financial_indicators["levelized_cost"] = convert_currency(
                cost["levelized_cost"],
                self.financial_year_in,
                self.financial_year_out,
                self.currency_in,
                self.currency_out,
            )
            self.financial_indicators["lifetime"] = cost["lifetime"]

        # Write to json template
        self.json_data["Economics"]["unit_CAPEX"] = self.financial_indicators[
            "module_capex"
        ]
        self.json_data["Economics"]["OPEX_fixed"] = self.financial_indicators[
            "opex_fix"
        ]
        self.json_data["Economics"]["OPEX_variable"] = self.financial_indicators[
            "opex_variable"
        ]
        self.json_data["Economics"]["lifetime"] = self.financial_indicators["lifetime"]

        return {"financial_indicators": self.financial_indicators}
