from pathlib import Path
import json

from .utilities import Dac_sievert
from ..utilities import convert_currency
from ..data_component import DataComponent
from ...data_management.utilities import open_json

PATH_CURRENT_DIR = Path(__file__).parent


class Dac_SolidSorbent(DataComponent):
    """
    DAC SOLID SORBENT

    Possible options are:
    If source = "Sievert"
    - cumulative_capacity_installed_t_per_a: total global installed capturing capacity in t/a. Determines the learning rates.
    - average_productivity_per_module_kg_per_h: average productivity of a DAC module in kg/h (default is at 20 degree, 43% humidity)
    - capacity_factor: used to calculate levelized cost of removal

    Financial indicators are:
    - module_capex in [currency]/module
    - fixed capex as fraction of annualized capex
    - variable opex in [currency]/ton
    - levelized cost in [currency]/ton without energy costs
    - lifetime in years
    """

    def __init__(self, options):
        super().__init__(options)

        # Json template
        self.json_template = open_json(
            "DAC_Adsorption", PATH_CURRENT_DIR.parent / "templates" / "technology_data"
        )

        # Default options:
        self.default_options["source"] = "Sievert"
        self.default_options["cumulative_capacity_installed_t_per_a"] = 0
        self.default_options["average_productivity_per_module_kg_per_h"] = 8.6
        self.default_options["capacity_factor"] = 0.9

        # Set options
        self._set_option_value("source", options)

        if self.options["source"] == "Sievert":
            # Input units
            self.currency_in = "USD"
            self.financial_year_in = 2022

            # Options
            self._set_option_value("cumulative_capacity_installed_t_per_a", options)
            self._set_option_value("average_productivity_per_module_kg_per_h", options)
            self._set_option_value("capacity_factor", options)
        else:
            raise ValueError("This source is not available")

    def calculate_financial_indicators(self):
        """
        Calculates financial indicators
        """
        if self.options["source"] == "Sievert":
            module_capacity_t_per_a = (
                self.options["average_productivity_per_module_kg_per_h"] * 8760 / 1000
            )

            calculation_module = Dac_sievert("SS", 2030)
            cost = calculation_module.calculate_cost(
                self.discount_rate,
                self.options["cumulative_capacity_installed_t_per_a"],
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
                calculation_module.calculate_levelized_cost(
                    self.options["capacity_factor"]
                ),
                self.financial_year_in,
                self.financial_year_out,
                self.currency_in,
                self.currency_out,
            )
            self.financial_indicators["lifetime"] = calculation_module.lifetime

        return self.financial_indicators

    def write_json(self, path):
        """
        Write json to specified path

        Overwritten in child classes
        :param str path: path to write to
        """
        self.calculate_financial_indicators()
        technology_data = self.json_template
        technology_data["Economics"]["unit_CAPEX"] = self.financial_indicators[
            "module_capex"
        ]
        technology_data["Economics"]["OPEX_fixed"] = self.financial_indicators[
            "opex_fix"
        ]
        technology_data["Economics"]["OPEX_variable"] = self.financial_indicators[
            "opex_variable"
        ]
        technology_data["Economics"]["lifetime"] = self.financial_indicators["lifetime"]

        with open(
            Path(path) / "DAC_Adsorption.json",
            "w",
        ) as f:
            json.dump(technology_data, f, indent=4)

    def calculate_technical_indicators(self):
        """
        Overwritten in child classes
        """
        raise NotImplementedError
