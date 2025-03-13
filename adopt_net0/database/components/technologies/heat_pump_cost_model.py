from .utilities import Irena, Nrel, Dea
from adopt_net0.database.utilities import convert_currency
from adopt_net0.database.data_component import DataComponent_CostModel


class HeatPump_CostModel(DataComponent_CostModel):
    """
    Heat Pump cost model

    Possible options are:

    If source = "DEA"

    - cost model is based on Danish Energy Agency (2025): Technology Data for Generation of Electricity and District Heating
    - projection_year: future year for which to estimate cost (possible values: 2022-2050)
    - hp_type: can be "air_sourced_1MW" or "air_sourced_3MW" or "air_sourced_10MW" or "seawater_20MW"

    Financial indicators are:

    - unit_capex in [currency]/MW (el)
    - fixed capex as fraction of annualized capex
    - variable opex in [currency]/MWh (el)
    - levelized cost in [currency]/MWh (el)
    - lifetime in years
    """

    def __init__(self, tec_name):
        super().__init__(tec_name)
        # Default options:
        self.default_options["source"] = "DEA"
        self.default_options["projection_year"] = 2030
        self.default_options["hp_type"] = "air_sourced_3MW"

    def _set_options(self, options: dict):
        """
        Sets all provided options
        """
        super()._set_options(options)

        # Set options
        self._set_option_value("source", options)
        self.options["discount_rate"] = self.discount_rate

        if self.options["source"] == "DEA":
            # Input units
            self.currency_in = "EUR"
            self.financial_year_in = 2020
            self.options["hp_type"] = options["hp_type"]
            self.options["projection_year"] = options["projection_year"]

        else:
            raise ValueError("This source is not available")

    def calculate_indicators(self, options: dict):
        """
        Calculates financial indicators
        """
        super().calculate_indicators(options)

        if self.options["source"] == "DEA":
            calculation_module = self._create_calculation_module_dea()

        cost = calculation_module.calculate_cost(self.options)

        self.financial_indicators["unit_capex"] = convert_currency(
            cost["unit_capex"] * 1000,
            self.financial_year_in,
            self.financial_year_out,
            self.currency_in,
            self.currency_out,
        )
        self.financial_indicators["opex_variable"] = convert_currency(
            cost["opex_var"] * 1000,
            self.financial_year_in,
            self.financial_year_out,
            self.currency_in,
            self.currency_out,
        )
        self.financial_indicators["opex_fix"] = cost["opex_fix"]
        self.financial_indicators["levelized_cost"] = (
            convert_currency(
                cost["levelized_cost"],
                self.financial_year_in,
                self.financial_year_out,
                self.currency_in,
                self.currency_out,
            )
            * 1000
        )
        self.financial_indicators["lifetime"] = cost["lifetime"]

        # Write to json template
        self.json_data["Economics"]["unit_CAPEX"] = self.financial_indicators[
            "unit_capex"
        ]
        self.json_data["Economics"]["OPEX_fixed"] = self.financial_indicators[
            "opex_fix"
        ]
        self.json_data["Economics"]["OPEX_variable"] = self.financial_indicators[
            "opex_variable"
        ]
        self.json_data["Economics"]["lifetime"] = self.financial_indicators["lifetime"]
        self.json_data["size_is_int"] = 0

        return {"financial_indicators": self.financial_indicators}

    def _create_calculation_module_dea(self):
        """
        Creates calculation module for source DEA

        :return: calculation_module
        """
        return Dea(self.options["hp_type"])
