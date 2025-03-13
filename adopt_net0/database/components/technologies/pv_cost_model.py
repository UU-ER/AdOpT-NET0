from .utilities import Irena, Nrel, Dea
from adopt_net0.database.utilities import convert_currency
from adopt_net0.database.data_component import DataComponent_CostModel


class PV_CostModel(DataComponent_CostModel):
    """
    Photovoltaic energy

    Possible options are:

    If source = "IRENA"

    - cost model is based on IRENA (2023): Renewable power generation costs in 2023 for utility scale photovoltaics
    - region can be chosen among different countries

    If source = "NREL"

    - cost model is based on NREL (2024): 2024 Annual Technology Baseline (ATB) for Wind Turbine Technology 1
    - projection_year: future year for which to estimate cost (possible values: 2022-2050)
    - projection_type: can be "Advanced", "Moderate", or "Conservative"
    - pv_type: can be "utility" or "rooftop commercial" or "rooftop residential"

    If source = "DEA"

    - cost model is based on Danish Energy Agency (2025): Technology Data for Generation of Electricity and District Heating
    - projection_year: future year for which to estimate cost (possible values: 2022-2050)
    - pv_type: can be "utility" or "rooftop commercial" or "rooftop residential"

    Financial indicators are:

    - unit_capex in [currency]/MWh
    - fixed capex as fraction of up-front capex
    - variable opex in [currency]/MWh
    - levelized cost in [currency]/MWh
    - lifetime in years
    """

    def __init__(self, tec_name):
        super().__init__(tec_name)
        # Default options:
        self.default_options["source"] = "IRENA"
        self.default_options["region"] = "Germany"

    def _set_options(self, options: dict):
        """
        Sets all provided options
        """
        super()._set_options(options)

        # Set options
        self._set_option_value("source", options)
        self.options["discount_rate"] = self.discount_rate

        if self.options["source"] == "IRENA":
            # Input units
            self.currency_in = "USD"
            self.financial_year_in = 2023

            # Options
            for o in self.default_options.keys():
                self._set_option_value(o, options)

        elif self.options["source"] == "NREL":
            # Input units
            self.currency_in = "USD"
            self.financial_year_in = 2022
            self.options["pv_type"] = options["pv_type"]
            self.options["projection_year"] = options["projection_year"]
            self.options["projection_type"] = options["projection_type"]

        elif self.options["source"] == "DEA":
            # Input units
            self.currency_in = "EUR"
            self.financial_year_in = 2020
            self.options["pv_type"] = options["pv_type"]
            self.options["projection_year"] = options["projection_year"]

        else:
            raise ValueError("This source is not available")

    def calculate_indicators(self, options: dict):
        """
        Calculates financial indicators
        """
        super().calculate_indicators(options)

        if self.options["source"] == "IRENA":
            calculation_module = self._create_calculation_module_irena()
        elif self.options["source"] == "NREL":
            calculation_module = self._create_calculation_module_nrel()
        elif self.options["source"] == "DEA":
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

        return {"financial_indicators": self.financial_indicators}

    def _create_calculation_module_irena(self):
        """
        Creates calculation module for source IRENA

        :return: calculation_module
        """

        return Irena("Photovoltaic")

    def _create_calculation_module_nrel(self):
        """
        Creates calculation module for source IRENA

        :return: calculation_module
        """
        if self.options["pv_type"] == "utility":
            return Nrel("Photovoltaic_utility")
        elif self.options["pv_type"] == "rooftop commercial":
            return Nrel("Photovoltaic_distributed_commercial")
        elif self.options["pv_type"] == "rooftop residential":
            return Nrel("Photovoltaic_distributed_residential")
        else:
            raise ValueError(
                "Wrong pv_type specified, needs to be utility or rooftop commercial or rooftop residential"
            )

    def _create_calculation_module_dea(self):
        """
        Creates calculation module for source DEA

        :return: calculation_module
        """
        if self.options["pv_type"] == "utility":
            return Dea("Photovoltaic_utility")
        elif self.options["pv_type"] == "rooftop commercial":
            return Dea("Photovoltaic_distributed_commercial")
        elif self.options["pv_type"] == "rooftop residential":
            return Dea("Photovoltaic_distributed_residential")
        else:
            raise ValueError(
                "Wrong pv_type specified, needs to be utility or rooftop commercial or rooftop residential"
            )
