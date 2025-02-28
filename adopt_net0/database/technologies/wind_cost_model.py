from .utilities import Irena, Nrel, Dea
from ..utilities import convert_currency
from ..data_component import DataComponent_CostModel


class WindEnergy_CostModel(DataComponent_CostModel):
    """
    Wind energy (onshore and offshore)

    Possible options are:

    - nameplate_capacity_MW: Capacity of a wind turbine in MW
    - terrain: Onshore or offshore wind turbine

    If source = "IRENA"

    - cost model is based on IRENA (2023): Renewable power generation costs in 2023
    - region can be chosen among different countries ('Brazil', 'Canada', 'China', 'France', 'Germany', 'India', 'Japan', 'Spain', 'Sweden', 'United Kingdom', 'United States', 'Australia', 'Ireland')

    If source = "NREL"

    - cost model is based on NREL (2024): 2024 Annual Technology Baseline (ATB) for Wind Turbine Technology 1
    - projection_year: future year for which to estimate cost (possible values: 2030, 2040, 2050)
    - projection_type: can be "Advanced", "Moderate", or "Conservative"
    - mounting_type: can be "fixed" or "floating" for offshore turbines

    If source = "DEA"

    - cost model is based on Danish Energy Agency (2025): Technology Data for Generation of Electricity and District Heating
    - projection_year: future year for which to estimate cost (possible values: 2022-2050)
    - mounting_type: can be "fixed" or "floating" for offshore turbines

    Financial indicators are:

    - unit_capex in [currency]/turbine
    - fixed capex as fraction of annualized capex
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

        try:
            self.options["terrain"] = options["terrain"]
            self.options["nameplate_capacity_MW"] = options["nameplate_capacity_MW"]
        except KeyError:
            raise KeyError(
                "You need to at least specify the terrain (onshore or offshore) and the nameplate capacity"
            )

        # Set options
        self._set_option_value("source", options)
        self.options["discount_rate"] = self.discount_rate

        if self.options["source"] == "IRENA":
            # Input units
            self.currency_in = "USD"
            self.financial_year_in = 2023
            for o in self.default_options.keys():
                self._set_option_value(o, options)

        elif self.options["source"] == "NREL":
            # Input units
            self.currency_in = "USD"
            self.financial_year_in = 2022
            self.options["terrain"] = options["terrain"]
            self.options["projection_year"] = options["projection_year"]
            self.options["projection_type"] = options["projection_type"]
            if self.options["terrain"] == "Offshore":
                self.options["mounting_type"] = options["mounting_type"]

        elif self.options["source"] == "DEA":
            # Input units
            self.currency_in = "EUR"
            self.financial_year_in = 2020
            self.options["terrain"] = options["terrain"]
            self.options["projection_year"] = options["projection_year"]
            if self.options["terrain"] == "Offshore":
                self.options["mounting_type"] = options["mounting_type"]

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

        self.financial_indicators["module_capex"] = convert_currency(
            cost["unit_capex"] * options["nameplate_capacity_MW"] * 1000,
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
        self.financial_indicators["lifetime"] = int(cost["lifetime"])

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
        self.json_data["size_is_int"] = 1

        return {"financial_indicators": self.financial_indicators}

    def _create_calculation_module_nrel(self):
        """
        Creates calculation module for source NREL

        :return: calculation_module
        """
        if self.options["terrain"] == "Offshore":
            if not "mounting_type" in self.options:
                raise ValueError(
                    "For offshore wind, you need to specify a mounting_type (fixed or floating)"
                )

            if self.options["mounting_type"] == "fixed":
                return Nrel("Wind_Offshore_fixed")
            elif self.options["mounting_type"] == "floating":
                return Nrel("Wind_Offshore_floating")
            else:
                raise ValueError("mounting_type can only be fixed or floating")
        elif self.options["terrain"] == "Onshore":
            return Nrel("Wind_Onshore")
        else:
            raise ValueError("Wrong terrain specified, needs to be Onshore or Offshore")

    def _create_calculation_module_irena(self):
        """
        Creates calculation module for source IRENA

        :return: calculation_module
        """
        if self.options["terrain"] == "Offshore":
            return Irena("Wind_Offshore")
        elif self.options["terrain"] == "Onshore":
            return Irena("Wind_Onshore")
        else:
            raise ValueError("Wrong terrain specified, needs to be Onshore or Offshore")

    def _create_calculation_module_dea(self):
        """
        Creates calculation module for source Danish Energy Agency

        :return: calculation_module
        """
        if self.options["terrain"] == "Offshore":
            if not "mounting_type" in self.options:
                raise ValueError(
                    "For offshore wind, you need to specify a mounting_type (fixed or floating)"
                )

            if self.options["mounting_type"] == "fixed":
                return Dea("Wind_Offshore_fixed")
            elif self.options["mounting_type"] == "floating":
                return Dea("Wind_Offshore_floating")
            else:
                raise ValueError("mounting_type can only be fixed or floating")
        elif self.options["terrain"] == "Onshore":
            return Dea("Wind_Onshore")
        else:
            raise ValueError("Wrong terrain specified, needs to be Onshore or Offshore")
