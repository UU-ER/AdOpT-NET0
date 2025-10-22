from .utilities import Irena, Nrel, Dea
from adopt_net0.database.utilities import convert_currency
from adopt_net0.database.data_component import DataComponent_CostModel


class Electrolyzer_CostModel(DataComponent_CostModel):
    """
    Hydrogen production with electrolyzer (PEM and AEM, small/medium/large scale)

    Possible options are:

    - type: PEM or AEM electrolyzer
    - capacity_MW: Capacity of a electrolyzer in MW (input electricity)

    If source = "DEA"

    - cost model is based on Danish Energy Agency (2025): Technology Data for Renewable Fuels
    - projection_year: future year for which to estimate cost (possible values: 2022-2050)
    - size: can be "small" or "medium" or "large"

    Financial indicators are:

    - unit_capex in [currency]/electrolyzer
    - fixed capex as fraction of annualized capex
    - variable opex in [currency]/MWh
    - levelized cost in [currency]/MWh
    - lifetime in years
    """

    def __init__(self, tec_name):
        super().__init__(tec_name)
        # Default options:
        self.default_options["source"] = "DEA"
        self.default_options["region"] = "Netherlands"

    def _set_options(self, options: dict):
        """
        Sets all provided options
        """
        super()._set_options(options)

        try:
            self.options["type"] = options["type"]
            # Convert capacity_MW to float if provided
            if "capacity_MW" in options and options["capacity_MW"] is not None:
                self.options["capacity_MW"] = float(options["capacity_MW"])
            else:
                self.options["capacity_MW"] = None
        except KeyError:
            raise KeyError("You need to at least specify the type (PEM or AEM)")
        except (ValueError, TypeError):
            raise TypeError("capacity_MW must be a valid number")

        # Set options
        self._set_option_value("source", options)
        self.options["discount_rate"] = self.discount_rate

        if self.options["source"] == "DEA":
            # Input units
            self.currency_in = "EUR"
            self.financial_year_in = 2020
            self.options["projection_year"] = options["projection_year"]
            # If capacity_MW is not provided or is None, require size
            if self.options["capacity_MW"] is None:
                if "size" not in options:
                    raise KeyError("You need to specify either capacity_MW or size")
                self.options["size"] = options["size"]

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

        # Use the stored capacity_MW instead of the original options
        capacity_mw = self.options["capacity_MW"] or 0
        self.financial_indicators["module_capex"] = convert_currency(
            cost["unit_capex"] * capacity_mw * 1000,
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
        self.json_data["Economics"]["unit_capex"] = self.financial_indicators[
            "module_capex"
        ]
        self.json_data["Economics"]["opex_fixed"] = self.financial_indicators[
            "opex_fix"
        ]
        self.json_data["Economics"]["opex_variable"] = self.financial_indicators[
            "opex_variable"
        ]
        self.json_data["Economics"]["lifetime"] = self.financial_indicators["lifetime"]
        self.json_data["size_is_int"] = 1

        return {"financial_indicators": self.financial_indicators}

    def _create_calculation_module_dea(self):
        """
        Creates calculation module for source Danish Energy Agency

        :return: calculation_module
        """
        # Determine size based on capacity_MW if specified, otherwise use provided size
        if self.options["capacity_MW"] is not None:
            capacity = self.options["capacity_MW"]
            if 0 <= capacity <= 50:
                size = "small"
            elif 51 <= capacity <= 500:
                size = "medium"
            elif 501 <= capacity <= 2000:
                size = "big"
            else:
                raise ValueError(
                    f"Capacity {capacity} MW is outside supported range (0-2000 MW)"
                )
        else:
            # Use provided size if capacity_MW is not specified
            if "size" not in self.options:
                raise ValueError(
                    "You need to specify either capacity_MW or size (small, medium or big)"
                )
            size = self.options["size"]
            if size not in ["small", "medium", "big"]:
                raise ValueError("size can only be small, medium or big")

        # Persist the resolved size so callers/tests can assert it
        self.options["size"] = size

        if self.options["type"] == "PEM":
            if size == "small":
                return Dea("PEMEC_10MW")
            elif size == "medium":
                return Dea("PEMEC_100MW")
            elif size == "big":
                return Dea("PEMEC_1GW")

        elif self.options["type"] == "AEM":
            if size == "small":
                return Dea("AEC_10MW")
            elif size == "medium":
                return Dea("AEC_100MW")
            elif size == "big":
                return Dea("AEC_1GW")
        else:
            raise ValueError("Wrong type specified, needs to be PEM or AEM")
