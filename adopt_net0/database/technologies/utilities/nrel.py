import pandas as pd
from pathlib import Path


class Nrel:
    """
    Calculates wind energy costs from NREL
    """

    def __init__(self, technology):

        nrel_input_path = Path(__file__).parent.parent.parent / Path(
            "./data/technologies/nrel/2024 v2 Annual Technology Baseline Workbook Errata 7-19-2024.xlsx"
        )

        # Read data
        lifetime = pd.read_excel(
            nrel_input_path,
            sheet_name="Financial and CRP Inputs",
            index_col=0,
            skiprows=3,
            usecols="H:I",
        )

        if technology == "Wind_Onshore":
            # Lifetime
            self.lifetime = int(lifetime.loc["Land-Based Wind", "Tech Life CRPs"])

            # Reading options
            sheet_name = "Land-Based Wind"
            capex_row_skip = 138
            cf_row_skip = 74
            opex_fix_row_skip = 234
            opex_variable_row_skip = 266
            cols = "L:AO"

        elif technology == "Wind_Offshore_fixed":
            # Lifetime
            self.lifetime = int(lifetime.loc["Offshore Wind", "Tech Life CRPs"])

            # Reading options
            sheet_name = "Fixed-Bottom Offshore Wind"
            capex_row_skip = 114
            cf_row_skip = 68
            opex_fix_row_skip = 183
            opex_variable_row_skip = 206
            cols = "L:AO"

        elif technology == "Wind_Offshore_floating":
            # Lifetime
            self.lifetime = int(lifetime.loc["Offshore Wind", "Tech Life CRPs"])

            # Reading options
            sheet_name = "Floating Offshore Wind"
            capex_row_skip = 114
            cf_row_skip = 68
            opex_fix_row_skip = 183
            opex_variable_row_skip = 206
            cols = "L:AG"

        elif technology == "Photovoltaic_utility":
            # Lifetime
            self.lifetime = int(lifetime.loc["Solar - Utility PV", "Tech Life CRPs"])

            # Reading options
            sheet_name = "Solar - Utility PV"
            capex_row_skip = 149
            cf_row_skip = 85
            opex_fix_row_skip = 245
            opex_variable_row_skip = 277
            cols = "L:AG"

        elif technology == "Photovoltaic_distributed_commercial":
            # Lifetime
            self.lifetime = int(lifetime.loc["Solar - PV Dist. Comm", "Tech Life CRPs"])

            # Reading options
            sheet_name = "Solar - PV Dist. Comm"
            capex_row_skip = 158
            cf_row_skip = 94
            opex_fix_row_skip = 222
            opex_variable_row_skip = 254
            cols = "L:AG"

        elif technology == "Photovoltaic_distributed_residential":
            # Lifetime
            self.lifetime = int(lifetime.loc["Solar - PV Dist. Res", "Tech Life CRPs"])

            # Reading options
            sheet_name = "Solar - PV Dist. Res"
            capex_row_skip = 158
            cf_row_skip = 94
            opex_fix_row_skip = 222
            opex_variable_row_skip = 254
            cols = "L:AG"

        else:
            raise ValueError("Technology not available")

        self.capex_usd_per_kw = pd.read_excel(
            nrel_input_path,
            sheet_name=sheet_name,
            index_col=0,
            skiprows=capex_row_skip,
            usecols=cols,
            nrows=3,
        )

        self.opex_fixed_usd_per_kw_per_year = pd.read_excel(
            nrel_input_path,
            sheet_name=sheet_name,
            index_col=0,
            skiprows=opex_fix_row_skip,
            usecols=cols,
            nrows=3,
        )

        self.opex_variable_usd_per_kwh_per_year = pd.read_excel(
            nrel_input_path,
            sheet_name=sheet_name,
            index_col=0,
            skiprows=opex_variable_row_skip,
            usecols=cols,
            nrows=3,
        )

        self.cf = pd.read_excel(
            nrel_input_path,
            sheet_name=sheet_name,
            index_col=0,
            skiprows=cf_row_skip,
            usecols=cols,
            nrows=3,
        )

        # Cost components
        self.unit_capex = None
        self.opex_fix = None
        self.opex_var = None
        self.levelized_cost = None

    def calculate_cost(self, options: dict) -> dict:
        """
        Calculates the cost of wind energy

        :param dict options: Options to use
        :return: unit_capex (USD2023/kW), opex_fix (USD2023/kW/yr), opex_var (0) and lifetime (yrs)
        :rtype: dict
        """
        discount_rate = options["discount_rate"]
        self.cf = self.cf.loc[options["projection_type"], options["projection_year"]]
        self.unit_capex = self.capex_usd_per_kw.loc[
            options["projection_type"], options["projection_year"]
        ]
        self.opex_var = self.opex_variable_usd_per_kwh_per_year.loc[
            options["projection_type"], options["projection_year"]
        ]

        crf = (
            discount_rate
            * (1 + discount_rate) ** self.lifetime
            / ((1 + discount_rate) ** self.lifetime - 1)
        )

        self.opex_fix = self.opex_fixed_usd_per_kw_per_year.loc[
            options["projection_type"], options["projection_year"]
        ] / (crf * self.unit_capex)

        self._calculate_levelized_cost(discount_rate)

        return {
            "unit_capex": self.unit_capex,
            "opex_fix": self.opex_fix,
            "opex_var": self.opex_var,
            "lifetime": self.lifetime,
            "levelized_cost": self.levelized_cost,
        }

    def _calculate_levelized_cost(self, discount_rate: float):
        """
        Calculates levelized costs

        :param str region: Region to calculate costs for
        :return: lcoc
        :rtype: float
        """
        crf = (
            discount_rate
            * (1 + discount_rate) ** self.lifetime
            / ((1 + discount_rate) ** self.lifetime - 1)
        )

        self.levelized_cost = (self.unit_capex * crf + self.opex_fix) / (self.cf * 8760)
