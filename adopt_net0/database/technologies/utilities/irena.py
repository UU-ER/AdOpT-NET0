import pandas as pd
from pathlib import Path


class Irena:
    """
    Calculates wind energy costs from IRENA
    """

    def __init__(self, technology):

        irena_input_path = Path(__file__).parent.parent.parent / Path(
            "./data/technologies/irena/IRENA-Datafile-RenPwrGenCosts-in-2023-v1.xlsx"
        )

        # Read data
        lifetime = pd.read_excel(
            irena_input_path,
            sheet_name="Table A1",
            index_col=0,
            skiprows=3,
            usecols="B:C",
        )

        if technology == "Wind_Onshore":
            # Lifetime
            self.lifetime = lifetime.loc["Wind power", "Economic life (years)"]

            # Costs
            capex_big_markets = pd.read_excel(
                irena_input_path,
                sheet_name="Fig 2.5",
                index_col=0,
                skiprows=6,
                usecols="B:AP",
            )
            capex_small_markets = pd.read_excel(
                irena_input_path,
                sheet_name="Fig 2.6",
                index_col=0,
                skiprows=6,
                usecols="B:P",
            )
            self.capex_usd_per_kw = pd.concat([capex_big_markets, capex_small_markets])[
                2023
            ].dropna()

            cf_big_markets = pd.read_excel(
                irena_input_path,
                sheet_name="Fig 2.7",
                index_col=0,
                skiprows=6,
                usecols="B:AP",
            )
            cf_small_markets = pd.read_excel(
                irena_input_path,
                sheet_name="Fig 2.8",
                index_col=0,
                skiprows=6,
                usecols="B:P",
            )
            self.cf = pd.concat([cf_big_markets, cf_small_markets])[2023].dropna()

            opex_usd_per_kw_per_year = pd.read_excel(
                irena_input_path,
                sheet_name="Table A3",
                index_col=0,
                skiprows=2,
                usecols="B:C",
            )
            opex_usd_per_kw_per_year.columns = ["2023"]
            self.opex_usd_per_kw_per_year = opex_usd_per_kw_per_year["2023"]

        elif technology == "Wind_Offshore":
            # Lifetime
            self.lifetime = lifetime.loc["Wind power", "Economic life (years)"]

            # Costs
            capex_index = (
                pd.read_excel(
                    irena_input_path,
                    sheet_name="Table 4.2",
                    index_col=None,
                    skiprows=9,
                    nrows=10,
                    usecols="C:C",
                    header=None,
                )
                .iloc[:, 0]
                .to_list()
            )
            capex_usd_per_kw = pd.read_excel(
                irena_input_path,
                sheet_name="Table 4.2",
                index_col=None,
                skiprows=9,
                nrows=10,
                usecols="H:H",
                header=None,
            )
            capex_usd_per_kw.index = capex_index
            capex_usd_per_kw.columns = [2023]
            capex_usd_per_kw.index = capex_usd_per_kw.index.str.replace(
                "*", "", regex=False
            )
            capex_usd_per_kw[2023] = capex_usd_per_kw[2023].str.replace(
                " ", "", regex=False
            )
            self.capex_usd_per_kw = pd.to_numeric(capex_usd_per_kw[2023].dropna())

            cf_index = (
                pd.read_excel(
                    irena_input_path,
                    sheet_name="Table 4.3",
                    index_col=None,
                    skiprows=7,
                    nrows=7,
                    usecols="B:B",
                    header=None,
                )
                .iloc[:, 0]
                .to_list()
            )
            cf = (
                pd.read_excel(
                    irena_input_path,
                    sheet_name="Table 4.3",
                    index_col=None,
                    skiprows=7,
                    nrows=7,
                    usecols="D:D",
                    header=None,
                )
                / 100
            )
            cf.index = cf_index
            cf.columns = [2023]
            cf.index = cf.index.str.replace("*", "", regex=False)
            self.cf = cf[2023].dropna()

            opex_usd_per_kw_per_year = pd.read_excel(
                irena_input_path,
                sheet_name="Table A5",
                index_col=0,
                skiprows=2,
                usecols="B:C",
            )
            opex_usd_per_kw_per_year.columns = ["2023"]
            self.opex_usd_per_kw_per_year = opex_usd_per_kw_per_year["2023"]

        elif technology == "Photovoltaic":
            self.lifetime = lifetime.loc["Solar PV", "Economic life (years)"]
            capex_usd_per_kw = pd.read_excel(
                irena_input_path,
                sheet_name="Figure 3.4",
                index_col=0,
                skiprows=3,
                usecols="B:P",
            )
            self.capex_usd_per_kw = capex_usd_per_kw[2023].dropna()

            opex = pd.read_excel(
                irena_input_path,
                sheet_name="Table A4",
                index_col=0,
                skiprows=3,
                usecols="B:D",
                header=None,
            )
            opex.columns = ["OECD", "non OECD"]
            opex = opex.loc[2023, :]
            oecd = [
                "Australia",
                "France",
                "Germany",
                "Italy",
                "Japan",
                "Netherlands",
                "Spain",
                "Turkey",
                "United Kingdom",
                "United States",
                "Republic of Korea",
            ]
            opex_usd_per_kw_per_year = pd.DataFrame(index=self.capex_usd_per_kw.index)
            opex_usd_per_kw_per_year[
                "2023"
            ] = opex_usd_per_kw_per_year.index.to_series().apply(
                lambda x: opex["OECD"] if x in oecd else opex["non OECD"]
            )
            self.opex_usd_per_kw_per_year = opex_usd_per_kw_per_year["2023"]

            cf = pd.read_excel(
                irena_input_path,
                sheet_name="Table 3.1",
                index_col=0,
                skiprows=2,
                usecols="B:D",
            )
            cf = cf.loc[2023, "Weighted average"]
            cf = pd.DataFrame(
                cf, index=self.opex_usd_per_kw_per_year.index, columns=["2023"]
            )
            self.cf = cf["2023"]

        else:
            raise ValueError("Technology not available")

        # Determine available regions
        self.available_regions = self.cf.index.intersection(
            self.opex_usd_per_kw_per_year.index
        ).intersection(self.capex_usd_per_kw.index)

        self.capex_usd_per_kw = self.capex_usd_per_kw.loc[self.available_regions]
        self.cf = self.cf.loc[self.available_regions]
        self.opex_usd_per_kw_per_year = self.opex_usd_per_kw_per_year.loc[
            self.available_regions
        ]

        # Cost components
        self.unit_capex = None
        self.opex_fix = None
        self.opex_var = None
        self.levelized_cost = None

    def calculate_cost(self, options: dict) -> dict:
        """
        Calculates the cost of wind energy

        :param dict options: Options to use
        :return: unit_capex (USD2023/kW), opex_fix (% of annualized investment), opex_var (0) and lifetime (yrs)
        :rtype: dict
        """
        region = options["region"]
        discount_rate = options["discount_rate"]
        if region not in self.available_regions:
            raise ValueError(
                f"Region is not available. Available regions are {self.available_regions.to_list()}"
            )

        self.cf = self.cf[region]
        self.unit_capex = self.capex_usd_per_kw[region]
        self.opex_var = 0

        crf = (
            discount_rate
            * (1 + discount_rate) ** self.lifetime
            / ((1 + discount_rate) ** self.lifetime - 1)
        )

        self.opex_fix = self.opex_usd_per_kw_per_year[region] / (crf * self.unit_capex)

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
