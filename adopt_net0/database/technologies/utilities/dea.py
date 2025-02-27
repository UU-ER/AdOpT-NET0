import pandas as pd
from pathlib import Path


class Dea:
    """
    Calculates wind energy costs from NREL
    """

    def __init__(self, technology):

        dea_input_path = Path(__file__).parent.parent.parent / Path(
            "./data/technologies/dea/technology_data_for_el_and_dh - 0016.xlsx"
        )

        all_data = pd.read_excel(
            dea_input_path, sheet_name="alldata_flat", index_col=None
        )

        if technology == "Wind_Onshore":
            filter_tec = "20 Onshore turbines"
            filter_cf = "Average annual full-load hours [MWh_e/MW_e]"
            filter_var_opex = "Variable O&M (*total) [EUR/MWh_e]"
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_e/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = [
                "Nominal investment (equipment) [MEUR/MW_e]",
                "Nominal investment (grid connection) [MEUR/MW_e]",
                "Nominal investment (installation/development) [MEUR/MW_e]",
                "Nominal investment (land purchase/rent) [MEUR/MW_e]",
                "Nominal investment (purchase of neighbour settlements) [MEUR/MW_e]",
            ]

        elif technology == "Wind_Offshore_fixed":
            filter_tec = "21 Far shore Wind DC Fixed"
            filter_cf = "Average annual full-load hours [MWh_e/MW_e]"
            filter_var_opex = "Variable O&M (*total) [EUR/MWh_e]"
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_e/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = [
                "Nominal investment (array cables) [MEUR/MW_e]",
                "Nominal investment (foundation) [MEUR/MW_e]",
                "Nominal investment (project development etc.) [MEUR/MW_e]",
                "Nominal investment (turbines) [MEUR/MW_e]",
            ]

        elif technology == "Wind_Offshore_floating":
            filter_tec = "21 Far shore Wind DC Floating"
            filter_cf = "Average annual full-load hours [MWh_e/MW_e]"
            filter_var_opex = "Variable O&M (*total) [EUR/MWh_e]"
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_e/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = [
                "Nominal investment (array cables) [MEUR/MW_e]",
                "Nominal investment (foundation) [MEUR/MW_e]",
                "Nominal investment (project development etc.) [MEUR/MW_e]",
                "Nominal investment (turbines) [MEUR/MW_e]",
            ]

        elif technology == "Photovoltaic_utility":
            filter_tec = "22 Utility-scale PV"
            filter_cf = "Average annual full-load hours [MWh_e/MW_e]"
            filter_var_opex = None
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_e/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = ["Nominal investment (*total) [MEUR/MW_e]"]

        elif technology == "Photovoltaic_distributed_commercial":
            filter_tec = "22 Rooftop PV comm.&industrial"
            filter_cf = "Average annual full-load hours [MWh_e/MW_e]"
            filter_var_opex = None
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_e/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = ["Nominal investment (*total) [MEUR/MW_e]"]

        elif technology == "Photovoltaic_distributed_residential":
            filter_tec = "22 Rooftop PV residential"
            filter_cf = "Average annual full-load hours [MWh_e/MW_e]"
            filter_var_opex = None
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_e/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = ["Nominal investment (*total) [MEUR/MW_e]"]

        else:
            raise ValueError("Technology not available")

        # Filter data
        technology_data = all_data[all_data["ws"] == filter_tec]
        technology_data = technology_data[technology_data["est"] == "ctrl"]
        self.capex_meur_per_mw = technology_data[
            technology_data["par"].isin(filter_capex)
        ]
        self.opex_fixed_eur_per_mw_per_year = technology_data[
            technology_data["par"] == filter_fixed_opex
        ]
        self.opex_variable_eur_per_mwh = technology_data[
            technology_data["par"] == filter_var_opex
        ]
        self.cf = technology_data[technology_data["par"] == filter_cf]
        self.cf["val"] = self.cf["val"] / 8760
        self.lifetime = technology_data[technology_data["par"] == filter_lifetime]

        # Cost components
        self.unit_capex = None
        self.opex_fix = None
        self.opex_var = None
        self.levelized_cost = None

    def calculate_cost(self, options: dict) -> dict:
        """
        Calculates the cost of wind energy

        :param dict options: Options to use
        :return: unit_capex (EUR2022/kW), opex_fix (EUR2022/kW/yr), opex_var (EUR2022/kWh) and lifetime (yrs)
        :rtype: dict
        """
        discount_rate = options["discount_rate"]

        # Capex
        capex_meur_per_mw = self.capex_meur_per_mw[
            self.capex_meur_per_mw["year"] == options["projection_year"]
        ]
        if len(capex_meur_per_mw) == 0:
            raise ValueError("projection_year is not available")
        self.unit_capex = capex_meur_per_mw["val"].sum() * 1e3

        # Lifetime
        lifetime = self.lifetime[self.lifetime["year"] == options["projection_year"]]
        if len(lifetime) != 1:
            raise ValueError("Something went wrong with lifetime calculation")
        self.lifetime = lifetime["val"].sum()

        # Opex fix
        opex_fixed_eur_per_mw_per_year = self.opex_fixed_eur_per_mw_per_year[
            self.opex_fixed_eur_per_mw_per_year["year"] == options["projection_year"]
        ]
        if len(opex_fixed_eur_per_mw_per_year) != 1:
            raise ValueError("Something went wrong with fixed opex calculation")
        opex_fixed_eur_per_kw_per_year = (
            opex_fixed_eur_per_mw_per_year["val"].sum() / 1000
        )

        crf = (
            discount_rate
            * (1 + discount_rate) ** self.lifetime
            / ((1 + discount_rate) ** self.lifetime - 1)
        )

        self.opex_fix = opex_fixed_eur_per_kw_per_year / (crf * self.unit_capex)

        # Opex var
        opex_var_eur_per_mwh = self.opex_variable_eur_per_mwh[
            self.opex_variable_eur_per_mwh["year"] == options["projection_year"]
        ]
        if len(opex_var_eur_per_mwh) == 0:
            self.opex_var = 0
        elif len(opex_var_eur_per_mwh) == 1:
            self.opex_var = opex_var_eur_per_mwh["val"].sum() / 1000
        else:
            raise ValueError("Something went wrong with variable opex calculation")

        # Capacity factor
        cf = self.cf[self.cf["year"] == options["projection_year"]]
        if len(cf) != 1:
            raise ValueError("Something went wrong with fixed opex calculation")
        self.cf = cf["val"].sum()

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

        self.levelized_cost = (
            self.unit_capex * crf + self.opex_fix + self.opex_var * self.cf * 8760
        ) / (self.cf * 8760)
