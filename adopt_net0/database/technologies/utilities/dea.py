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
        self.other = {}

        if technology == "Wind_Onshore":
            self.tec_type = "RES"
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
            self.tec_type = "RES"
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
            self.tec_type = "RES"
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
            self.tec_type = "RES"
            filter_tec = "22 Utility-scale PV"
            filter_cf = "Average annual full-load hours [MWh_e/MW_e]"
            filter_var_opex = None
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_e/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = ["Nominal investment (*total) [MEUR/MW_e]"]

        elif technology == "Photovoltaic_distributed_commercial":
            self.tec_type = "RES"
            filter_tec = "22 Rooftop PV comm.&industrial"
            filter_cf = "Average annual full-load hours [MWh_e/MW_e]"
            filter_var_opex = None
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_e/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = ["Nominal investment (*total) [MEUR/MW_e]"]

        elif technology == "Photovoltaic_distributed_residential":
            self.tec_type = "RES"
            filter_tec = "22 Rooftop PV residential"
            filter_cf = "Average annual full-load hours [MWh_e/MW_e]"
            filter_var_opex = None
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_e/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = ["Nominal investment (*total) [MEUR/MW_e]"]

        elif technology == "air_sourced_1MW":
            self.tec_type = "HP"
            filter_tec = "40 Comp. hp, airsource 1 MW"
            filter_var_opex = "Variable O&M (other O&M) [EUR/MWh_h]"
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_h/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = ["Nominal investment (*total) [MEUR/MW_h]"]
            filter_cop = "Heat efficiency (net, name plate) []"

        elif technology == "air_sourced_3MW":
            self.tec_type = "HP"
            filter_tec = "40 Comp. hp, airsource 3 MW"
            filter_var_opex = "Variable O&M (other O&M) [EUR/MWh_h]"
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_h/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = ["Nominal investment (*total) [MEUR/MW_h]"]
            filter_cop = "Heat efficiency (net, name plate) []"

        elif technology == "air_sourced_10MW":
            self.tec_type = "HP"
            filter_tec = "40 Comp. hp, airsource 10 MW"
            filter_var_opex = "Variable O&M (other O&M) [EUR/MWh_h]"
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_h/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = ["Nominal investment (*total) [MEUR/MW_h]"]
            filter_cop = "Heat efficiency (net, name plate) []"

        elif technology == "seawater_20MW":
            self.tec_type = "HP"
            filter_tec = "40 Comp. hp, seawater 20 MW"
            filter_var_opex = "Variable O&M (other O&M) [EUR/MWh_h]"
            filter_fixed_opex = "Fixed O&M (*total) [EUR/MW_h/y]"
            filter_lifetime = "Technical lifetime [years]"
            filter_capex = ["Nominal investment (*total) [MEUR/MW_h]"]
            filter_cop = "Heat efficiency (net, name plate) []"

        else:
            raise ValueError("Technology not available")

        # Filter data
        technology_data = all_data[all_data["ws"] == filter_tec]
        technology_data = technology_data[technology_data["est"] == "ctrl"]

        if self.tec_type == "RES":
            self.other["cf"] = technology_data[technology_data["par"] == filter_cf]
            self.other["cf"]["val"] = self.other["cf"]["val"] / 8760
        elif self.tec_type == "HP":
            self.other["cop"] = technology_data[technology_data["par"] == filter_cop]

        self.capex_meur_per_mw = technology_data[
            technology_data["par"].isin(filter_capex)
        ]
        self.opex_fixed_eur_per_mw_per_year = technology_data[
            technology_data["par"] == filter_fixed_opex
        ]
        self.opex_variable_eur_per_mwh = technology_data[
            technology_data["par"] == filter_var_opex
        ]
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

        if self.tec_type == "HP":
            capacity_correction = (
                1
                / self.other["cop"][
                    self.other["cop"]["year"] == options["projection_year"]
                ]["val"].mean()
            )
        else:
            capacity_correction = 1

        # Capex
        capex_meur_per_mw = self.capex_meur_per_mw[
            self.capex_meur_per_mw["year"] == options["projection_year"]
        ]
        if len(capex_meur_per_mw) == 0:
            raise ValueError("projection_year is not available")

        self.unit_capex = capex_meur_per_mw["val"].sum() * 1e3 * capacity_correction

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
            opex_fixed_eur_per_mw_per_year["val"].sum() / 1000 * capacity_correction
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
            self.opex_var = (
                opex_var_eur_per_mwh["val"].sum() / 1000 * capacity_correction
            )
        else:
            raise ValueError("Something went wrong with variable opex calculation")

        # Capacity factor
        if self.tec_type == "RES":
            cf = self.other["cf"][
                self.other["cf"]["year"] == options["projection_year"]
            ]
            if len(cf) != 1:
                raise ValueError("Something went wrong with fixed opex calculation")
            self.cf = cf["val"].sum()
        elif self.tec_type == "HP":
            self.cf = 0.5

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
