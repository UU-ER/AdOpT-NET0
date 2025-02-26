import pandas as pd
from pathlib import Path
import numpy as np


def learning_factor(learning_rate, foak_scale, capacity):
    """
    LEARNING_FACTOR: This function computes the cost multiplier (Learning Factor) associated
    with cost reductions from learning-by-doing. The learning rate is used to calculate the learning exponent 'b',
    which is then used to calculate the learning factor. This learning factor serves as a cost multiplier associated with the learning rate.
    It is multiplied with "first-of-a-kind" (FOAK) costs in order to get "nth-of-a-kind" costs.

    Parameters:
    learning_rate (double or array): The learning rate of the technology component,
                         expressed as a fraction (e.g 0.05 for 5%).
    foak_scale (double): The technology FOAK scale in tCO2/yr.
    capacity (double): The current installed "nth-of-a-kind" capacity of interest in the model.

    Returns:
    lf (double or array): The learning factor which can be multiplied by FOAK costs
                  to get nth-of-a-kind costs. The size of the output matches
                  the size of the "learning_rate" input.

    Note: If capacity and FOAK scale are the same, then the learning factor equals 1.
        This is necessary to calculate initial cost without learning.
    """

    x = capacity / foak_scale  # ratio of capacity to FOAK scale
    b = -np.log((1 - learning_rate)) / np.log(2)  # learning exponent

    lf = np.power(x, -b)  # compute learning factor as x^b

    return lf


class Dac_sievert:
    """
    Calculates cost of direct air capture (DAC) for different DAC technologies.

    dac_technology can be "LS" (liquid solvent), "SS" (solid solvent), or "CaO" (calcium loop)
    The units of the cost parameters are as follows:

    - unit_capex is in [selected currency] / t(CO2) / h
    - opex_fix is in % of annualized capex using the given discount rate and lifetime
    - opex_var is in [selected currency] / t(CO2) and excludes energy costs
    """

    def __init__(self, dac_technology, gas_price_year):
        # DAC SPECIFIC
        # Read universal input
        self.dac_technology = dac_technology

        dac_input_path = Path(__file__).parent.parent.parent / Path(
            "./data/technologies/dac_sievert/inputs_DACS_single.xlsx"
        )
        universal = pd.read_excel(dac_input_path, "Universal_Inputs", index_col=0)
        technology = pd.read_excel(
            dac_input_path, sheet_name="Technology_Inputs", index_col=0
        )
        epc_cost = pd.read_excel(dac_input_path, sheet_name="EPC_Cost", index_col=0)
        self.gas_prices = pd.read_excel(
            dac_input_path, sheet_name="Heat_Prices", index_col=0
        )

        self.gas_price_year = gas_price_year

        # Write it to class
        self.lifetime = universal.loc["plant_life", "Value"]

        self.universal = universal.loc[
            [
                "epc_factor",
                "project_contingency_factor",
                "owners_cost",
                "spare_parts_cost",
                "startup_capital",
                "startup_labor",
                "startup_chemicals",
                "startup_fuel",
                "water_cost",
                "maintenance_factor",
                "indirect_labour_factor",
                "insurance_factor",
                "taxes_fees_factor",
                "operator_salary",
                "productivity_factor",
                "learning_rate_opex",
            ]
        ]

        self.technology = technology.loc[
            [
                "initial_scale",
                "foak_scale",
                "co2_purity",
                "ratio_co2_compressed_to_captured",
                "temperature_heat",
                "water_requirement",
                "chemicals_cost",
                "process_contingency_factor",
                "learning_rate_system",
                "employees",
                "heat_requirement_gas",
                "learning_rate_lcor",
                "learning_rate_installed_cost",
                "learning_rate_epc",
                "learning_rate_process_contingency",
                "learning_rate_project_contingency",
                "learning_rate_startup_cost",
            ]
        ]

        self.epc_cost_components = epc_cost

        # Cost components
        self.cost_total_plant = None
        self.cost_start_up = None
        self.cost_total_overnight = None

    def calculate_cost(self, discount_rate, cumulative_capacity):
        """
        Calculates cost (Capex, fixed opex, variable opex) and corrects for exchange rates and inflation

        If cumulative_capacity=0, it uses first-of-a-kind capacity

        The following adjustments have been made compared to Sievert et al. (2024):
        - We do not correct the first-of-a-kind scale based on the co2 intensity of heat and electricity
        - Opex var does not include energy and downstream transport/storage cost

        :param float cumulative_capacity: total global installed capturing capacity in t/a. Determines the learning rates
        """
        self.discount_rate = discount_rate
        if cumulative_capacity == 0:
            cumulative_capacity = self.technology.loc["foak_scale", self.dac_technology]

        self._calculate_capex(cumulative_capacity)
        self._calculate_opex_var(cumulative_capacity)
        self._calculate_opex_fix(cumulative_capacity)

        return {
            "unit_capex": self.unit_capex,
            "opex_fix": self.opex_fix,
            "opex_var": self.opex_var,
        }

    def _calculate_capex(self, cumulative_capacity):
        """
        Calculates capex

        :param float cumulative_capacity: total global installed capturing capacity in t/a. Determines the learning rates
        """
        foak_scale = self.technology.loc["foak_scale", self.dac_technology]

        self.cost_total_plant = self._calculate_total_plant_cost(cumulative_capacity)
        self.cost_start_up = self._calculate_startup_costs(cumulative_capacity)
        self.cost_total_overnight = self.cost_total_plant + self.cost_start_up
        self.unit_capex = self.cost_total_overnight / foak_scale

    def _calculate_opex_var(self, cumulative_capacity):
        """
        Calculates variable opex in USD/t

        :param float cumulative_capacity: total global installed capturing capacity in t/a. Determines the learning rates
        """

        foak_scale = self.technology.loc["foak_scale", self.dac_technology]

        # Use learning rate to create learning factor
        lr_opex = self.universal.loc[
            "learning_rate_opex", "Value"
        ]  # retrieves learning rate for variable OPEX from input data
        lf_opex = learning_factor(lr_opex, foak_scale, cumulative_capacity)

        water_cost = (
            self.technology.loc["water_requirement", self.dac_technology]
            * self.universal.loc["water_cost", "Value"]
            * lf_opex
        )
        chemicals_cost = (
            self.technology.loc["chemicals_cost", self.dac_technology] * lf_opex
        )

        self.opex_var = water_cost + chemicals_cost

    def _calculate_opex_fix(self, cumulative_capacity):
        """
        Calculates fixed opex as fraction of annualized capex

        :param float cumulative_capacity: total global installed capturing capacity in t/a. Determines the learning rates
        """
        foak_scale = self.technology.loc["foak_scale", self.dac_technology]
        total_plant_cost_unlearned = self._calculate_total_plant_cost(foak_scale)

        labor_cost = self._annual_labor_cost()
        insurance_cost = (
            self.universal.loc["insurance_factor", "Value"] * total_plant_cost_unlearned
        )
        tax_fees_cost = (
            self.universal.loc["taxes_fees_factor", "Value"]
            * total_plant_cost_unlearned
        )

        lr_system = self.technology.loc["learning_rate_system", self.dac_technology]
        lf_system = learning_factor(lr_system, foak_scale, cumulative_capacity)

        fom = (
            (labor_cost + insurance_cost + tax_fees_cost)
            * lf_system
            / foak_scale
            / 8760
        )

        crf = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.lifetime
            / ((1 + self.discount_rate) ** self.lifetime - 1)
        )
        annualized_capex = crf * self.unit_capex

        self.opex_fix = fom / annualized_capex

    def _calculate_total_plant_cost(self, cumulative_capacity):
        """
        Calculates total_plant_cost

        :param float cumulative_capacity: total global installed capturing capacity in t/a. Determines the learning rates
        :return: total plant costs
        :rtype: float
        """

        cost_epc = self._calculate_epc_cost(cumulative_capacity)
        cost_project_contingency = self._calculate_project_contingency(
            cumulative_capacity
        )
        cost_process_contingency = self._calculate_process_contingency(
            cumulative_capacity
        )
        return 10**6 * (cost_epc + cost_project_contingency + cost_process_contingency)

    def _calculate_startup_costs(self, cumulative_capacity):
        """
        Calculates startup_costs

        Sum of owners, spare parts, startup capital costs, labor costs, fuel costs, and chemicals costs

        :param float cumulative_capacity: total global installed capturing capacity in t/a. Determines the learning rates
        :return: startup costs
        :rtype: float
        """
        foak_scale = self.technology.loc["foak_scale", self.dac_technology]
        total_plant_cost_unlearned = self._calculate_total_plant_cost(foak_scale)

        # Start-up learning rate
        lr_startup = self.technology.loc[
            "learning_rate_startup_cost", self.dac_technology
        ]
        lf_startup = learning_factor(lr_startup, foak_scale, cumulative_capacity)
        gas_price = self.gas_prices.loc["NG", self.gas_price_year]

        owners_cost = (
            self.universal.loc["owners_cost", "Value"] * total_plant_cost_unlearned
        )
        spare_parts_cost = (
            self.universal.loc["spare_parts_cost", "Value"] * total_plant_cost_unlearned
        )
        startup_capital = (
            self.universal.loc["startup_capital", "Value"] * total_plant_cost_unlearned
        )
        startup_labor = (
            self._annual_labor_cost() * self.universal.loc["startup_labor", "Value"]
        )
        startup_fuel = (
            gas_price
            * self.technology.loc["heat_requirement_gas", self.dac_technology]
            * foak_scale
            * self.universal.loc["startup_fuel", "Value"]
        )
        startup_chemicals = (
            self.universal.loc["startup_chemicals", "Value"]
            * self.technology.loc["chemicals_cost", self.dac_technology]
            * foak_scale
        )

        return (
            owners_cost
            + spare_parts_cost
            + startup_capital
            + startup_labor
            + startup_fuel
            + startup_chemicals
        ) * lf_startup

    def _calculate_epc_cost(self, cumulative_capacity):
        """
        Calculates initial engineering, procurement, and construction (EPC) cost [$] for a technology.

        Sum of direct material costs + epc adder

        :param float cumulative_capacity: total global installed capturing capacity in t/a. Determines the learning rates
        :return: epc costs
        :rtype: float
        """
        epc_cost = self.epc_cost_components[
            self.epc_cost_components["Technology"] == self.dac_technology
        ]

        # Extract FOAK scale and initial scale
        foak_scale = self.technology.loc["foak_scale", self.dac_technology]
        initial_scale = self.technology.loc["initial_scale", self.dac_technology]

        # Extract epc_factor
        epc_factor = self.universal.loc["epc_factor", "Value"]

        scaled_direct_materials_cost = epc_cost.loc[
            :, "Direct_materials_cost"
        ] * np.power(foak_scale / initial_scale, epc_cost.loc[:, "Exponent"])
        learning_rate = epc_cost["Learning_rate"]
        lf = learning_factor(learning_rate, foak_scale, cumulative_capacity)

        epc_cost.loc[:, "Learned_direct_materials_cost"] = (
            scaled_direct_materials_cost * lf
        )

        cost = epc_cost["Learned_direct_materials_cost"].sum()

        lr_epc = self.technology.loc["learning_rate_epc", self.dac_technology]
        lf_epc = learning_factor(lr_epc, foak_scale, cumulative_capacity)
        unlearned_cost = scaled_direct_materials_cost.sum()
        epc_adder = unlearned_cost * epc_factor * lf_epc

        # Scale by epc_factor to calculate the total initial EPC cost [$]
        return cost + epc_adder

    def _calculate_project_contingency(self, cumulative_capacity):
        """
        Calculates project contingency.

        :param float cumulative_capacity: total global installed capturing capacity in t/a. Determines the learning rates
        :return: project contingency costs
        :rtype: float
        """

        foak_scale = self.technology.loc["foak_scale", self.dac_technology]

        lr_project_contingency = self.technology.loc[
            "learning_rate_project_contingency", self.dac_technology
        ]
        lf_project_contingency = learning_factor(
            lr_project_contingency, foak_scale, cumulative_capacity
        )

        epc_unlearned = self._calculate_epc_cost(foak_scale)

        return (
            self.universal.loc["project_contingency_factor", "Value"]
            * epc_unlearned
            * lf_project_contingency
        )

    def _calculate_process_contingency(self, cumulative_capacity):
        """
        Calculates process contingency.

        :param float cumulative_capacity: total global installed capturing capacity in t/a. Determines the learning rates
        :return: process contingency costs
        :rtype: float
        """

        foak_scale = self.technology.loc["foak_scale", self.dac_technology]

        lr_process_contingency = self.technology.loc[
            "learning_rate_process_contingency", self.dac_technology
        ]
        lf_process_contingency = learning_factor(
            lr_process_contingency, foak_scale, cumulative_capacity
        )

        epc_unlearned = self._calculate_epc_cost(foak_scale)

        return (
            self.technology.loc["process_contingency_factor", self.dac_technology]
            * epc_unlearned
            * lf_process_contingency
        )

    def _annual_labor_cost(self):
        """
        Calculates annual labor costs

        :return: annual labor costs
        :rtype: float
        """
        # Load labor inputs
        operator_salary = self.universal.loc["operator_salary", "Value"]
        productivity_factor = self.universal.loc["productivity_factor", "Value"]
        employees = self.technology.loc["employees", self.dac_technology]
        maintenance_factor = self.universal.loc["maintenance_factor", "Value"]
        indirect_labour_factor = self.universal.loc["indirect_labour_factor", "Value"]
        foak_scale = self.technology.loc["foak_scale", self.dac_technology]

        # Compute total plant cost
        total_plant_cost_unlearned = self._calculate_total_plant_cost(foak_scale)

        # Calculate various components of labor cost
        direct_labor_cost = employees * operator_salary * productivity_factor
        maintenance_cost = maintenance_factor * total_plant_cost_unlearned
        indirect_labor_cost = indirect_labour_factor * (
            direct_labor_cost + maintenance_cost
        )

        return direct_labor_cost + indirect_labor_cost + maintenance_cost

    def calculate_levelized_cost(self, capacity_factor):
        """
        Calculates levelized costs of co2 capture (excluding energy costs).

        :param float capacity_factor: Assumed capacity factor of plant
        :return: lcoc
        :rtype: float
        """

        crf = (
            self.discount_rate
            * (1 + self.discount_rate) ** self.lifetime
            / ((1 + self.discount_rate) ** self.lifetime - 1)
        )

        lcoc = (
            self.unit_capex * crf * (1 + self.opex_fix)
            + capacity_factor * self.opex_var
        ) / (capacity_factor)
        return lcoc
