import pandas as pd
import pyomo.environ as pyo
import h5py
import numpy as np

from ..technology import Technology
from src.components.utilities import annualize, set_discount_rate, Parameters


class Sink(Technology):
    """
    This model resembles a permanent storage technology (sink). It takes energy and a main carrier (e.g. CO2 etc)
    as inputs, and it has no output.

    **Parameter declarations:**

    - Min Size
    - Max Size
    - Unit CAPEX storage size (annualized from given data on up-front CAPEX, lifetime and discount rate)
    - Unit CAPEX injection capacity (annualized from given data on up-front CAPEX, lifetime and discount rate)

    **Variable declarations:**

    - Storage level in :math:`t`: :math:`E_t`
    - Injection capacity
    - CAPEX storage size
    - CAPEX injection capacity

    **Constraint declarations:**

    - Maximal injection rate:

      .. math::
        Input_{t} \leq injCapacity

    - Maximal injection capacity:

      .. math::
        injCapacity \leq injRateMax

    - Size constraint:

      .. math::
        E_{t} \leq storageSize

    - Storage level calculation:

      .. math::
        E_{t} = E_{t-1} + Input_{t}

    - If an energy consumption for the injection is given, the respective carrier input is:

      .. math::
        Input_{t, car} = cons_{car, in} Input_{t}

    - CAPEX is given by two contributions

        .. math::
            CAPEX_{storSize} = Size_{storSize} * UnitCost_{storSize}
            CAPEX_{injCapacity} = injCapacity * UnitCost_{injCapacity}

    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.options.emissions_based_on = "input"
        self.info.main_input_carrier = tec_data["Performance"]["main_input_carrier"]
        self.flexibility_data = tec_data["Flexibility"]

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Calculate input bounds and select new capex model

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        super(Sink, self).fit_technology_performance(climate_data, location)

        time_steps = len(climate_data)

        # Input Bounds
        for car in self.info.input_carrier:
            if car == self.info.main_input_carrier:
                self.bounds["input"][car] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps))
                        * self.flexibility_data["injection_rate_max"],
                    )
                )
            else:
                if "energy_consumption" in self.parameters.unfitted_data["performance"]:
                    energy_consumption = self.parameters.unfitted_data["performance"][
                        "energy_consumption"
                    ]
                    self.bounds["input"][car] = np.column_stack(
                        (
                            np.zeros(shape=(time_steps)),
                            np.ones(shape=(time_steps))
                            * self.flexibility_data["injection_rate_max"]
                            * energy_consumption["in"][car],
                        )
                    )

        # For a flexibly optimized storage technology (i.e., not a fixed P-E ratio), an adapted CAPEX function is used
        # to account for charging and discharging capacity costs.
        if self.flexibility_data["injection_capacity_is_decision_var"]:
            self.economics.capex_model = 4

        self.coeff.time_independent["injection_rate_max"] = self.flexibility_data[
            "injection_rate_max"
        ]
        if "energy_consumption" in self.parameters.unfitted_data["performance"]:
            self.coeff.time_independent["energy_consumption"] = (
                self.parameters.unfitted_data["performance"]["energy_consumption"]
            )

    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Construct SINK constraints

        Adds constraints to technology blocks for tec_type SINK, resembling a permanent storage technology

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """

        super(Sink, self).construct_tech_model(b_tec, data, set_t_full, set_t_clustered)

        # DATA OF TECHNOLOGY
        config = data["config"]
        c_ti = self.coeff.time_independent
        dynamics = self.coeff.dynamics

        # Sotrage level and injection capacity decision variables
        b_tec.var_storage_level = pyo.Var(
            set_t_full,
            domain=pyo.NonNegativeReals,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )
        b_tec.var_injection_capacity = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(0, c_ti["injection_rate_max"]),
        )

        if self.flexibility_data["injection_capacity_is_decision_var"]:
            b_tec = self._define_sink_capex(b_tec, data)

        # Maximum storage level constraint
        def init_size_constraint(const, t):
            # storageLevel <= storSize
            return b_tec.var_storage_level[t] <= b_tec.var_size

        b_tec.const_size = pyo.Constraint(set_t_full, rule=init_size_constraint)

        # Constraint storage level
        if (
            config["optimization"]["typicaldays"]["N"]["value"] != 0
            and not self.options.modelled_with_full_res
        ):

            def init_storage_level(const, t):
                # storageLevel[1] <= injRate[1]
                if t == 1:
                    return (
                        b_tec.var_storage_level[t]
                        == self.input[
                            self.sequence[t - 1], self.info.main_input_carrier
                        ]
                    )
                else:
                    # storageLevel[t] <= storageLevel[t-1]+injRate[t]
                    return (
                        b_tec.var_storage_level[t]
                        == b_tec.var_storage_level[t - 1]
                        + self.input[self.sequence[t - 1], self.info.main_input_carrier]
                    )

            b_tec.const_storage_level = pyo.Constraint(
                set_t_full, rule=init_storage_level
            )

        else:

            def init_storage_level(const, t):
                # storageLevel[1] <= injRate[1]
                if t == 1:
                    return (
                        b_tec.var_storage_level[t]
                        == self.input[t, self.info.main_input_carrier]
                    )
                else:
                    # storageLevel[t] <= storageLevel[t-1]+injRate[t]
                    return (
                        b_tec.var_storage_level[t]
                        == b_tec.var_storage_level[t - 1]
                        + self.input[t, self.info.main_input_carrier]
                    )

            b_tec.const_storage_level = pyo.Constraint(
                set_t_full, rule=init_storage_level
            )

        # Maximal injection rate
        def init_maximal_injection(const, t):
            # input[t] <= injectionCapacity
            return (
                self.input[t, self.info.main_input_carrier]
                <= b_tec.var_injection_capacity
            )

        b_tec.const_max_injection = pyo.Constraint(
            self.set_t, rule=init_maximal_injection
        )

        # if injection rates are fixed/ flexible:
        def init_max_capacity_injection(const):
            if self.flexibility_data["injection_capacity_is_decision_var"]:
                # injectionCapacity <= injectionRateMax
                return b_tec.var_injection_capacity <= c_ti["injection_rate_max"]
            else:
                # injectionCapacity == injectionRateMax
                return b_tec.var_injection_capacity == c_ti["injection_rate_max"]

        b_tec.const_max_injection_cap = pyo.Constraint(rule=init_max_capacity_injection)

        # Energy consumption for injection
        if "energy_consumption" in c_ti:
            energy_consumption = c_ti["energy_consumption"]
            if "in" in energy_consumption:
                b_tec.set_energyconsumption_carriers_in = pyo.Set(
                    initialize=energy_consumption["in"].keys()
                )

                def init_energyconsumption_in(const, t, car):
                    # energyInput[t] = mainInput[t] * energyConsumption
                    return (
                        self.input[t, car]
                        == self.input[t, self.info.main_input_carrier]
                        * energy_consumption["in"][car]
                    )

                b_tec.const_energyconsumption_in = pyo.Constraint(
                    self.set_t,
                    b_tec.set_energyconsumption_carriers_in,
                    rule=init_energyconsumption_in,
                )

        # RAMPING RATES
        if "ramping_time" in dynamics:
            if not dynamics["ramping_time"] == -1:
                b_tec = self._define_ramping_rates(b_tec)

        return b_tec

    def _define_sink_capex(self, b_tec, data: dict):
        """

        Construct CAPEX of SINK constraints

        Adds constraints to technology blocks for tec_type SINK to calculate the CAPEX

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :return: pyomo block with technology model
        """

        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )
        flexibility = self.flexibility_data
        c_ti = self.coeff.time_independent

        # CAPEX PARAMETERS
        b_tec.para_unit_capex_injection_cap = pyo.Param(
            domain=pyo.Reals,
            initialize=flexibility["capex_injection_cap"],
            mutable=True,
        )
        b_tec.para_unit_capex_stor_size = pyo.Param(
            domain=pyo.Reals,
            initialize=economics.capex_data["unit_capex"],
            mutable=True,
        )

        b_tec.para_unit_capex_injection_cap_annual = pyo.Param(
            domain=pyo.Reals,
            initialize=(annualization_factor * b_tec.para_unit_capex_injection_cap),
            mutable=True,
        )
        b_tec.para_unit_capex_stor_size_annual = pyo.Param(
            domain=pyo.Reals,
            initialize=(annualization_factor * b_tec.para_unit_capex_stor_size),
            mutable=True,
        )

        # BOUNDS
        max_capex_injection_cap = (
            b_tec.para_unit_capex_injection_cap_annual * c_ti["injection_rate_max"]
        )
        max_capex_stor_size = (
            b_tec.para_unit_capex_stor_size_annual * b_tec.para_size_max
        )

        b_tec.var_capex_injection_cap = pyo.Var(
            domain=pyo.NonNegativeReals, bounds=(0, max_capex_injection_cap)
        )
        b_tec.var_capex_stor_size = pyo.Var(
            domain=pyo.NonNegativeReals, bounds=(0, max_capex_stor_size)
        )

        # CAPEX constraints
        # CAPEXinjection = injCapacity * UnitCost_injCapacity
        b_tec.const_capex_injection_cap = pyo.Constraint(
            expr=b_tec.var_injection_capacity
            * b_tec.para_unit_capex_injection_cap_annual
            == b_tec.var_capex_injection_cap
        )
        # CAPEXstorSize = storSize * UnitCost_storSize
        b_tec.const_capex_stor_size = pyo.Constraint(
            expr=b_tec.var_size * b_tec.para_unit_capex_stor_size_annual
            == b_tec.var_capex_stor_size
        )
        b_tec.const_capex_aux = pyo.Constraint(
            expr=b_tec.var_capex_injection_cap + b_tec.var_capex_stor_size
            == b_tec.var_capex_aux
        )

        return b_tec

    def write_results_tec_operation(self, h5_group: h5py.Group, model_block: pyo.Block):
        """
        Function to report results of technologies operations after optimization

        :param Block b_tec: technology model block
        :param h5py.Group h5_group: technology model block
        """
        super(Sink, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "storage_level",
            data=[model_block.var_storage_level[t].value for t in self.set_t_full],
        )

    def write_results_tec_design(self, h5_group: h5py.Group, model_block: pyo.Block):
        """
        Function to report results of technologies design after optimization

        :param  h5py.Group h5_group: h5 file structure
        :param Block b_tec: technology model block
        """
        super(Sink, self).write_results_tec_design(h5_group, model_block)

        if self.flexibility_data["injection_capacity_is_decision_var"]:
            h5_group.create_dataset(
                "injection_capacity", data=[model_block.var_injection_capacity.value]
            )
            h5_group.create_dataset(
                "capex_injection_cap", data=[model_block.var_capex_injection_cap.value]
            )
            h5_group.create_dataset(
                "capex_stor_size", data=[model_block.var_capex_stor_size.value]
            )

    def _define_ramping_rates(self, b_tec):
        """
        Constraints the inputs for a ramping rate

        :param Block b_tec: technology model block
        :return: technology Block with the constraints for the dynamic behaviour
        """
        dynamics = self.coeff.dynamics

        ramping_time = dynamics["ramping_time"]

        # Calculate ramping rates
        if "ref_size" in dynamics and not dynamics["ref_size"] == -1:
            ramping_rate = dynamics["ref_size"] / ramping_time
        else:
            ramping_rate = b_tec.var_size / ramping_time

        def init_ramping_down_rate_input(const, t):
            if t > 1:
                # -rampingRate <= input[t] - input[t-1]
                return (
                    -ramping_rate
                    <= self.input[t, self.info.main_input_carrier]
                    - self.input[t - 1, self.info.main_input_carrier]
                )
            else:
                return pyo.Constraint.Skip

        b_tec.const_ramping_down_rate_input = pyo.Constraint(
            self.set_t, rule=init_ramping_down_rate_input
        )

        def init_ramping_up_rate_input(const, t):
            if t > 1:
                # input[t] - input[t-1] <= rampingRate
                return (
                    self.input[t, self.info.main_input_carrier]
                    - self.input[t - 1, self.info.main_input_carrier]
                    <= ramping_rate
                )
            else:
                return pyo.Constraint.Skip

        b_tec.const_ramping_up_rate_input = pyo.Constraint(
            self.set_t, rule=init_ramping_up_rate_input
        )

        return b_tec
