from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import numpy as np

from ..utilities import FittedPerformance
from ..technology import Technology
from src.components.utilities import annualize, set_discount_rate


class Sink(Technology):
    """
    This model resembles a permanent storage technology (sink). It takes energy and a main carrier (e.g. CO2, H2 etc)
    as inputs, and it has no output.


    **Variable declarations:**

    - Storage level in :math:`t`: :math:`E_t`

    **Constraint declarations:**

    - Maximal injection rate:

      .. math::
        Input_{t} \leq Inj_rate

    - Size constraint:

      .. math::
        E_{t} \leq Size

    - Storage level calculation:

      .. math::
        E_{t} = E_{t-1} + Input_{t}

    - If an energy consumption for the injection is given, the respective carrier input is:

      .. math::
        Input_{t, car} = cons_{car, in} Input_{t}

    """

    def __init__(self, tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()
        self.flexibility_data = tec_data["Flexibility"]

    def fit_technology_performance(self, climate_data, location):
        """
        Fits conversion technology type SINK and returns fitted parameters as a dict

        :param node_data: contains data on demand, climate data, etc.
        """

        time_steps = len(climate_data)

        # Main carrier (carrier to be stored)
        self.main_car = self.performance_data["main_input_carrier"]

        # Input Bounds
        for car in self.performance_data["input_carrier"]:
            if car == self.performance_data["main_input_carrier"]:
                self.fitted_performance.bounds["input"][car] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps))
                        * self.flexibility_data["injection_rate_max"],
                    )
                )
            else:
                if "energy_consumption" in self.performance_data["performance"]:
                    energy_consumption = self.performance_data["performance"][
                        "energy_consumption"
                    ]
                    self.fitted_performance.bounds["input"][car] = np.column_stack(
                        (
                            np.zeros(shape=(time_steps)),
                            np.ones(shape=(time_steps))
                            * self.flexibility_data["injection_rate_max"]
                            * energy_consumption["in"][car],
                        )
                    )

        # For a flexibly optimized storage technology (i.e., not a fixed P-E ratio), an adapted CAPEX function is used
        # to account for charging and discharging capacity costs.
        if self.flexibility_data["power_energy_ratio"] == "flex":
            self.economics.capex_model = 4
        if self.flexibility_data["power_energy_ratio"] not in ["flex", "fixed"]:
            raise Warning(
                "power_energy_ratio should be either flexible ('flex') or fixed ('fixed')"
            )

        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 0

    def construct_tech_model(self, b_tec, data, set_t, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type SINK, resembling a permanent storage technology

        :param b_tec:
        :param energyhub:
        :return: b_tec
        """

        super(Sink, self).construct_tech_model(b_tec, data, set_t, set_t_clustered)

        # DATA OF TECHNOLOGY
        config = data["config"]

        # Additional decision variables
        b_tec.var_storage_level = Var(
            set_t,
            domain=NonNegativeReals,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )
        b_tec.var_capacity_injection = Var(
            domain=NonNegativeReals,
            bounds=(0, self.flexibility_data["injection_rate_max"]),
        )

        if self.flexibility_data["power_energy_ratio"] == "flex":
            b_tec = self._define_sink_capex(b_tec, data)

        # Size constraint
        def init_size_constraint(const, t):
            return b_tec.var_storage_level[t] <= b_tec.var_size

        b_tec.const_size = Constraint(set_t, rule=init_size_constraint)

        # Constraint storage level
        if (
            config["optimization"]["typicaldays"]["N"]["value"] != 0
            and not self.modelled_with_full_res
        ):

            def init_storage_level(const, t):
                if t == 1:
                    return (
                        b_tec.var_storage_level[t]
                        == self.input[self.sequence[t - 1], self.main_car]
                    )
                else:
                    return (
                        b_tec.var_storage_level[t]
                        == b_tec.var_storage_level[t - 1]
                        + self.input[self.sequence[t - 1], self.main_car]
                    )

            b_tec.const_storage_level = Constraint(set_t, rule=init_storage_level)

        else:

            def init_storage_level(const, t):
                if t == 1:
                    return b_tec.var_storage_level[t] == self.input[t, self.main_car]
                else:
                    return (
                        b_tec.var_storage_level[t]
                        == b_tec.var_storage_level[t - 1] + self.input[t, self.main_car]
                    )

            b_tec.const_storage_level = Constraint(set_t, rule=init_storage_level)

        # Maximal injection rate
        def init_maximal_injection(const, t):
            return self.input[t, self.main_car] <= b_tec.var_capacity_injection

        b_tec.const_max_charge = Constraint(self.set_t, rule=init_maximal_injection)

        # if injection rates are fixed/ flexible:
        def init_max_capacity_charge(const):
            if self.flexibility_data["power_energy_ratio"] == "fixed":
                return (
                    b_tec.var_capacity_injection
                    == self.flexibility_data["injection_rate_max"]
                )
            else:
                return (
                    b_tec.var_capacity_injection
                    <= self.flexibility_data["injection_rate_max"]
                )

        b_tec.const_max_cap_charge = Constraint(rule=init_max_capacity_charge)

        # Energy consumption for injection
        if "energy_consumption" in self.performance_data["performance"]:
            energy_consumption = self.performance_data["performance"][
                "energy_consumption"
            ]
            if "in" in energy_consumption:
                b_tec.set_energyconsumption_carriers_in = Set(
                    initialize=energy_consumption["in"].keys()
                )

                def init_energyconsumption_in(const, t, car):
                    return (
                        self.input[t, car]
                        == self.input[t, self.main_car] * energy_consumption["in"][car]
                    )

                b_tec.const_energyconsumption_in = Constraint(
                    self.set_t,
                    b_tec.set_energyconsumption_carriers_in,
                    rule=init_energyconsumption_in,
                )

        # RAMPING RATES
        if "ramping_rate" in self.performance_data:
            if not self.performance_data["ramping_rate"] == -1:
                b_tec = self._define_ramping_rates(b_tec)

        return b_tec

    def _define_sink_capex(self, b_tec, data):

        flexibility = self.flexibility_data
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )

        # CAPEX PARAMETERS
        b_tec.para_unit_capex_injection_cap = Param(
            domain=Reals, initialize=flexibility["capex_injection_cap"], mutable=True
        )
        b_tec.para_unit_capex_sink_cap = Param(
            domain=Reals, initialize=economics.capex_data["unit_capex"], mutable=True
        )

        b_tec.para_unit_capex_injection_cap_annual = Param(
            domain=Reals,
            initialize=(annualization_factor * b_tec.para_unit_capex_injection_cap),
            mutable=True,
        )
        b_tec.para_unit_capex_sink_cap_annual = Param(
            domain=Reals,
            initialize=(annualization_factor * b_tec.para_unit_capex_sink_cap),
            mutable=True,
        )

        # BOUNDS
        max_capex_injection_cap = (
            b_tec.para_unit_capex_injection_cap_annual
            * flexibility["injection_rate_max"]
        )
        max_capex_sink_cap = b_tec.para_unit_capex_sink_cap_annual * b_tec.para_size_max

        b_tec.var_capex_injection_cap = Var(
            domain=NonNegativeReals, bounds=(0, max_capex_injection_cap)
        )
        b_tec.var_capex_sink_cap = Var(
            domain=NonNegativeReals, bounds=(0, max_capex_sink_cap)
        )

        # CAPEX constraint
        b_tec.const_capex_injection_cap = Constraint(
            expr=b_tec.var_capacity_injection
            * b_tec.para_unit_capex_injection_cap_annual
            == b_tec.var_capex_injection_cap
        )
        b_tec.const_capex_sink_cap = Constraint(
            expr=b_tec.var_size * b_tec.para_unit_capex_sink_cap_annual
            == b_tec.var_capex_sink_cap
        )
        b_tec.const_capex_aux = Constraint(
            expr=b_tec.var_capex_injection_cap + b_tec.var_capex_sink_cap
            == b_tec.var_capex_aux
        )

        return b_tec

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        super(Sink, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "storage_level",
            data=[model_block.var_storage_level[t].value for t in self.set_t_full],
        )

    def write_results_tec_design(self, h5_group, model_block):

        super(Sink, self).write_results_tec_design(h5_group, model_block)

        if self.flexibility_data["power_energy_ratio"] == "flex":
            h5_group.create_dataset(
                "capacity_injection", data=[model_block.var_capacity_injection.value]
            )
            h5_group.create_dataset(
                "capex_injection_cap", data=[model_block.var_capex_injection_cap.value]
            )
            h5_group.create_dataset(
                "capex_sink_cap", data=[model_block.var_capex_sink_cap.value]
            )

    def _define_ramping_rates(self, b_tec):
        """
        Constraints the inputs for a ramping rate. Implemented for input and output

        :param b_tec: technology model block
        :return:
        """
        ramping_rate = self.performance_data["ramping_rate"]

        def init_ramping_down_rate_input(const, t):
            if t > 1:
                return -ramping_rate <= sum(
                    self.input[t, car_input] - self.input[t - 1, car_input]
                    for car_input in b_tec.set_input_carriers
                )
            else:
                return Constraint.Skip

        b_tec.const_ramping_down_rate_input = Constraint(
            self.set_t, rule=init_ramping_down_rate_input
        )

        def init_ramping_up_rate_input(const, t):
            if t > 1:
                return (
                    sum(
                        self.input[t, car_input] - self.input[t - 1, car_input]
                        for car_input in b_tec.set_input_carriers
                    )
                    <= ramping_rate
                )
            else:
                return Constraint.Skip

        b_tec.const_ramping_up_rate_input = Constraint(
            self.set_t, rule=init_ramping_up_rate_input
        )

        def init_ramping_down_rate_output(const, t):
            if t > 1:
                return -ramping_rate <= sum(
                    self.output[t, car_output] - self.output[t - 1, car_output]
                    for car_output in b_tec.set_ouput_carriers
                )
            else:
                return Constraint.Skip

        b_tec.const_ramping_down_rate_output = Constraint(
            self.set_t, rule=init_ramping_down_rate_output
        )

        def init_ramping_down_rate_output(const, t):
            if t > 1:
                return (
                    sum(
                        self.output[t, car_output] - self.output[t - 1, car_output]
                        for car_output in b_tec.set_ouput_carriers
                    )
                    <= ramping_rate
                )
            else:
                return Constraint.Skip

        b_tec.const_ramping_up_rate_output = Constraint(
            self.set_t, rule=init_ramping_down_rate_output
        )

        return b_tec
