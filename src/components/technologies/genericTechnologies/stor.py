from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import numpy as np
import pandas as pd


from ..utilities import FittedPerformance
from ..technology import Technology
from src.components.utilities import annualize, set_discount_rate


class Stor(Technology):
    """
    This model resembles a storage technology.
    Note that this technology only works for one carrier, and thus the carrier index is dropped in the below notation.

    **Parameter declarations:**

    - :math:`{\\eta}_{in}`: Charging efficiency

    - :math:`{\\eta}_{out}`: Discharging efficiency

    - :math:`{\\lambda_1}`: Self-Discharging coefficient (independent of environment)

    - :math:`{\\lambda_2(\\Theta)}`: Self-Discharging coefficient (dependent on environment)

    - :math:`Input_{max}`: Maximal charging capacity in one time-slice

    - :math:`Output_{max}`: Maximal discharging capacity in one time-slice

    **Variable declarations:**

    - Storage level in :math:`t`: :math:`E_t`

    - Charging in in :math:`t`: :math:`Input_{t}`

    - Discharging in in :math:`t`: :math:`Output_{t}`

    **Constraint declarations:**

    - Maximal charging and discharging:

      .. math::
        Input_{t} \leq Input_{max}

      .. math::
        Output_{t} \leq Output_{max}

    - Size constraint:

      .. math::
        E_{t} \leq S

    - Storage level calculation:

      .. math::
        E_{t} = E_{t-1} * (1 - \\lambda_1) - \\lambda_2(\\Theta) * E_{t-1} + {\\eta}_{in} * Input_{t} - 1 / {\\eta}_{out} * Output_{t}

    - If ``allow_only_one_direction == 1``, then only input or output can be unequal to zero in each respective time
      step (otherwise, simultaneous charging and discharging can lead to unwanted 'waste' of energy/material).

     - If in ``Flexibility`` the ``power_energy_ratio == fixed``, then the capacity of the charging and discharging power is fixed as a ratio of the energy capacity. Thus,

      .. math::
        Input_{max} = \gamma_{charging} * S

    - If in 'Flexibility' the "power_energy_ratio == flex" (flexible), then the capacity of the charging and discharging power is a variable in the optimization. In this case, the charging and discharging rates specified in the json file are the maximum installed
        capacities as a ratio of the energy capacity. The model will optimize the charging and discharging capacities,
        based on the incorporation of these components in the CAPEX function.


    - If an energy consumption for charging or dis-charging process is given, the respective carrier input is:

      .. math::
        Input_{t, car} = cons_{car, in} Input_{t}

      .. math::
        Input_{t, car} = cons_{car, out} Output_{t}
    """

    def __init__(self, tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()
        self.flexibility_data = tec_data["Flexibility"]

    def fit_technology_performance(self, climate_data: pd.DataFrame, location):
        """
        Fits conversion technology type STOR and fills in the fitted parameters in a dict

        :param pd.DataFrame climate_data: needed for the timesteps
        :param dict location: location data

        """

        time_steps = len(climate_data)

        # Main carrier (carrier to be stored)
        self.main_car = self.performance_data["main_input_carrier"]

        # Calculate ambient loss factor
        theta = self.performance_data["performance"]["theta"]
        ambient_loss_factor = (65 - climate_data["temp_air"]) / (90 - 65) * theta

        # Output Bounds
        for car in self.performance_data["output_carrier"]:
            self.fitted_performance.bounds["output"][car] = np.column_stack(
                (
                    np.zeros(shape=(time_steps)),
                    np.ones(shape=(time_steps))
                    * self.flexibility_data["discharge_rate"],
                )
            )
        # Input Bounds
        for car in self.performance_data["input_carrier"]:
            if car == self.performance_data["main_input_carrier"]:
                self.fitted_performance.bounds["input"][car] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps))
                        * self.flexibility_data["charge_rate"],
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
                            * self.flexibility_data["charge_rate"]
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

        # Coefficients
        self.fitted_performance.coefficients["ambient_loss_factor"] = (
            ambient_loss_factor.to_numpy()
        )
        for par in self.performance_data["performance"]:
            if not par == "theta":
                self.fitted_performance.coefficients[par] = self.performance_data[
                    "performance"
                ][par]

        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 1

    def construct_tech_model(self, b_tec: Block, data: dict, set_t: Set, set_t_clustered: Set
    ) -> Block:
        """
        Adds constraints to technology blocks for tec_type STOR, resembling a storage technology

        :param b_tec: technology Block
        :param dict data: input data
        :param Set set_t: set of timesteps in the model
        :param Set set_t_clustered: set of timesteps when clustering algorithm is used
        :return: technology Block with the constraints for the storage technology
        :rtype: b_tec
        """

        super(Stor, self).construct_tech_model(b_tec, data, set_t, set_t_clustered)

        set_t_full = self.set_t_full
        config = data["config"]

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        coeff = self.fitted_performance.coefficients

        if "allow_only_one_direction" in performance_data:
            allow_only_one_direction = performance_data["allow_only_one_direction"]
        else:
            allow_only_one_direction = 0

        # Todo: needs to be fixed with averaging algorithm
        # nr_timesteps_averaged = (
        #     energyhub.model_information.averaged_data_specs.nr_timesteps_averaged
        # )
        nr_timesteps_averaged = 1

        # Additional parameters
        eta_in = coeff["eta_in"]
        eta_out = coeff["eta_out"]
        eta_lambda = coeff["lambda"]
        charge_rate = self.flexibility_data["charge_rate"]
        discharge_rate = self.flexibility_data["discharge_rate"]
        ambient_loss_factor = coeff["ambient_loss_factor"]

        # Additional decision variables
        b_tec.var_storage_level = Var(
            set_t_full,
            domain=NonNegativeReals,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )
        b_tec.var_capacity_charge = Var(
            domain=NonNegativeReals, bounds=(0, b_tec.para_size_max * charge_rate)
        )
        b_tec.var_capacity_discharge = Var(
            domain=NonNegativeReals, bounds=(0, b_tec.para_size_max * discharge_rate)
        )

        if self.flexibility_data["power_energy_ratio"] == "flex":
            b_tec = self._define_stor_capex(b_tec, data)

        # Size constraint
        def init_size_constraint(const, t):
            # storageLevel <= storSize
            return b_tec.var_storage_level[t] <= b_tec.var_size

        b_tec.const_size = Constraint(set_t_full, rule=init_size_constraint)

        # Storage level calculation
        if (
            config["optimization"]["typicaldays"]["N"]["value"] != 0
            and not self.modelled_with_full_res
        ):

            def init_storage_level(const, t):
                if t == 1:  # couple first and last time interval: storageLevel[1] == storageLevel[end]
                    return b_tec.var_storage_level[t] == b_tec.var_storage_level[
                        max(set_t_full)
                    ] * (
                        1 - eta_lambda
                    ) ** nr_timesteps_averaged - b_tec.var_storage_level[
                        max(set_t_full)
                    ] * ambient_loss_factor[
                        max(set_t_full) - 1
                    ] ** nr_timesteps_averaged + (
                        eta_in * self.input[self.sequence[t - 1], self.main_car]
                        - 1 / eta_out * self.output[self.sequence[t - 1], self.main_car]
                    ) * sum(
                        (1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged)
                    )
                else:  # all other time intervalls
                    return b_tec.var_storage_level[t] == b_tec.var_storage_level[
                        t - 1
                    ] * (
                        1 - eta_lambda
                    ) ** nr_timesteps_averaged - b_tec.var_storage_level[
                        t
                    ] * ambient_loss_factor[
                        t - 1
                    ] ** nr_timesteps_averaged + (
                        eta_in * self.input[self.sequence[t - 1], self.main_car]
                        - 1 / eta_out * self.output[self.sequence[t - 1], self.main_car]
                    ) * sum(
                        (1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged)
                    )

            b_tec.const_storage_level = Constraint(set_t_full, rule=init_storage_level)
        else:

            def init_storage_level(const, t):
                if t == 1:  # couple first and last time interval
                    return b_tec.var_storage_level[t] == b_tec.var_storage_level[
                        max(set_t_full)
                    ] * (
                        1 - eta_lambda
                    ) ** nr_timesteps_averaged - b_tec.var_storage_level[
                        max(set_t_full)
                    ] * ambient_loss_factor[
                        max(set_t_full) - 1
                    ] ** nr_timesteps_averaged + (
                        eta_in * self.input[t, self.main_car]
                        - 1 / eta_out * self.output[t, self.main_car]
                    ) * sum(
                        (1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged)
                    )
                else:  # all other time intervalls
                    return b_tec.var_storage_level[t] == b_tec.var_storage_level[
                        t - 1
                    ] * (
                        1 - eta_lambda
                    ) ** nr_timesteps_averaged - b_tec.var_storage_level[
                        t
                    ] * ambient_loss_factor[
                        t - 1
                    ] ** nr_timesteps_averaged + (
                        eta_in * self.input[t, self.main_car]
                        - 1 / eta_out * self.output[t, self.main_car]
                    ) * sum(
                        (1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged)
                    )

            b_tec.const_storage_level = Constraint(set_t_full, rule=init_storage_level)

        # This makes sure that only either input or output is larger zero.
        if allow_only_one_direction == 1:
            self.big_m_transformation_required = 1
            s_indicators = range(0, 2)

            # Cut according to Germans work
            def init_cut_bidirectional(const, t):
                return (
                    self.output[t, self.main_car] / discharge_rate
                    + self.input[t, self.main_car] / charge_rate
                    <= b_tec.var_size
                )

            b_tec.const_cut_bidirectional = Constraint(
                self.set_t, rule=init_cut_bidirectional
            )

            def init_input_output(dis, t, ind):
                if ind == 0:  # input only

                    def init_output_to_zero(const, car_output):
                        return self.output[t, car_output] == 0

                    dis.const_output_to_zero = Constraint(
                        b_tec.set_output_carriers, rule=init_output_to_zero
                    )

                elif ind == 1:  # output only

                    def init_input_to_zero(const, car_input):
                        return self.input[t, car_input] == 0

                    dis.const_input_to_zero = Constraint(
                        b_tec.set_input_carriers, rule=init_input_to_zero
                    )

            b_tec.dis_input_output = Disjunct(
                self.set_t, s_indicators, rule=init_input_output
            )

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_input_output[t, i] for i in s_indicators]

            b_tec.disjunction_input_output = Disjunction(
                self.set_t, rule=bind_disjunctions
            )

        # Maximal charging and discharging rates
        def init_maximal_charge(const, t):
            return self.input[t, self.main_car] <= b_tec.var_capacity_charge

        b_tec.const_max_charge = Constraint(self.set_t, rule=init_maximal_charge)

        def init_maximal_discharge(const, t):
            return self.output[t, self.main_car] <= b_tec.var_capacity_discharge

        b_tec.const_max_discharge = Constraint(self.set_t, rule=init_maximal_discharge)

        # if the charging / discharging rates are fixed or flexible as a ratio of the energy capacity:
        def init_max_capacity_charge(const):
            if self.flexibility_data["power_energy_ratio"] == "fixed":
                return b_tec.var_capacity_charge == charge_rate * b_tec.var_size
            else:
                return b_tec.var_capacity_charge <= charge_rate * b_tec.var_size

        b_tec.const_max_cap_charge = Constraint(rule=init_max_capacity_charge)

        def init_max_capacity_discharge(const):
            if self.flexibility_data["power_energy_ratio"] == "fixed":
                return b_tec.var_capacity_discharge == discharge_rate * b_tec.var_size
            else:
                return b_tec.var_capacity_discharge <= discharge_rate * b_tec.var_size

        b_tec.const_max_cap_discharge = Constraint(rule=init_max_capacity_discharge)

        # Energy consumption charging/discharging
        if "energy_consumption" in coeff:
            energy_consumption = coeff["energy_consumption"]
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

            if "out" in energy_consumption:
                b_tec.set_energyconsumption_carriers_out = Set(
                    initialize=energy_consumption["out"].keys()
                )

                def init_energyconsumption_out(const, t, car):
                    return (
                        self.output[t, car]
                        == self.output[t, self.main_car]
                        * energy_consumption["out"][car]
                    )

                b_tec.const_energyconsumption_out = Constraint(
                    self.set_t,
                    b_tec.set_energyconsumption_carriers_out,
                    rule=init_energyconsumption_out,
                )

        # RAMPING RATES
        if "ramping_time" in self.performance_data:
            if not self.performance_data["ramping_time"] == -1:
                b_tec = self._define_ramping_rates(b_tec)

        return b_tec

    def _define_stor_capex(self, b_tec, data):

        flexibility = self.flexibility_data
        config = data["config"]
        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics.lifetime, fraction_of_year_modelled
        )

        # CAPEX PARAMETERS
        b_tec.para_unit_capex_charging_cap = Param(
            domain=Reals, initialize=flexibility["capex_charging_power"], mutable=True
        )
        b_tec.para_unit_capex_discharging_cap = Param(
            domain=Reals,
            initialize=flexibility["capex_discharging_power"],
            mutable=True,
        )
        b_tec.para_unit_capex_energy_cap = Param(
            domain=Reals, initialize=economics.capex_data["unit_capex"], mutable=True
        )

        b_tec.para_unit_capex_charging_cap_annual = Param(
            domain=Reals,
            initialize=(annualization_factor * b_tec.para_unit_capex_charging_cap),
            mutable=True,
        )
        b_tec.para_unit_capex_discharging_cap_annual = Param(
            domain=Reals,
            initialize=(annualization_factor * b_tec.para_unit_capex_discharging_cap),
            mutable=True,
        )
        b_tec.para_unit_capex_energy_cap_annual = Param(
            domain=Reals,
            initialize=(annualization_factor * b_tec.para_unit_capex_energy_cap),
            mutable=True,
        )

        # BOUNDS
        max_capex_charging_cap = (
            b_tec.para_unit_capex_charging_cap_annual
            * flexibility["charge_rate"]
            * b_tec.para_size_max
        )
        max_capex_discharging_cap = (
            b_tec.para_unit_capex_discharging_cap_annual
            * flexibility["discharge_rate"]
            * b_tec.para_size_max
        )
        max_capex_energy_cap = (
            b_tec.para_unit_capex_energy_cap_annual * b_tec.para_size_max
        )

        b_tec.var_capex_charging_cap = Var(
            domain=NonNegativeReals, bounds=(0, max_capex_charging_cap)
        )
        b_tec.var_capex_discharging_cap = Var(
            domain=NonNegativeReals, bounds=(0, max_capex_discharging_cap)
        )
        b_tec.var_capex_energy_cap = Var(
            domain=NonNegativeReals, bounds=(0, max_capex_energy_cap)
        )

        # CAPEX constraint
        b_tec.const_capex_charging_cap = Constraint(
            expr=b_tec.var_capacity_charge * b_tec.para_unit_capex_charging_cap_annual
            == b_tec.var_capex_charging_cap
        )
        b_tec.const_capex_discharging_cap = Constraint(
            expr=b_tec.var_capacity_discharge
            * b_tec.para_unit_capex_discharging_cap_annual
            == b_tec.var_capex_discharging_cap
        )
        b_tec.const_capex_energy_cap = Constraint(
            expr=b_tec.var_size * b_tec.para_unit_capex_energy_cap_annual
            == b_tec.var_capex_energy_cap
        )
        b_tec.const_capex_aux = Constraint(
            expr=b_tec.var_capex_charging_cap
            + b_tec.var_capex_discharging_cap
            + b_tec.var_capex_energy_cap
            == b_tec.var_capex_aux
        )

        return b_tec

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        super(Stor, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "storage_level",
            data=[model_block.var_storage_level[t].value for t in self.set_t_full],
        )

    def write_results_tec_design(self, h5_group, model_block):

        super(Stor, self).write_results_tec_design(h5_group, model_block)

        if self.flexibility_data["power_energy_ratio"] == "flex":
            h5_group.create_dataset(
                "capacity_charge", data=[model_block.var_capacity_charge.value]
            )
            h5_group.create_dataset(
                "capacity_discharge", data=[model_block.var_capacity_discharge.value]
            )
            h5_group.create_dataset(
                "capex_charging_cap", data=[model_block.var_capex_charging_cap.value]
            )
            h5_group.create_dataset(
                "capex_discharging_cap",
                data=[model_block.var_capex_discharging_cap.value],
            )
            h5_group.create_dataset(
                "capex_energy_cap", data=[model_block.var_capex_energy_cap.value]
            )

    def _define_ramping_rates(self, b_tec):
        """
        Constraints the inputs for a ramping rate. Implemented for input and output

        :param b_tec: technology model block
        :return:
        """
        ramping_time = self.performance_data["ramping_time"]

        # Calculate ramping rates
        if (
            "ref_size" in self.performance_data
            and not self.performance_data["ref_size"] == -1
        ):
            ramping_rate = self.performance_data["ref_size"] / ramping_time
        else:
            ramping_rate = b_tec.var_size / ramping_time

        # Constraints ramping rates
        if (
            "ramping_const_int" in self.performance_data
            and self.performance_data["ramping_const_int"] == 1
        ):

            s_indicators = range(0, 3)

            def init_ramping_operation_on(dis, t, ind):
                if t > 1:
                    if ind == 0:  # ramping constrained
                        dis.const_ramping_on = Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 0
                        )

                        def init_ramping_down_rate_operation_in(const):
                            return -ramping_rate <= sum(
                                self.input[t, car_input] - self.input[t - 1, car_input]
                                for car_input in b_tec.set_input_carriers
                            )

                        dis.const_ramping_down_rate_in = Constraint(
                            rule=init_ramping_down_rate_operation_in
                        )

                        def init_ramping_up_rate_operation_in(const):
                            return (
                                sum(
                                    self.input[t, car_input]
                                    - self.input[t - 1, car_input]
                                    for car_input in b_tec.set_input_carriers
                                )
                                <= ramping_rate
                            )

                        dis.const_ramping_up_rate_in = Constraint(
                            rule=init_ramping_up_rate_operation_in
                        )

                        def init_ramping_down_rate_operation_out(const):
                            return -ramping_rate <= sum(
                                self.output[t, car_output]
                                - self.output[t - 1, car_output]
                                for car_output in b_tec.set_output_carriers
                            )

                        dis.const_ramping_down_rate_out = Constraint(
                            rule=init_ramping_down_rate_operation_out
                        )

                        def init_ramping_up_rate_operation_out(const):
                            return (
                                sum(
                                    self.output[t, car_output]
                                    - self.output[t - 1, car_output]
                                    for car_output in b_tec.set_output_carriers
                                )
                                <= ramping_rate
                            )

                        dis.const_ramping_up_rate_out = Constraint(
                            rule=init_ramping_up_rate_operation_out
                        )

                    elif ind == 1:  # startup, no ramping constraint
                        dis.const_ramping_on = Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 1
                        )

                    else:  # shutdown, no ramping constraint
                        dis.const_ramping_on = Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == -1
                        )

            b_tec.dis_ramping_operation_on = Disjunct(
                self.set_t, s_indicators, rule=init_ramping_operation_on
            )

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_ramping_operation_on[t, i] for i in s_indicators]

            b_tec.disjunction_ramping_operation_on = Disjunction(
                self.set_t, rule=bind_disjunctions
            )

        else:

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
