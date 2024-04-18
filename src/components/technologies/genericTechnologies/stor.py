from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import numpy as np

from ..utilities import FittedPerformance
from ..technology import Technology


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
      step (otherwise, simultanous charging and discharging can lead to unwanted 'waste' of energy/material).

    - If an energy consumption for charging or dis-charging process is given, the respective carrier input is:

      .. math::
        Input_{t, car} = cons_{car, in} Input_{t}

      .. math::
        Input_{t, car} = cons_{car, out} Output_{t}
    """

    def __init__(self, tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()

    def fit_technology_performance(self, node_data):
        """
        Fits conversion technology type STOR and returns fitted parameters as a dict

        :param node_data: contains data on demand, climate data, etc.
        """

        climate_data = node_data.data["climate_data"]

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
                    * self.performance_data["performance"]["discharge_max"],
                )
            )
        # Input Bounds
        for car in self.performance_data["input_carrier"]:
            self.fitted_performance.bounds["input"][car] = np.column_stack(
                (
                    np.zeros(shape=(time_steps)),
                    np.ones(shape=(time_steps))
                    * self.performance_data["performance"]["charge_max"],
                )
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

    def construct_tech_model(self, b_tec, energyhub):
        """
        Adds constraints to technology blocks for tec_type STOR, resembling a storage technology

        :param b_tec:
        :param energyhub:
        :return: b_tec
        """

        super(Stor, self).construct_tech_model(b_tec, energyhub)

        set_t_full = energyhub.model.set_t_full

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        coeff = self.fitted_performance.coefficients

        if "allow_only_one_direction" in performance_data:
            allow_only_one_direction = performance_data["allow_only_one_direction"]
        else:
            allow_only_one_direction = 0

        nr_timesteps_averaged = (
            energyhub.model_information.averaged_data_specs.nr_timesteps_averaged
        )

        # Additional decision variables
        b_tec.var_storage_level = Var(
            set_t_full,
            domain=NonNegativeReals,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )

        # Abdditional parameters
        eta_in = coeff["eta_in"]
        eta_out = coeff["eta_out"]
        eta_lambda = coeff["lambda"]
        charge_max = coeff["charge_max"]
        discharge_max = coeff["discharge_max"]
        ambient_loss_factor = coeff["ambient_loss_factor"]

        # Size constraint
        def init_size_constraint(const, t):
            return b_tec.var_storage_level[t] <= b_tec.var_size

        b_tec.const_size = Constraint(set_t_full, rule=init_size_constraint)

        # Storage level calculation
        if (
            energyhub.model_information.clustered_data
            and not self.modelled_with_full_res
        ):

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
                    self.output[t, self.main_car] / discharge_max
                    + self.input[t, self.main_car] / charge_max
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
            return self.input[t, self.main_car] <= charge_max * b_tec.var_size

        b_tec.const_max_charge = Constraint(self.set_t, rule=init_maximal_charge)

        def init_maximal_discharge(const, t):
            return self.output[t, self.main_car] <= discharge_max * b_tec.var_size

        b_tec.const_max_discharge = Constraint(self.set_t, rule=init_maximal_discharge)

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
        if "ramping_rate" in self.performance_data:
            if not self.performance_data["ramping_rate"] == -1:
                b_tec = self._define_ramping_rates(b_tec)

        return b_tec

    def write_tec_operation_results_to_group(self, h5_group, model_block):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        super(Stor, self).write_tec_operation_results_to_group(h5_group, model_block)

        h5_group.create_dataset(
            "storage_level",
            data=[model_block.var_storage_level[t].value for t in self.set_t_full],
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
