import pyomo.environ as pyo
import pyomo.gdp as gdp
import numpy as np
import pandas as pd

from ..utilities import FittedPerformance
from ..technology import Technology


class HydroOpen(Technology):
    """
    Resembles a pumped hydro plant with additional natural inflows (defined in climate data)

    Note that this technology only works for one carrier, and thus the carrier index is dropped in the below notation.

    **Parameter declarations:**

    - :math:`{\\eta}_{in}`: Charging efficiency

    - :math:`{\\eta}_{out}`: Discharging efficiency

    - :math:`{\\lambda}`: Self-Discharging coefficient

    - :math:`Input_{max}`: Maximal charging capacity in one time-slice

    - :math:`Output_{max}`: Maximal discharging capacity in one time-slice

    - :math:`Natural_Inflow{t}`: Natural water inflow in time slice (can be negative, i.e. being an outflow)

    **Variable declarations:**

    - Storage level in :math:`t`: :math:`E_t`

    - Charging in :math:`t`: :math:`Input_{t}`

    - Discharging in :math:`t`: :math:`Output_{t}`

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
        E_{t} = E_{t-1} * (1 - \\lambda) + {\\eta}_{in} * Input_{t} - 1 / {\\eta}_{out} * Output_{t} + Natural_Inflow_{t}

    - If ``allow_only_one_direction == 1``, then only input or output can be unequal to zero in each respective time
      step (otherwise, simultanous charging and discharging can lead to unwanted 'waste' of energy/material).

    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()
        self.main_car = self.performance_data["main_input_carrier"]

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits technology performance

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """

        # Climate data & Number of timesteps
        time_steps = len(climate_data)

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
        for par in self.performance_data["performance"]:
            self.fitted_performance.coefficients[par] = self.performance_data[
                "performance"
            ][par]

        # Natural inflow
        if self.name + "_inflow" in climate_data:
            self.fitted_performance.coefficients["hydro_inflow"] = climate_data[
                self.name + "_inflow"
            ]
        else:
            raise Exception(
                "Using Technology Type Hydro_Open requires a hydro_natural_inflow in climate data"
                " to be defined for this node. Add a column in the climate data for respective node with column name"
                f" {self.name}_inflow"
            )

        # Maximum discharge
        if self.performance_data["maximum_discharge_time_discrete"]:
            if self.name + "_maximum_discharge" in climate_data:
                self.fitted_performance.coefficients["hydro_maximum_discharge"] = (
                    climate_data[self.name + "_maximum_discharge"]
                )
            else:
                raise Exception(
                    "Using Technology Type Hydro_Open with maximum_discharge_time_discrete == 1 requires "
                    "hydro_maximum_discharge to be defined for this node."
                )

        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 1

    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type Hydro_Open

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(HydroOpen, self).construct_tech_model(
            b_tec, data, set_t_full, set_t_clustered
        )

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        coeff = self.fitted_performance.coefficients
        bounds = self.fitted_performance.bounds

        if "allow_only_one_direction" in performance_data:
            allow_only_one_direction = performance_data["allow_only_one_direction"]
        else:
            allow_only_one_direction = 0

        can_pump = performance_data["can_pump"]
        if performance_data["maximum_discharge_time_discrete"]:
            hydro_maximum_discharge = coeff["hydro_maximum_discharge"]

        # Todo: needs to be fixed with averaging algorithm
        # nr_timesteps_averaged = (
        #     energyhub.model_information.averaged_data_specs.nr_timesteps_averaged
        # )
        nr_timesteps_averaged = 1

        # Additional decision variables
        b_tec.var_storage_level = pyo.Var(
            self.set_t,
            b_tec.set_input_carriers,
            domain=pyo.NonNegativeReals,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )
        b_tec.var_spilling = pyo.Var(
            self.set_t,
            domain=pyo.NonNegativeReals,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )

        # Abdditional parameters
        eta_in = coeff["eta_in"]
        eta_out = coeff["eta_out"]
        eta_lambda = coeff["lambda"]
        charge_max = coeff["charge_max"]
        discharge_max = coeff["discharge_max"]
        hydro_natural_inflow = coeff["hydro_inflow"]
        spilling_max = coeff["spilling_max"]

        # Size constraint
        def init_size_constraint(const, t, car):
            return b_tec.var_storage_level[t, car] <= b_tec.var_size

        b_tec.const_size = pyo.Constraint(
            self.set_t, b_tec.set_input_carriers, rule=init_size_constraint
        )

        # Storage level calculation
        def init_storage_level(const, t, car):
            if t == 1:  # couple first and last time interval
                return (
                    b_tec.var_storage_level[t, car]
                    == b_tec.var_storage_level[max(self.set_t), car]
                    * (1 - eta_lambda) ** nr_timesteps_averaged
                    + (
                        eta_in * self.input[t, car]
                        - 1 / eta_out * self.output[t, car]
                        - b_tec.var_spilling[t]
                    )
                    * sum(
                        (1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged)
                    )
                    + hydro_natural_inflow.iloc[t - 1]
                )
            else:  # all other time intervals
                return (
                    b_tec.var_storage_level[t, car]
                    == b_tec.var_storage_level[t - 1, car]
                    * (1 - eta_lambda) ** nr_timesteps_averaged
                    + (
                        eta_in * self.input[t, car]
                        - 1 / eta_out * self.output[t, car]
                        - b_tec.var_spilling[t]
                    )
                    * sum(
                        (1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged)
                    )
                    + hydro_natural_inflow.iloc[t - 1]
                )

        b_tec.const_storage_level = pyo.Constraint(
            self.set_t, b_tec.set_input_carriers, rule=init_storage_level
        )

        if not can_pump:

            def init_input_zero(const, t, car):
                return self.input[t, car] == 0

            b_tec.const_input_zero = pyo.Constraint(
                self.set_t, b_tec.set_input_carriers, rule=init_input_zero
            )

        # This makes sure that only either input or output is larger zero.
        if allow_only_one_direction == 1:

            # Cut according to Germans work
            def init_cut_bidirectional(const, t, car):
                return (
                    self.output[t, car] / discharge_max
                    + self.input[t, car] / charge_max
                    <= b_tec.var_size
                )

            b_tec.const_cut_bidirectional = pyo.Constraint(
                self.set_t, b_tec.set_input_carriers, rule=init_cut_bidirectional
            )

            # Disjunct modelling
            if "bidirectional_precise" in self.performance_data:
                if self.performance_data["bidirectional_precise"] == 1:
                    self.big_m_transformation_required = 1
                    s_indicators = range(0, 2)

                    def init_input_output(dis, t, ind):
                        if ind == 0:  # input only

                            def init_output_to_zero(const, car_input):
                                return self.output[t, car_input] == 0

                            dis.const_output_to_zero = pyo.Constraint(
                                b_tec.set_input_carriers, rule=init_output_to_zero
                            )

                        elif ind == 1:  # output only

                            def init_input_to_zero(const, car_input):
                                return self.input[t, car_input] == 0

                            dis.const_input_to_zero = pyo.Constraint(
                                b_tec.set_input_carriers, rule=init_input_to_zero
                            )

                    b_tec.dis_input_output = gdp.Disjunct(
                        self.set_t, s_indicators, rule=init_input_output
                    )

                    # Bind disjuncts
                    def bind_disjunctions(dis, t):
                        return [b_tec.dis_input_output[t, i] for i in s_indicators]

                    b_tec.disjunction_input_output = gdp.Disjunction(
                        self.set_t, rule=bind_disjunctions
                    )

        # Maximal charging and discharging rates
        def init_maximal_charge(const, t, car):
            return self.input[t, car] <= charge_max * b_tec.var_size

        b_tec.const_max_charge = pyo.Constraint(
            self.set_t, b_tec.set_input_carriers, rule=init_maximal_charge
        )

        def init_maximal_discharge(const, t, car):
            return self.output[t, car] <= discharge_max * b_tec.var_size

        b_tec.const_max_discharge = pyo.Constraint(
            self.set_t, b_tec.set_input_carriers, rule=init_maximal_discharge
        )

        if performance_data["maximum_discharge_time_discrete"]:

            def init_maximal_discharge2(const, t, car):
                return self.output[t, car] <= hydro_maximum_discharge[t - 1]

            b_tec.const_max_discharge2 = pyo.Constraint(
                self.set_t, b_tec.set_input_carriers, rule=init_maximal_discharge2
            )

        # Maximum spilling
        def init_maximal_spilling(const, t):
            return b_tec.var_spilling[t] <= spilling_max * b_tec.var_size

        b_tec.const_max_spilling = pyo.Constraint(
            self.set_t, rule=init_maximal_spilling
        )

        # RAMPING RATES
        if "ramping_time" in self.performance_data:
            if not self.performance_data["ramping_time"] == -1:
                b_tec = self._define_ramping_rates(b_tec)

        return b_tec

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report technology operation

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super(HydroOpen, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "spilling", data=[model_block.var_spilling[t].value for t in self.set_t]
        )
        for car in model_block.set_input_carriers:
            h5_group.create_dataset(
                "storage_level_" + car,
                data=[model_block.var_storage_level[t, car].value for t in self.set_t],
            )

    def _define_ramping_rates(self, b_tec):
        """
        Constraints the inputs for a ramping rate

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
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
                        dis.const_ramping_on = pyo.Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 0
                        )

                        def init_ramping_down_rate_operation_in(const):
                            return -ramping_rate <= sum(
                                self.input[t, car_input] - self.input[t - 1, car_input]
                                for car_input in b_tec.set_input_carriers
                            )

                        dis.const_ramping_down_rate_in = pyo.Constraint(
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

                        dis.const_ramping_up_rate_in = pyo.Constraint(
                            rule=init_ramping_up_rate_operation_in
                        )

                        def init_ramping_down_rate_operation_out(const):
                            return -ramping_rate <= sum(
                                self.output[t, car_output]
                                - self.output[t - 1, car_output]
                                for car_output in b_tec.set_output_carriers
                            )

                        dis.const_ramping_down_rate_out = pyo.Constraint(
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

                        dis.const_ramping_up_rate_out = pyo.Constraint(
                            rule=init_ramping_up_rate_operation_out
                        )

                    elif ind == 1:  # startup, no ramping constraint
                        dis.const_ramping_on = pyo.Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 1
                        )

                    else:  # shutdown, no ramping constraint
                        dis.const_ramping_on = pyo.Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == -1
                        )

            b_tec.dis_ramping_operation_on = gdp.Disjunct(
                self.set_t, s_indicators, rule=init_ramping_operation_on
            )

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_ramping_operation_on[t, i] for i in s_indicators]

            b_tec.disjunction_ramping_operation_on = gdp.Disjunction(
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
                    return pyo.Constraint.Skip

            b_tec.const_ramping_down_rate_input = pyo.Constraint(
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
                    return pyo.Constraint.Skip

            b_tec.const_ramping_up_rate_input = pyo.Constraint(
                self.set_t, rule=init_ramping_up_rate_input
            )

            def init_ramping_down_rate_output(const, t):
                if t > 1:
                    return -ramping_rate <= sum(
                        self.output[t, car_output] - self.output[t - 1, car_output]
                        for car_output in b_tec.set_ouput_carriers
                    )
                else:
                    return pyo.Constraint.Skip

            b_tec.const_ramping_down_rate_output = pyo.Constraint(
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
                    return pyo.Constraint.Skip

            b_tec.const_ramping_up_rate_output = pyo.Constraint(
                self.set_t, rule=init_ramping_down_rate_output
            )

        return b_tec
