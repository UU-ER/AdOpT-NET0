import pyomo.environ as pyo
import pyomo.gdp as gdp
import numpy as np
import pandas as pd

from ...utilities import get_attribute_from_dict, link_full_resolution_to_clustered
from ..technology import Technology


class HydroOpen(Technology):
    """
    Open pumped hydro technology

    Resembles a pumped hydro plant with additional natural inflows (defined in
    climate data). Note that this technology only works for one carrier, and thus the
    carrier index is dropped in the below notation.

    **Variable declarations:**

    - Storage level in :math:`t`: :math:`E_t`

    - Charging in :math:`t`: :math:`Input_{t}`

    - Discharging in :math:`t`: :math:`Output_{t}`

    **Constraint declarations:**

    The following constants are used:

    - :math:`{\\eta}_{in}`: Charging efficiency

    - :math:`{\\eta}_{out}`: Discharging efficiency

    - :math:`{\\lambda}`: Self-Discharging coefficient

    - :math:`Input_{max}`: Maximal charging capacity in one time-slice

    - :math:`Output_{max}`: Maximal discharging capacity in one time-slice

    - :math:`Natural_Inflow{t}`: Natural water inflow in time slice (can be negative, i.e. being an outflow)


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

    - Additionally, ramping rates of the technology can be constraint (for input and
      output).

      .. math::
         -rampingrate \leq Input_{t, maincar} - Input_{t-1, maincar} \leq rampingrate

      .. math::
         -rampingrate \leq \sum(Input_{t, car}) - \sum(Input_{t-1, car}) \leq rampingrate

    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.component_options.emissions_based_on = "input"
        self.component_options.main_input_carrier = tec_data["Performance"][
            "main_input_carrier"
        ]

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits technology performance

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        super(HydroOpen, self).fit_technology_performance(climate_data, location)

        # Coefficients
        for par in self.input_parameters.performance_data["performance"]:
            self.processed_coeff.time_independent[par] = (
                self.input_parameters.performance_data["performance"][par]
            )

        # Natural inflow
        if self.name + "_inflow" in climate_data:
            self.processed_coeff.time_dependent_full["hydro_inflow"] = climate_data[
                self.name + "_inflow"
            ]
        else:
            raise Exception(
                "Using Technology Type Hydro_Open requires a hydro_natural_inflow in climate data"
                " to be defined for this node. Add a column in the climate data for respective node with column name"
                f" {self.name}_inflow"
            )

        # Maximum discharge
        if self.input_parameters.performance_data["maximum_discharge_time_discrete"]:
            if self.name + "_maximum_discharge" in climate_data:
                self.processed_coeff.time_dependent_full["hydro_maximum_discharge"] = (
                    climate_data[self.name + "_maximum_discharge"]
                )
            else:
                raise Exception(
                    "Using Technology Type Hydro_Open with maximum_discharge_time_discrete == 1 requires "
                    "hydro_maximum_discharge to be defined for this node."
                )

        # Options
        self.component_options.other["allow_only_one_direction"] = (
            get_attribute_from_dict(
                self.input_parameters.performance_data, "allow_only_one_direction", 0
            )
        )
        self.component_options.other["can_pump"] = get_attribute_from_dict(
            self.input_parameters.performance_data, "can_pump", 1
        )
        self.component_options.other["bidirectional_precise"] = get_attribute_from_dict(
            self.input_parameters.performance_data, "bidirectional_precise", 1
        )
        self.component_options.other["maximum_discharge_time_discrete"] = (
            get_attribute_from_dict(
                self.input_parameters.performance_data,
                "maximum_discharge_time_discrete",
                1,
            )
        )

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(HydroOpen, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        # Output Bounds
        for car in self.component_options.output_carrier:
            self.bounds["output"][car] = np.column_stack(
                (
                    np.zeros(shape=(time_steps)),
                    np.ones(shape=(time_steps))
                    * self.input_parameters.performance_data["performance"][
                        "discharge_max"
                    ],
                )
            )

        # Input Bounds
        for car in self.component_options.input_carrier:
            self.bounds["input"][car] = np.column_stack(
                (
                    np.zeros(shape=(time_steps)),
                    np.ones(shape=(time_steps))
                    * self.input_parameters.performance_data["performance"][
                        "charge_max"
                    ],
                )
            )

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

        config = data["config"]

        # DATA OF TECHNOLOGY
        coeff_td = self.processed_coeff.time_dependent_used
        coeff_ti = self.processed_coeff.time_independent
        dynamics = self.processed_coeff.dynamics
        allow_only_one_direction = self.component_options.other[
            "allow_only_one_direction"
        ]

        eta_in = coeff_ti["eta_in"]
        eta_out = coeff_ti["eta_out"]
        eta_lambda = coeff_ti["lambda"]
        charge_max = coeff_ti["charge_max"]
        discharge_max = coeff_ti["discharge_max"]
        spilling_max = coeff_ti["spilling_max"]
        hydro_natural_inflow = coeff_td["hydro_inflow"]

        if config["optimization"]["timestaging"]["value"] != 0:
            nr_timesteps_averaged = config["optimization"]["timestaging"]["value"]
        else:
            nr_timesteps_averaged = 1

        # Additional decision variables
        b_tec.var_storage_level = pyo.Var(
            self.set_t_performance,
            b_tec.set_input_carriers,
            domain=pyo.NonNegativeReals,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )
        b_tec.var_spilling = pyo.Var(
            self.set_t_performance,
            domain=pyo.NonNegativeReals,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )

        # Abdditional parameters

        # Size constraint
        def init_size_constraint(const, t, car):
            return b_tec.var_storage_level[t, car] <= b_tec.var_size

        b_tec.const_size = pyo.Constraint(
            self.set_t_performance, b_tec.set_input_carriers, rule=init_size_constraint
        )

        # Storage level calculation
        def init_storage_level(const, t, car):
            if t == 1:  # couple first and last time interval
                return (
                    b_tec.var_storage_level[t, car]
                    == b_tec.var_storage_level[max(self.set_t_performance), car]
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
            self.set_t_performance, b_tec.set_input_carriers, rule=init_storage_level
        )

        if not self.component_options.other["can_pump"]:

            def init_input_zero(const, t, car):
                return self.input[t, car] == 0

            b_tec.const_input_zero = pyo.Constraint(
                self.set_t_performance, b_tec.set_input_carriers, rule=init_input_zero
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
                self.set_t_performance,
                b_tec.set_input_carriers,
                rule=init_cut_bidirectional,
            )

            # Disjunct modelling
            if self.component_options.other["bidirectional_precise"]:
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
                    self.set_t_performance, s_indicators, rule=init_input_output
                )

                # Bind disjuncts
                def bind_disjunctions(dis, t):
                    return [b_tec.dis_input_output[t, i] for i in s_indicators]

                b_tec.disjunction_input_output = gdp.Disjunction(
                    self.set_t_performance, rule=bind_disjunctions
                )

        # Maximal charging and discharging rates
        def init_maximal_charge(const, t, car):
            return self.input[t, car] <= charge_max * b_tec.var_size

        b_tec.const_max_charge = pyo.Constraint(
            self.set_t_performance, b_tec.set_input_carriers, rule=init_maximal_charge
        )

        def init_maximal_discharge(const, t, car):
            return self.output[t, car] <= discharge_max * b_tec.var_size

        b_tec.const_max_discharge = pyo.Constraint(
            self.set_t_performance,
            b_tec.set_input_carriers,
            rule=init_maximal_discharge,
        )

        if self.component_options.other["maximum_discharge_time_discrete"]:

            def init_maximal_discharge2(const, t, car):
                return self.output[t, car] <= coeff_td["hydro_maximum_discharge"][t - 1]

            b_tec.const_max_discharge2 = pyo.Constraint(
                self.set_t_performance,
                b_tec.set_input_carriers,
                rule=init_maximal_discharge2,
            )

        # Maximum spilling
        def init_maximal_spilling(const, t):
            return b_tec.var_spilling[t] <= spilling_max * b_tec.var_size

        b_tec.const_max_spilling = pyo.Constraint(
            self.set_t_performance, rule=init_maximal_spilling
        )

        # RAMPING RATES
        if "ramping_time" in dynamics:
            if not dynamics["ramping_time"] == -1:
                b_tec = self._define_ramping_rates(b_tec, data)

        return b_tec

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report technology operation

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super(HydroOpen, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "spilling",
            data=[model_block.var_spilling[t].value for t in self.set_t_performance],
        )
        for car in model_block.set_input_carriers:
            h5_group.create_dataset(
                "storage_level_" + car,
                data=[
                    model_block.var_storage_level[t, car].value
                    for t in self.set_t_performance
                ],
            )

    def _define_ramping_rates(self, b_tec, data):
        """
        Constraints the inputs for a ramping rate

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        dynamics = self.processed_coeff.dynamics

        ramping_time = dynamics["ramping_time"]

        # Calculate ramping rates
        if "ref_size" in dynamics and not dynamics["ref_size"] == -1:
            ramping_rate = dynamics["ref_size"] / ramping_time
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
                self.set_t_performance, s_indicators, rule=init_ramping_operation_on
            )

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_ramping_operation_on[t, i] for i in s_indicators]

            b_tec.disjunction_ramping_operation_on = gdp.Disjunction(
                self.set_t_performance, rule=bind_disjunctions
            )

        else:
            if data["config"]["optimization"]["typicaldays"]["N"]["value"] == 0:
                input_aux_rr = self.input
                output_aux_rr = self.output
                set_t_rr = self.set_t_performance
            else:
                # init bounds at full res
                bounds_rr_full = {
                    "input": self.fitting_class.calculate_input_bounds(
                        self.component_options.size_based_on, len(self.set_t_full)
                    ),
                    "output": self.fitting_class.calculate_output_bounds(
                        self.component_options.size_based_on, len(self.set_t_full)
                    ),
                }

                # create input and output variable for full res
                def init_input_bounds(bounds, t, car):
                    return tuple(
                        bounds_rr_full["input"][car][t - 1, :]
                        * self.processed_coeff.time_independent["size_max"]
                        * self.processed_coeff.time_independent["rated_power"]
                    )

                def init_output_bounds(bounds, t, car):
                    return tuple(
                        bounds_rr_full["output"][car][t - 1, :]
                        * self.processed_coeff.time_independent["size_max"]
                        * self.processed_coeff.time_independent["rated_power"]
                    )

                b_tec.var_input_rr_full = pyo.Var(
                    self.set_t_full,
                    b_tec.set_input_carriers,
                    within=pyo.NonNegativeReals,
                    bounds=init_input_bounds,
                )
                b_tec.var_output_rr_full = pyo.Var(
                    self.set_t_full,
                    b_tec.set_output_carriers,
                    within=pyo.NonNegativeReals,
                    bounds=init_output_bounds,
                )

                b_tec.const_link_full_resolution_rr_input = (
                    link_full_resolution_to_clustered(
                        self.input,
                        b_tec.var_input_rr_full,
                        self.set_t_full,
                        self.sequence,
                        b_tec.set_input_carriers,
                    )
                )

                b_tec.const_link_full_resolution_rr_output = (
                    link_full_resolution_to_clustered(
                        self.output,
                        b_tec.var_output_rr_full,
                        self.set_t_full,
                        sequence_storage,
                        b_tec.set_output_carriers,
                    )
                )

                input_aux_rr = b_tec.var_input_rr_full
                output_aux_rr = b_tec.var_output_rr_full
                set_t_rr = self.set_t_full

            # Ramping constraint without integers
            def init_ramping_down_rate_input(const, t):
                # -rampingRate <= input[t] - input[t-1]
                if t > 1:
                    return (
                        -ramping_rate
                        <= input_aux_rr[t, self.component_options.main_input_carrier]
                        - input_aux_rr[t - 1, self.component_options.main_input_carrier]
                    )
                else:
                    return pyo.Constraint.Skip

            b_tec.const_ramping_down_rate = pyo.Constraint(
                set_t_rr, rule=init_ramping_down_rate_input
            )

            def init_ramping_up_rate_input(const, t):
                # input[t] - input[t-1] <= rampingRate
                if t > 1:
                    return (
                        input_aux_rr[t, self.component_options.main_input_carrier]
                        - input_aux_rr[t - 1, self.component_options.main_input_carrier]
                        <= ramping_rate
                    )
                else:
                    return pyo.Constraint.Skip

            b_tec.const_ramping_up_rate = pyo.Constraint(
                set_t_rr, rule=init_ramping_up_rate_input
            )

            def init_ramping_down_rate_output(const, t):
                # -rampingRate <= output[t] - output[t-1]
                if t > 1:
                    return -ramping_rate <= sum(
                        output_aux_rr[t, car_output] - output_aux_rr[t - 1, car_output]
                        for car_output in b_tec.set_ouput_carriers
                    )
                else:
                    return pyo.Constraint.Skip

            b_tec.const_ramping_down_rate_output = pyo.Constraint(
                self.set_t_performance, rule=init_ramping_down_rate_output
            )

            def init_ramping_down_rate_output(const, t):
                # output[t] - output[t-1] <= rampingRate
                if t > 1:
                    return (
                        sum(
                            output_aux_rr[t, car_output]
                            - output_aux_rr[t - 1, car_output]
                            for car_output in b_tec.set_ouput_carriers
                        )
                        <= ramping_rate
                    )
                else:
                    return pyo.Constraint.Skip

            b_tec.const_ramping_up_rate_output = pyo.Constraint(
                self.set_t_performance, rule=init_ramping_down_rate_output
            )

        return b_tec
