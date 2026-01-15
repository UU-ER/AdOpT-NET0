"""
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
    Input_{t} \\leq Input_{max}

  .. math::
    Output_{t} \\leq Output_{max}

- Size constraint:

  .. math::
    E_{t} \\leq S

- Storage level calculation:

  .. math::
    E_{t} = E_{t-1} * (1 - \\lambda) + {\\eta}_{in} * Input_{t} - 1 / {\\eta}_{out} * Output_{t} + Natural_Inflow_{t}

- If ``allow_only_one_direction == 1``, then only input or output can be unequal to zero in each respective time
  step (otherwise, simultanous charging and discharging can lead to unwanted 'waste' of energy/material).

"""


import pyomo.environ as pyo
import pyomo.gdp as gdp
import numpy as np

from adopt_net0.core.components.utilities import get_attribute_from_dict, link_full_resolution_to_clustered
from adopt_net0.core.components.technology import Technology


class HydroOpen(Technology):
    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.emissions_based_on = "input"
        self.main_input_carrier = tec_data["Performance"]["main_input_carrier"]

        self.allow_only_one_direction = tec_data["Performance"][
            "allow_only_one_direction"
        ]
        if self.allow_only_one_direction:
            self.allow_only_one_direction_precise = get_attribute_from_dict(
                tec_data["Performance"], "allow_only_one_direction_precise", 1
            )

    def fit_performance(self, modelhub: object, component_id: tuple, **kwargs):
        """
        Fits technology performance and writes it to self.

        :param modelhub: model hub
        :param tuple component_id: component id containing (period, node, component name)
        """
        super(HydroOpen, self).fit_performance(modelhub, component_id)
        period = component_id[0]
        node = component_id[1]
        technology_time_series = modelhub.data["time_series_data"]["full_resolution"][(period, node, "TechnologyTimeSeries", self.name)]

        # Coefficients
        for par in self.performance_data["performance"]:
            self.processed_coeff.time_independent[par] = self.performance_data[
                "performance"
            ][par]

        # Natural inflow
        if self.name + "_inflow" in technology_data:
            self.processed_coeff.time_dependent_full["hydro_inflow"] = technology_data["inflow"].to_numpy()
        else:
            raise Exception(
                "Using Technology Type Hydro_Open requires a hydro_natural_inflow in climate data"
                " to be defined for this node. Add a column in the climate data for respective node with column name"
                f" {self.name}_inflow"
            )

        # Maximum discharge
        if self.performance_data["maximum_discharge_time_discrete"]:
            if self.name + "_maximum_discharge" in technology_data:
                self.processed_coeff.time_dependent_full["hydro_maximum_discharge"] = (
                    technology_data["maximum_discharge"]
                )
            else:
                raise Exception(
                    "Using Technology Type Hydro_Open with maximum_discharge_time_discrete == 1 requires "
                    "hydro_maximum_discharge to be defined for this node."
                )

        # Options
        self.performance_data["can_pump"] = get_attribute_from_dict(
            self.performance_data, "can_pump", 1
        )

        self.performance_data["maximum_discharge_time_discrete"] = (
            get_attribute_from_dict(
                self.performance_data,
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
        for car in self.output_carrier:
            self.bounds["output"][car] = np.column_stack(
                (
                    np.zeros(shape=(time_steps)),
                    np.ones(shape=(time_steps))
                    * self.performance_data["performance"]["discharge_max"],
                )
            )

        # Input Bounds
        for car in self.input_carrier:
            self.bounds["input"][car] = np.column_stack(
                (
                    np.zeros(shape=(time_steps)),
                    np.ones(shape=(time_steps))
                    * self.performance_data["performance"]["charge_max"],
                )
            )

    def construct_model(self, b_tec, modelhub, set_t_full, set_t_clustered, **kwargs):
        """
        Adds constraints to technology blocks for tec_type Hydro_Open

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(HydroOpen, self).construct_model(
            b_tec, modelhub, set_t_full, set_t_clustered, **kwargs
        )

        config = modelhub.data["config"]


        # DATA OF TECHNOLOGY
        coeff_td = self.processed_coeff.time_dependent_used
        coeff_ti = self.processed_coeff.time_independent
        dynamics = self.processed_coeff.dynamics

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
            bounds=(0, b_tec.para_size_max),
        )
        b_tec.var_spilling = pyo.Var(
            self.set_t_performance,
            domain=pyo.NonNegativeReals,
            bounds=(0, b_tec.para_size_max),
        )

        # Additional parameters

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
                    + hydro_natural_inflow[t - 1]
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
                    + hydro_natural_inflow[t - 1]
                )

        b_tec.const_storage_level = pyo.Constraint(
            self.set_t_performance, b_tec.set_input_carriers, rule=init_storage_level
        )

        if not self.performance_data["can_pump"]:

            def init_input_zero(const, t, car):
                return self.input[t, car] == 0

            b_tec.const_input_zero = pyo.Constraint(
                self.set_t_performance, b_tec.set_input_carriers, rule=init_input_zero
            )

        # This makes sure that only either input or output is larger zero.
        if self.allow_only_one_direction == 1:

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
            if self.allow_only_one_direction_precise:
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

        if self.performance_data["maximum_discharge_time_discrete"]:

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
