"""
Renewable technology with capacity factor (has no input)

Resembles a renewable technology with no input. In technology_time_series, the following
time-dependent data is required:
- capfactor: Capacity factor time series (between 0 and 1)

For pv and wind technologies, this time series can be generated with the
performance_from_climate_data plugin from climate data.

**Constraint declarations:**

- Output of technology. The output can be curtailed in three different ways.
  For ``curtailment == 0``, there is no curtailment possible. For ``curtailment
  == 1``, the curtailment is continuous. For ``curtailment == 2``,
  the size needs to be an integer, and the technology can only be curtailed discretely, i.e. by turning full
  modules off. For ``curtailment == 0`` (default), it holds:

.. math::
    Output_{t, car} = CapFactor_t * Size
"""

import pyomo.environ as pyo
import numpy as np

from adopt_net0.core.components.technology import Technology
from adopt_net0.core.components.utilities import get_attribute_from_dict


class Res(Technology):

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.emissions_based_on = "output"

        # Options
        self.curtailment = get_attribute_from_dict(
            self.performance_data, "curtailment", 0
        )

    def fit_performance(self, modelhub, component_id: tuple, **kwargs):
        """
        Fits technology performance and writes them to self.

        :param modelhub: modelhub
        :param tuple component_id: component id containing (period, node, component name)
        """
        super(Res, self).fit_performance(modelhub, component_id)

        period = component_id[0]
        node = component_id[1]
        technology_time_series = modelhub.data["time_series_data"]["full_resolution"][(period, node, "TechnologyTimeSeries", self.name)].to_dict(orient="list")
        self.processed_coeff.time_dependent_full = technology_time_series

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(Res, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        # Output bounds
        lower_output_bound = np.zeros(shape=(time_steps))
        upper_output_bound = self.processed_coeff.time_dependent_used["capfactor"]
        output_bounds = np.column_stack((lower_output_bound, upper_output_bound))
        self.bounds["output"]["electricity"] = output_bounds

    def construct_model(self, b_tec, modelhub, set_t_full, set_t_clustered, **kwargs):
        """
        Adds constraints to technology blocks for tec_type RES (renewable technology)

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """
        super(Res, self).construct_model(b_tec, modelhub, set_t_full, set_t_clustered, **kwargs)

        # DATA OF TECHNOLOGY
        coeff_td = self.processed_coeff.time_dependent_used
        coeff_ti = self.processed_coeff.time_independent
        rated_capacity = coeff_ti["rated_capacity"]

        # CONSTRAINTS
        if self.curtailment == 0:  # no curtailment allowed (default)

            def init_input_output(const, t, c_output):
                return (
                    self.output[t, c_output]
                    == coeff_td["capfactor"][t - 1] * b_tec.var_size * rated_capacity
                )

            b_tec.const_input_output = pyo.Constraint(
                self.set_t_performance,
                b_tec.set_output_carriers,
                rule=init_input_output,
            )

        elif self.curtailment == 1:  # continuous curtailment

            def init_input_output(const, t, c_output):
                return (
                    self.output[t, c_output]
                    <= coeff_td["capfactor"][t - 1] * b_tec.var_size * rated_capacity
                )

            b_tec.const_input_output = pyo.Constraint(
                self.set_t_performance,
                b_tec.set_output_carriers,
                rule=init_input_output,
            )

        elif self.curtailment == 2:  # discrete curtailment
            b_tec.var_size_on = pyo.Var(
                self.set_t_performance,
                within=pyo.NonNegativeIntegers,
                bounds=(b_tec.para_size_min, b_tec.para_size_max),
            )

            def init_curtailed_units(const, t):
                return b_tec.var_size_on[t] <= b_tec.var_size

            b_tec.const_curtailed_units = pyo.Constraint(
                self.set_t_performance, rule=init_curtailed_units
            )

            def init_input_output(const, t, c_output):
                return (
                    self.output[t, c_output]
                    == coeff_td["capfactor"][t - 1]
                    * b_tec.var_size_on[t]
                    * rated_capacity
                )

            b_tec.const_input_output = pyo.Constraint(
                self.set_t_performance,
                b_tec.set_output_carriers,
                rule=init_input_output,
            )



    def write_results_tec_design(self, h5_group, model_block):
        """
        Function to report technology design

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """

        super(Res, self).write_results_tec_design(h5_group, model_block)

        h5_group.create_dataset(
            "rated_capacity",
            data=self.processed_coeff.time_independent["rated_capacity"],
        )

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report technology operation

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super(Res, self).write_results_tec_operation(h5_group, model_block)

        coeff_ti = self.processed_coeff.time_independent
        rated_capacity = coeff_ti["rated_capacity"]

        capfactor = self.processed_coeff.time_dependent_used["capfactor"]

        h5_group.create_dataset(
            "max_out",
            data=[
                capfactor[t - 1] * model_block.var_size.value * rated_capacity
                for t in self.set_t_performance
            ],
        )

        h5_group.create_dataset("cap_factor", data=capfactor)

        if self.curtailment == 2:
            h5_group.create_dataset(
                "units_on",
                data=[model_block.var_size_on[t].value for t in self.set_t_performance],
            )

        for car in model_block.set_output_carriers:
            h5_group.create_dataset(
                "curtailment_" + car,
                data=[
                    capfactor[t - 1] * model_block.var_size.value * rated_capacity
                    - model_block.var_output[t, car].value
                    for t in self.set_t_performance
                ],
            )
