"""
Permanent storage technology (has no output)

This model resembles a permanent storage technology (sink). It takes energy and a main carrier (e.g. CO2 etc)
as inputs, and it has no output.

**Variable declarations:**

- ``var_storage_level``: Storage level in :math:`t`: :math:`E_t`

- ``var_injection_capacity``: Injection capacity :math: `injCapacity`

**Constraint declarations:**

- Size constraint:

  .. math::
    E_{t} \\leq S

- Maximal injection rate:

  .. math::
    Input_{t, maincar} \\leq injCapacity

- Maximal injection capacity:

  .. math::
    injCapacity \\leq injRateMax


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

import pyomo.environ as pyo
import h5py
import numpy as np

from adopt_net0.core.components.technology import Technology
from adopt_net0.core.components.utilities import annualize, set_discount_rate, link_full_resolution_to_clustered


class Sink(Technology):


    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.emissions_based_on = "input"
        self.main_input_carrier = tec_data["Performance"]["main_input_carrier"]
        self.flexibility_data = tec_data["Flexibility"]

    def fit_performance(self, modelhub: object, component_id: tuple, **kwargs):
        """
        Fits technology performance and writes it to self.

        :param modelhub: model hub
        :param tuple component_id: component id containing (period, node, component name)
        """
        super(Sink, self).fit_performance(modelhub, component_id)

        # For a flexibly optimized storage technology (i.e., not a fixed P-E ratio), an adapted CAPEX function is used
        # to account for charging and discharging capacity costs.
        if self.flexibility_data["injection_capacity_is_decision_var"]:
            self.economics["capex_model"] = 4

        self.processed_coeff.time_independent["injection_rate_max"] = (
            self.flexibility_data["injection_rate_max"]
        )
        if "energy_consumption" in self.performance_data["performance"]:
            self.processed_coeff.time_independent["energy_consumption"] = (
                self.performance_data["performance"]["energy_consumption"]
            )

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(Sink, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        # Input Bounds
        for car in self.input_carrier:
            if car == self.main_input_carrier:
                self.bounds["input"][car] = np.column_stack(
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
                    self.bounds["input"][car] = np.column_stack(
                        (
                            np.zeros(shape=(time_steps)),
                            np.ones(shape=(time_steps))
                            * self.flexibility_data["injection_rate_max"]
                            * energy_consumption["in"][car],
                        )
                    )

    def construct_model(self, b_tec, modelhub, set_t_full, set_t_clustered, **kwargs):
        """
        Construct SINK constraints

        Adds constraints to technology blocks for tec_type SINK, resembling a permanent storage technology

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """

        super(Sink, self).construct_model(b_tec, modelhub, set_t_full, set_t_clustered, **kwargs)

        # DATA OF TECHNOLOGY
        config = modelhub.data["config"]

        coeff_ti = self.processed_coeff.time_independent
        dynamics = self.processed_coeff.dynamics

        # sequence_storage = self.sequence
        if config["optimization"]["typicaldays"]["N"]["value"] == 0:
            sequence_storage = self.sequence
        elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
            sequence_storage = data["k_means_specs"]["sequence"]
        elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
            sequence_storage = self.sequence

        # Storage level and injection capacity decision variables
        b_tec.var_storage_level = pyo.Var(
            set_t_full,
            domain=pyo.NonNegativeReals,
            bounds=(0, b_tec.para_size_max),
        )
        b_tec.var_injection_capacity = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(0, coeff_ti["injection_rate_max"]),
        )

        if self.flexibility_data["injection_capacity_is_decision_var"]:
            self._define_sink_capex(b_tec, data)

        # Maximum storage level constraint
        def init_size_constraint(const, t):
            # storageLevel <= storSize
            return b_tec.var_storage_level[t] <= b_tec.var_size

        b_tec.const_size = pyo.Constraint(set_t_full, rule=init_size_constraint)

        # Constraint storage level

        def init_storage_level(const, t):
            # storageLevel[1] <= injRate[1]
            if t == 1:
                return (
                    b_tec.var_storage_level[t]
                    == self.input[
                        sequence_storage[t - 1],
                        self.main_input_carrier,
                    ]
                )
            else:
                # storageLevel[t] <= storageLevel[t-1]+injRate[t]
                return (
                    b_tec.var_storage_level[t]
                    == b_tec.var_storage_level[t - 1]
                    + self.input[
                        sequence_storage[t - 1],
                        self.main_input_carrier,
                    ]
                )

        b_tec.const_storage_level = pyo.Constraint(set_t_full, rule=init_storage_level)

        # Maximal injection rate
        def init_maximal_injection(const, t):
            # input[t] <= injectionCapacity
            return (
                self.input[t, self.main_input_carrier] <= b_tec.var_injection_capacity
            )

        b_tec.const_max_injection = pyo.Constraint(
            self.set_t_performance, rule=init_maximal_injection
        )

        # if injection rates are fixed/ flexible:
        def init_max_capacity_injection(const):
            if self.flexibility_data["injection_capacity_is_decision_var"]:
                # injectionCapacity <= injectionRateMax
                return b_tec.var_injection_capacity <= coeff_ti["injection_rate_max"]
            else:
                # injectionCapacity == injectionRateMax
                return b_tec.var_injection_capacity == coeff_ti["injection_rate_max"]

        b_tec.const_max_injection_cap = pyo.Constraint(rule=init_max_capacity_injection)

        # Energy consumption for injection
        if "energy_consumption" in coeff_ti:
            energy_consumption = coeff_ti["energy_consumption"]
            if "in" in energy_consumption:
                b_tec.set_energyconsumption_carriers_in = pyo.Set(
                    initialize=list(energy_consumption["in"].keys())
                )

                def init_energyconsumption_in(const, t, car):
                    # energyInput[t] = mainInput[t] * energyConsumption
                    return (
                        self.input[t, car]
                        == self.input[t, self.main_input_carrier]
                        * energy_consumption["in"][car]
                    )

                b_tec.const_energyconsumption_in = pyo.Constraint(
                    self.set_t_performance,
                    b_tec.set_energyconsumption_carriers_in,
                    rule=init_energyconsumption_in,
                )


    def _define_sink_capex(self, b_tec, data: dict):
        """

        Construct CAPEX of SINK constraints

        Adds constraints to technology blocks for tec_type SINK to calculate the CAPEX

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :return: pyomo block with technology model
        """

        config = modelhub.data["config"]

        economics = self.economics
        discount_rate = set_discount_rate(config, economics)
        fraction_of_year_modelled = modelhub.data["topology"]["temporal_information"]["fraction_of_year_modelled"]
        annualization_factor = annualize(
            discount_rate, economics["lifetime"], fraction_of_year_modelled
        )
        flexibility = self.flexibility_data
        coeff_ti = self.processed_coeff.time_independent

        # CAPEX PARAMETERS
        b_tec.para_unit_capex_injection_cap = pyo.Param(
            domain=pyo.Reals,
            initialize=flexibility["capex_injection_cap"],
            mutable=True,
        )
        b_tec.para_unit_capex_stor_size = pyo.Param(
            domain=pyo.Reals,
            initialize=economics["unit_capex"],
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
            b_tec.para_unit_capex_injection_cap_annual * coeff_ti["injection_rate_max"]
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

