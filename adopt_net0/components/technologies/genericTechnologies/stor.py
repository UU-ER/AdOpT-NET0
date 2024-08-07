import pyomo.core
import pyomo.environ as pyo
import pyomo.gdp as gdp
import numpy as np
import pandas as pd

from ..technology import Technology, set_capex_model
from ....components.utilities import (
    annualize,
    set_discount_rate,
    get_attribute_from_dict,
    link_full_resolution_to_clustered,
)


class Stor(Technology):
    """
    Storage Technology

    This model resembles a storage technology.
    Note that this technology only works for one carrier, and thus the carrier index is dropped in the below notation.

    **Variable declarations:**

    - ``var_storage_level``: Storage level in :math:`t`: :math:`E_t`

    - ``var_capacity_charge``: Charging capacity

    - ``var_capacity_discharge``: Discharging capacity

    **Constraint declarations:**

    The following constants are used:

    - :math:`{\\eta}_{in}`: Charging efficiency

    - :math:`{\\eta}_{out}`: Discharging efficiency

    - :math:`{\\lambda_1}`: Self-Discharging coefficient (independent of environment)

    - :math:`{\\lambda_2(\\Theta)}`: Self-Discharging coefficient (dependent on environment)

    - :math:`Input_{max}`: Maximal charging capacity

    - :math:`Output_{max}`: Maximal discharging capacity

    - Size constraint:

      .. math::
        E_{t} \leq S

    - Maximal charging and discharging:

      .. math::
        Input_{t} \leq Input_{max}

      .. math::
        Output_{t} \leq Output_{max}

    - Storage level calculation:

      .. math::
        E_{t} = E_{t-1} * (1 - \\lambda_1) - \\lambda_2(\\Theta) * E_{t-1} + {\\eta}_{in} * Input_{t} - 1 / {\\eta}_{out} * Output_{t}

    - If ``allow_only_one_direction == 1``, simultaneous charging and discharging is limited
      using Equation 24 from Morales-España, Germán & Hernandez, Ricardo & Helistö, Niina & Kiviluoma,
      Juha. (2022). LP Formulation for Optimal Investment and Operation of Storage Including Reserves.
      10.13140/RG.2.2.27048.03840.

    - If ``allow_only_one_direction_precise == 1``, then only input or output can be unequal
      to zero in each respective time step (otherwise, simultaneous charging and
      discharging can lead to unwanted 'waste' of energy/material).

     - If in ``Flexibility`` the ``power_energy_ratio == fixed``, then the capacity of
       the charging and discharging power is fixed as a ratio of the energy capacity.
       Thus:

       .. math::
         Input_{max} = \gamma_{charging} * S

    - If in 'Flexibility' the "power_energy_ratio == flex" (flexible), then the
      capacity of the charging and discharging power is a variable in the
      optimization. In this case, the charging and discharging rates specified in the
      json file are the maximum installed capacities as a ratio of the energy
      capacity. The model will optimize the charging and discharging capacities,
      based on the incorporation of these components in the CAPEX function.

    - If an energy consumption for charging or dis-charging process is given, the respective carrier input is:

      .. math::
        Input_{t, car} = cons_{car, in} Input_{t}

      .. math::
        Input_{t, car} = cons_{car, out} Output_{t}

    - CAPEX is given by three contributions, the CAPEX of the charging capacity, the CAPEX of the discharging capacity,
      and the CAPEX of the energy capacity (i.e., the size of the storage).

      .. math::
        CAPEX_{chargeCapacity} = chargeCapacity * unitCost_{chargeCapacity}

      .. math::
        CAPEX_{dischargeCapacity} = dischargeCapacity * unitCost_{dischargeCapacity}

      .. math::
        CAPEX_{storSize} = storSize * unitCost_{storSize}

    - Additionally, ramping rates of the technology can be constraint (for input and
      output).

      .. math::
         -rampingrate \leq Input_{t, maincar} - Input_{t-1, maincar} \leq rampingrate

      .. math::
         -rampingrate \leq \sum(Output_{t, car}) - \sum(Output_{t-1, car}) \leq
         rampingrate


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

        self.flexibility_data = tec_data["Flexibility"]

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits conversion technology type STOR and fills in the fitted parameters in a dict

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        super(Stor, self).fit_technology_performance(climate_data, location)

        # For a flexibly optimized storage technology (i.e., not a fixed P-E ratio), an adapted CAPEX function is used
        # to account for charging and discharging capacity costs.
        if self.flexibility_data["power_energy_ratio"] == "flexratio":
            self.economics.capex_model = 4
        if self.flexibility_data["power_energy_ratio"] not in [
            "flexratio",
            "fixedratio",
            "fixedcapacity",
        ]:
            raise Warning(
                "power_energy_ratio should be either flexible ('flexratio') or fixed ('fixedratio') or as capacity ('fixedcapacity')"
            )

        # Coefficients
        theta = self.input_parameters.performance_data["performance"]["theta"]
        ambient_loss_factor = (65 - climate_data["temp_air"]) / (90 - 65) * theta

        self.processed_coeff.time_dependent_full["ambient_loss_factor"] = (
            ambient_loss_factor.to_numpy()
        )

        for par in self.input_parameters.performance_data["performance"]:
            if not par == "theta":
                self.processed_coeff.time_independent[par] = (
                    self.input_parameters.performance_data["performance"][par]
                )

        self.processed_coeff.time_independent["charge_rate"] = self.flexibility_data[
            "charge_rate"
        ]
        self.processed_coeff.time_independent["discharge_rate"] = self.flexibility_data[
            "discharge_rate"
        ]
        if (
            "energy_consumption"
            in self.input_parameters.performance_data["performance"]
        ):
            self.processed_coeff.time_independent["energy_consumption"] = (
                self.input_parameters.performance_data["performance"][
                    "energy_consumption"
                ]
            )

    def _calculate_bounds(self):
        """
        Calculates the bounds of the variables used
        """
        super(Stor, self)._calculate_bounds()

        time_steps = len(self.set_t_performance)

        # Output Bounds
        for car in self.component_options.output_carrier:
            self.bounds["output"][car] = np.column_stack(
                (
                    np.zeros(shape=(time_steps)),
                    np.ones(shape=(time_steps))
                    * self.flexibility_data["discharge_rate"],
                )
            )
        # Input Bounds
        for car in self.component_options.input_carrier:
            if car == self.component_options.main_input_carrier:
                self.bounds["input"][car] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps))
                        * self.flexibility_data["charge_rate"],
                    )
                )
            else:
                if (
                    "energy_consumption"
                    in self.input_parameters.performance_data["performance"]
                ):
                    energy_consumption = self.input_parameters.performance_data[
                        "performance"
                    ]["energy_consumption"]
                    self.bounds["input"][car] = np.column_stack(
                        (
                            np.zeros(shape=(time_steps)),
                            np.ones(shape=(time_steps))
                            * self.flexibility_data["charge_rate"]
                            * energy_consumption["in"][car],
                        )
                    )

    def construct_tech_model(self, b_tec, data: dict, set_t_full, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type STOR, resembling a storage technology

        :param b_tec: pyomo block with technology model
        :param dict data: data containing model configuration
        :param set_t_full: pyomo set containing timesteps
        :param set_t_clustered: pyomo set containing clustered timesteps
        :return: pyomo block with technology model
        """

        super(Stor, self).construct_tech_model(b_tec, data, set_t_full, set_t_clustered)

        config = data["config"]

        # DATA OF TECHNOLOGY
        coeff_td = self.processed_coeff.time_dependent_used
        coeff_ti = self.processed_coeff.time_independent
        dynamics = self.processed_coeff.dynamics

        # sequence_storage = self.sequence
        if config["optimization"]["typicaldays"]["N"]["value"] == 0:
            sequence_storage = self.sequence
        elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
            sequence_storage = data["k_means_specs"]["sequence"]
        elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
            sequence_storage = self.sequence

        if config["optimization"]["timestaging"]["value"] != 0:
            nr_timesteps_averaged = config["optimization"]["timestaging"]["value"]
        else:
            nr_timesteps_averaged = 1

        # Additional parameters
        eta_in = coeff_ti["eta_in"]
        eta_out = coeff_ti["eta_out"]
        eta_lambda = coeff_ti["lambda"]
        charge_rate = coeff_ti["charge_rate"]
        discharge_rate = coeff_ti["discharge_rate"]
        ambient_loss_factor = coeff_td["ambient_loss_factor"]

        # Additional decision variables
        b_tec.var_storage_level = pyo.Var(
            self.set_t_full,
            domain=pyo.NonNegativeReals,
            bounds=(b_tec.para_size_min, b_tec.para_size_max),
        )

        # Size constraint
        def init_size_constraint(const, t):
            return b_tec.var_storage_level[t] <= b_tec.var_size

        b_tec.const_size = pyo.Constraint(self.set_t_full, rule=init_size_constraint)

        # Storage level calculation
        def init_storage_level(const, t):
            if t == 1:
                # couple first and last time interval: storageLevel[1] ==
                # storageLevel[end] * (1-self_discharge)^nr_timesteps_averaged +
                # - storageLevel[end] * ambient_loss_factor[end-1]^nr_timesteps_averaged +
                # + (eta_in * input[] +1/eta_out * output[]) *
                # (sum (1-self_discharge)^i for i in [0, nr_timesteps_averaged])
                #
                # soc[1] = soc[end] + input[seq] - output[seq]

                return b_tec.var_storage_level[t] == b_tec.var_storage_level[
                    max(self.set_t_full)
                ] * (1 - eta_lambda) ** nr_timesteps_averaged - b_tec.var_storage_level[
                    max(self.set_t_full)
                ] * ambient_loss_factor[
                    sequence_storage[t - 1] - 1
                ] ** nr_timesteps_averaged + (
                    eta_in
                    * self.input[
                        sequence_storage[t - 1],
                        self.component_options.main_input_carrier,
                    ]
                    - 1
                    / eta_out
                    * self.output[
                        sequence_storage[t - 1],
                        self.component_options.main_input_carrier,
                    ]
                ) * sum(
                    (1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged)
                )
            else:  # all other time intervals
                return b_tec.var_storage_level[t] == b_tec.var_storage_level[t - 1] * (
                    1 - eta_lambda
                ) ** nr_timesteps_averaged - b_tec.var_storage_level[
                    t
                ] * ambient_loss_factor[
                    sequence_storage[t - 1] - 1
                ] ** nr_timesteps_averaged + (
                    eta_in
                    * self.input[
                        sequence_storage[t - 1],
                        self.component_options.main_input_carrier,
                    ]
                    - 1
                    / eta_out
                    * self.output[
                        sequence_storage[t - 1],
                        self.component_options.main_input_carrier,
                    ]
                ) * sum(
                    (1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged)
                )

        b_tec.const_storage_level = pyo.Constraint(
            self.set_t_full, rule=init_storage_level
        )

        # CONSTRAINTS FOR BIDIRECTIONAL STORAGE
        if self.component_options.allow_only_one_direction:

            # Cut according to Morales-Espana "LP Formulation for Optimal Investment and
            # Operation of Storage Including Reserves"
            def init_cut_bidirectional(const, t):
                if self.flexibility_data["power_energy_ratio"] == "fixedcapacity":
                    return (
                        self.output[t, self.component_options.main_input_carrier]
                        + self.input[t, self.component_options.main_input_carrier]
                        <= b_tec.var_size
                    )
                else:
                    # output[t]/discharge_rate + input[t]/charge_rate <= storSize
                    return (
                        self.output[t, self.component_options.main_input_carrier]
                        / discharge_rate
                        + self.input[t, self.component_options.main_input_carrier]
                        / charge_rate
                        <= b_tec.var_size
                    )

            b_tec.const_cut_bidirectional = pyo.Constraint(
                self.set_t_performance, rule=init_cut_bidirectional
            )

            if self.component_options.allow_only_one_direction_precise:

                self.big_m_transformation_required = 1
                s_indicators = range(0, 2)

                def init_input_output(dis, t, ind):
                    if ind == 0:  # input only

                        def init_output_to_zero(const, car_output):
                            return self.output[t, car_output] == 0

                        dis.const_output_to_zero = pyo.Constraint(
                            b_tec.set_output_carriers, rule=init_output_to_zero
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
        def init_maximal_charge(const, t):
            if self.flexibility_data["power_energy_ratio"] == "fixedcapacity":
                return (
                    self.input[t, self.component_options.main_input_carrier]
                    <= charge_rate
                )
            else:
                return (
                    self.input[t, self.component_options.main_input_carrier]
                    <= b_tec.var_capacity_charge
                )

        b_tec.const_max_charge = pyo.Constraint(
            self.set_t_performance, rule=init_maximal_charge
        )

        def init_maximal_discharge(const, t):
            if self.flexibility_data["power_energy_ratio"] == "fixedcapacity":
                return (
                    self.output[t, self.component_options.main_input_carrier]
                    <= discharge_rate
                )
            else:
                return (
                    self.output[t, self.component_options.main_input_carrier]
                    <= b_tec.var_capacity_discharge
                )

        b_tec.const_max_discharge = pyo.Constraint(
            self.set_t_performance, rule=init_maximal_discharge
        )

        # if the charging / discharging rates are fixed or flexible as a ratio of the energy capacity:
        if not self.flexibility_data["power_energy_ratio"] == "fixedcapacity":

            def init_max_capacity_charge(const):
                if self.flexibility_data["power_energy_ratio"] == "fixedratio":
                    return b_tec.var_capacity_charge == charge_rate * b_tec.var_size
                else:
                    return b_tec.var_capacity_charge <= charge_rate * b_tec.var_size

            b_tec.const_max_cap_charge = pyo.Constraint(rule=init_max_capacity_charge)

            def init_max_capacity_discharge(const):
                if self.flexibility_data["power_energy_ratio"] == "fixedratio":
                    # dischargeCapacity == dischargeRate * storSize
                    return (
                        b_tec.var_capacity_discharge == discharge_rate * b_tec.var_size
                    )
                else:
                    # dischargeCapacity <= dischargeRate * storSize
                    return (
                        b_tec.var_capacity_discharge <= discharge_rate * b_tec.var_size
                    )

            b_tec.const_max_cap_discharge = pyo.Constraint(
                rule=init_max_capacity_discharge
            )

        # Energy consumption charging/discharging
        if "energy_consumption" in coeff_ti:
            energy_consumption = coeff_ti["energy_consumption"]
            if "in" in energy_consumption:
                b_tec.set_energyconsumption_carriers_in = pyo.Set(
                    initialize=energy_consumption["in"].keys()
                )

                def init_energyconsumption_in(const, t, car):
                    # e.g electricity_cons[t] == input[t] * energy_cons[electricity]
                    return (
                        self.input[t, car]
                        == self.input[t, self.component_options.main_input_carrier]
                        * energy_consumption["in"][car]
                    )

                b_tec.const_energyconsumption_in = pyo.Constraint(
                    self.set_t_performance,
                    b_tec.set_energyconsumption_carriers_in,
                    rule=init_energyconsumption_in,
                )

            if "out" in energy_consumption:
                b_tec.set_energyconsumption_carriers_out = pyo.Set(
                    initialize=energy_consumption["out"].keys()
                )

                def init_energyconsumption_out(const, t, car):
                    # NOTE: even though we call it "energy consumption", this constraint actually
                    # simulates for example the production of electricity from a salt cavern H2 storage,
                    # where H2 is stored at e.g. 130bar and released at e.g. 40bar; the pressure difference can be
                    # used to drive a trubine and create electricity.
                    # e.g electricity_prod[t] == output[t] * energy_cons[electricity]
                    return (
                        self.output[t, car]
                        == self.output[t, self.component_options.main_input_carrier]
                        * energy_consumption["out"][car]
                    )

                b_tec.const_energyconsumption_out = pyo.Constraint(
                    self.set_t_performance,
                    b_tec.set_energyconsumption_carriers_out,
                    rule=init_energyconsumption_out,
                )

        # RAMPING RATES
        if "ramping_time" in dynamics:
            if not dynamics["ramping_time"] == -1:
                b_tec = self._define_ramping_rates(b_tec, data, sequence_storage)

        return b_tec

    def _define_size(self, b_tec):
        """
        Adds variables related to technology size specific to storage technologies (i.e., variables for the charging and
        discharging capacities). Regular technology parameters and variables are taken from the parent class (Technology).

        :param b_tec: pyomo block with technology model
        :return: pyomo block with technology model
        """
        super(Stor, self)._define_size(b_tec)

        coeff_ti = self.processed_coeff.time_independent
        charge_rate = coeff_ti["charge_rate"]
        discharge_rate = coeff_ti["discharge_rate"]

        if not self.flexibility_data["power_energy_ratio"] == "fixedcapacity":
            b_tec.var_capacity_charge = pyo.Var(
                domain=pyo.NonNegativeReals,
                bounds=(0, b_tec.para_size_max * charge_rate),
            )
            b_tec.var_capacity_discharge = pyo.Var(
                domain=pyo.NonNegativeReals,
                bounds=(0, b_tec.para_size_max * discharge_rate),
            )

        return b_tec

    def _define_capex_parameters(self, b_tec, data: dict):
        """
        Defines the capex parameters for a flexible storage technology of capex model 4 (for all other capex models,
        the parameters are defined in the Technology class).

        For capex model 4:
        - para_unit_capex_charging_cap: the capex per unit of charging capacity (i.e., per MW charging).
        - para_unit_capex_charging_cap_annual: annualized capex per unit of charging capacity.
        - para_unit_capex_discharging_cap: the capex per unit of discharging capacity (i.e., per MW discharging).
        - para_unit_capex_discharging_cap_annual: annualized capex per unit of discharging capacity.
        - para_unit_capex: capex per unit of energy storage capacity (i.e., per MWh storage).
        - para_unit_capex_annual: annualized capex per unit of storage capacity.

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """
        super(Stor, self)._define_capex_parameters(b_tec, data)

        config = data["config"]
        economics = self.economics

        # CAPEX PARAMETERS
        capex_model = set_capex_model(config, economics)

        if capex_model == 4:
            discount_rate = set_discount_rate(config, economics)
            fraction_of_year_modelled = data["topology"]["fraction_of_year_modelled"]
            annualization_factor = annualize(
                discount_rate, economics.lifetime, fraction_of_year_modelled
            )
            flexibility = self.flexibility_data

            # additional parameters needed for a flexible storage system
            b_tec.para_unit_capex_charging_cap = pyo.Param(
                domain=pyo.Reals,
                initialize=flexibility["capex_charging_power"],
                mutable=True,
            )
            b_tec.para_unit_capex_discharging_cap = pyo.Param(
                domain=pyo.Reals,
                initialize=flexibility["capex_discharging_power"],
                mutable=True,
            )
            b_tec.para_unit_capex = pyo.Param(
                domain=pyo.Reals,
                initialize=economics.capex_data["unit_capex"],
                mutable=True,
            )

            # annualized parameters for a flexible storage system
            b_tec.para_unit_capex_charging_cap_annual = pyo.Param(
                domain=pyo.Reals,
                initialize=(annualization_factor * b_tec.para_unit_capex_charging_cap),
                mutable=True,
            )
            b_tec.para_unit_capex_discharging_cap_annual = pyo.Param(
                domain=pyo.Reals,
                initialize=(
                    annualization_factor * b_tec.para_unit_capex_discharging_cap
                ),
                mutable=True,
            )
            b_tec.para_unit_capex_annual = pyo.Param(
                domain=pyo.Reals,
                initialize=(annualization_factor * b_tec.para_unit_capex),
                mutable=True,
            )

        return b_tec

    def _define_capex_variables(self, b_tec, data: dict):
        """
        Defines variables related to storage technology capex for a flexible storage technology (with capex model 4).

        - var_capex_charging_cap: the capex of the charging capacity (MW) installed.
        - var_capex_discharging_cap: the capex of the discharging capacity (MW) installed.
        - var_capex_energy_cap: the capex of the energy storage capacity (MWh) installed.

        For a non-flexible storage technology (with fixed charging and discharging capacities), the capex
        variables are taken from the parent class (Technology).

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """

        super(Stor, self)._define_capex_variables(b_tec, data)

        config = data["config"]
        economics = self.economics
        capex_model = set_capex_model(config, economics)

        if capex_model == 4:
            coeff_ti = self.processed_coeff.time_independent

            # BOUNDS
            max_capex_charging_cap = (
                b_tec.para_unit_capex_charging_cap_annual
                * coeff_ti["charge_rate"]
                * b_tec.para_size_max
            )
            max_capex_discharging_cap = (
                b_tec.para_unit_capex_discharging_cap_annual
                * coeff_ti["discharge_rate"]
                * b_tec.para_size_max
            )
            max_capex_energy_cap = b_tec.para_unit_capex_annual * b_tec.para_size_max

            # VARIABLES
            b_tec.var_capex_charging_cap = pyo.Var(
                domain=pyo.NonNegativeReals, bounds=(0, max_capex_charging_cap)
            )
            b_tec.var_capex_discharging_cap = pyo.Var(
                domain=pyo.NonNegativeReals, bounds=(0, max_capex_discharging_cap)
            )
            b_tec.var_capex_energy_cap = pyo.Var(
                domain=pyo.NonNegativeReals, bounds=(0, max_capex_energy_cap)
            )

        return b_tec

    def _define_capex_constraints(self, b_tec, data: dict):
        """
        Defines constraints related to capex for a flexible storage technology of capex model 4, for which the capex
        comprises the capex for the charging capacity, the discharging capacity and the energy storage capacity.
        For a non-flexible storage technology, the capex constraints are taken from the parent class (Technology).

        :param b_tec: pyomo block with technology model
        :param dict data: dict containing model information
        :return: pyomo block with technology model
        """

        super(Stor, self)._define_capex_constraints(b_tec, data)

        config = data["config"]
        economics = self.economics
        capex_model = set_capex_model(config, economics)

        if capex_model == 4:

            # CAPEX constraints for flexible storage technologies
            # CAPEX_chargeCapacity = chargeCapacity * unitCost_chargeCapacity
            b_tec.const_capex_charging_cap = pyo.Constraint(
                expr=b_tec.var_capacity_charge
                * b_tec.para_unit_capex_charging_cap_annual
                == b_tec.var_capex_charging_cap
            )
            # CAPEX_dischargeCapacity = dischargeCapacity * unitCost_dischargeCapacity
            b_tec.const_capex_discharging_cap = pyo.Constraint(
                expr=b_tec.var_capacity_discharge
                * b_tec.para_unit_capex_discharging_cap_annual
                == b_tec.var_capex_discharging_cap
            )
            # CAPEX_storSize = storSize * unitCost_storSize
            b_tec.const_capex_energy_cap = pyo.Constraint(
                expr=b_tec.var_size * b_tec.para_unit_capex_annual
                == b_tec.var_capex_energy_cap
            )
            # CAPEX = CAPEX_chargeCapacity + CAPEX_dischargeCapacity + CAPEX_storSize
            b_tec.const_capex_aux = pyo.Constraint(
                expr=b_tec.var_capex_charging_cap
                + b_tec.var_capex_discharging_cap
                + b_tec.var_capex_energy_cap
                == b_tec.var_capex_aux
            )

        return b_tec

    def write_results_tec_design(self, h5_group, model_block):
        """
        Function to report technology design

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super(Stor, self).write_results_tec_design(h5_group, model_block)

        if self.flexibility_data["power_energy_ratio"] == "flexratio":
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

    def write_results_tec_operation(self, h5_group, model_block):
        """
        Function to report technology operation

        :param model_block: pyomo network block
        :param h5_group: h5 group to write to
        """
        super(Stor, self).write_results_tec_operation(h5_group, model_block)

        h5_group.create_dataset(
            "storage_level",
            data=[model_block.var_storage_level[t].value for t in self.set_t_full],
        )

    def _define_ramping_rates(self, b_tec, data, sequence_storage):
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
        if "ramping_const_int" in dynamics and dynamics["ramping_const_int"] == 1:

            s_indicators = range(0, 3)

            def init_ramping_operation_on(dis, t, ind):
                if t > 1:
                    if ind == 0:  # ramping constrained x[t] == x[t-1]
                        dis.const_ramping_on = pyo.Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 0
                        )

                        def init_ramping_down_rate_operation_in(const):
                            # -rampingRate <= input[t] - input[t-1]
                            return (
                                -ramping_rate
                                <= self.input[
                                    t, self.component_options.main_input_carrier
                                ]
                                - self.input[
                                    t - 1, self.component_options.main_input_carrier
                                ]
                            )

                        dis.const_ramping_down_rate_in = pyo.Constraint(
                            rule=init_ramping_down_rate_operation_in
                        )

                        def init_ramping_up_rate_operation_in(const):
                            # input[t] - input[t-1] <= rampingRate
                            return (
                                self.input[t, self.component_options.main_input_carrier]
                                - self.input[
                                    t - 1, self.component_options.main_input_carrier
                                ]
                                <= ramping_rate
                            )

                        dis.const_ramping_up_rate_in = pyo.Constraint(
                            rule=init_ramping_up_rate_operation_in
                        )

                        def init_ramping_down_rate_operation_out(const):
                            # -rampingRate <= output[t] - output[t-1]

                            return -ramping_rate <= sum(
                                self.output[t, car_output]
                                - self.output[t - 1, car_output]
                                for car_output in b_tec.set_output_carriers
                            )

                        dis.const_ramping_down_rate_out = pyo.Constraint(
                            rule=init_ramping_down_rate_operation_out
                        )

                        def init_ramping_up_rate_operation_out(const):
                            # output[t] - output[t-1] <= rampingRate
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

                    elif ind == 1:  # startup, no ramping constraint x[t] - x[t-1] == 1
                        dis.const_ramping_on = pyo.Constraint(
                            expr=b_tec.var_x[t] - b_tec.var_x[t - 1] == 1
                        )

                    else:  # shutdown, no ramping constraint x[t] - x[t-1] == -1
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
                bounds_rr_full = {"input": {}, "output": {}}

                # Output Bounds
                for carr in self.component_options.output_carrier:
                    bounds_rr_full["output"][carr] = np.column_stack(
                        (
                            np.zeros(shape=(len(self.set_t_full))),
                            np.ones(shape=(len(self.set_t_full)))
                            * self.flexibility_data["discharge_rate"],
                        )
                    )

                # Input Bounds
                for carr in self.component_options.input_carrier:
                    if carr == self.component_options.main_input_carrier:
                        bounds_rr_full["input"][carr] = np.column_stack(
                            (
                                np.zeros(shape=(len(self.set_t_full))),
                                np.ones(shape=(len(self.set_t_full)))
                                * self.flexibility_data["charge_rate"],
                            )
                        )
                    else:
                        if (
                            "energy_consumption"
                            in self.input_parameters.performance_data["performance"]
                        ):
                            energy_consumption = self.input_parameters.performance_data[
                                "performance"
                            ]["energy_consumption"]
                            bounds_rr_full["input"][carr] = np.column_stack(
                                (
                                    np.zeros(shape=(len(self.set_t_full))),
                                    np.ones(shape=(len(self.set_t_full)))
                                    * self.flexibility_data["charge_rate"]
                                    * energy_consumption["in"][carr],
                                )
                            )

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
                        for car_output in b_tec.set_output_carriers
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
                            for car_output in b_tec.set_output_carriers
                        )
                        <= ramping_rate
                    )
                else:
                    return pyo.Constraint.Skip

            b_tec.const_ramping_up_rate_output = pyo.Constraint(
                self.set_t_performance, rule=init_ramping_down_rate_output
            )

        return b_tec
