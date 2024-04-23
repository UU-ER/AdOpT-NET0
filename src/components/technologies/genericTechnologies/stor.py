from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import numpy as np

from ..utilities import FittedPerformance
from ..technology import Technology
from src.components.utilities import annualize, set_discount_rate


class Stor(Technology):

    def __init__(self,
                 tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()

        self.flexibility_data = tec_data['Flexibility']

    def fit_technology_performance(self, node_data):
        """
        Fits storage technology and returns fitted parameters as a dict

        :param node_data: contains data on demand, climate data, etc.
        :param performance_data: contains performance parameters for the storage technology (efficiencies and losses).
        :param flexibility_data: options for type of storage optimization (fixed: charging and discharging power capacities are fixed as a ratio of the energy capacity; flexible ("flex"): charging and discharging power capacities can be optimized separately from the energy capacity.
        """

        climate_data = node_data.data['climate_data']

        time_steps = len(climate_data)

        # Calculate ambient loss factor
        theta = self.performance_data['performance']['theta']
        ambient_loss_factor = (65 - climate_data['temp_air']) / (90 - 65) * theta

        # Output Bounds
        for car in self.performance_data['output_carrier']:
            self.fitted_performance.bounds['output'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                                             np.ones(shape=(time_steps)) * self.size_max *
                                                                             self.flexibility_data['discharge_rate']))
        # Input Bounds
        for car in self.performance_data['input_carrier']:
            self.fitted_performance.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                                            np.ones(shape=(time_steps)) * self.size_max *
                                                                            self.flexibility_data['charge_rate']))

        # For a flexibly optimized storage technology (i.e., not a fixed P-E ratio), an adapted CAPEX function is used
        # to account for charging and discharging capacity costs.
        economics = self.economics
        if self.flexibility_data['power_energy_ratio'] == "fixed":
            pass
        elif self.flexibility_data['power_energy_ratio'] == "flex":
            economics.capex_model = 4
        else:
            raise Warning("power_energy_ratio should be either flexible ('flex') or fixed ('fixed')")

        # Coefficients
        self.fitted_performance.coefficients['ambient_loss_factor'] = ambient_loss_factor.to_numpy()
        for par in self.performance_data['performance']:
            if not par == 'theta':
                self.fitted_performance.coefficients[par] = self.performance_data['performance'][par]

        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 1


    def construct_tech_model(self, b_tec, energyhub):
        """
        Adds constraints to technology blocks for tec_type STOR, resembling a storage technology

        
        Note that this technology only works for one carrier, and thus the carrier index is dropped in the below notation.

        **Parameter declarations:**

        - :math:`{\\eta}_{in}`: Charging efficiency

        - :math:`{\\eta}_{out}`: Discharging efficiency

        - :math:`{\\lambda_1}`: Self-Discharging coefficient (independent of environment)

        - :math:`{\\lambda_2(\\Theta)}`: Self-Discharging coefficient (dependent on environment)

        **Variable declarations:**

        - Storage level in :math:`t`: :math:`E_t`

        - Charging in :math:`t`: :math:`Input_{t}`

        - Discharging in :math:`t`: :math:`Output_{t}`

        - :math:`Input_{max}`: Maximal charging capacity in one time-slice (this is a fixed value when the power-energy ratio is fixed).

        - :math:`Output_{max}`: Maximal discharging capacity in one time-slice (this is a fixed value when the power-energy ratio is fixed).

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

        :param obj b_tec: technology block
        :param Energyhub energyhub: energyhub instance
        :return: technology block
        """
        super(Stor, self).construct_tech_model(b_tec, energyhub)

        set_t_full = energyhub.model.set_t_full

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        coeff = self.fitted_performance.coefficients
        rated_power = self.fitted_performance.rated_power

        if 'allow_only_one_direction' in performance_data:
            allow_only_one_direction = performance_data['allow_only_one_direction']
        else:
            allow_only_one_direction = 0

        nr_timesteps_averaged = energyhub.model_information.averaged_data_specs.nr_timesteps_averaged

        # Additional parameters
        eta_in = coeff['eta_in']
        eta_out = coeff['eta_out']
        eta_lambda = coeff['lambda']
        charge_rate = self.flexibility_data['charge_rate']
        discharge_rate = self.flexibility_data['discharge_rate']
        ambient_loss_factor = coeff['ambient_loss_factor']

        # Additional decision variables
        b_tec.var_storage_level = Var(set_t_full, b_tec.set_input_carriers,
                                      domain=NonNegativeReals,
                                      bounds=(b_tec.para_size_min, b_tec.para_size_max))

        b_tec.var_capacity_charge = Var(domain=NonNegativeReals, bounds=(0, b_tec.para_size_max * charge_rate))

        b_tec.var_capacity_discharge = Var(domain=NonNegativeReals, bounds=(0, b_tec.para_size_max * discharge_rate))

        # methods

        b_tec = self._define_stor_capex(b_tec, energyhub)

        # Size constraint
        def init_size_constraint(const, t, car):
            return b_tec.var_storage_level[t, car] <= b_tec.var_size

        b_tec.const_size = Constraint(set_t_full, b_tec.set_input_carriers, rule=init_size_constraint)

        # Storage level calculation
        if energyhub.model_information.clustered_data and not self.modelled_with_full_res:
            def init_storage_level(const, t, car):
                if t == 1:  # couple first and last time interval
                    return b_tec.var_storage_level[t, car] == \
                           b_tec.var_storage_level[max(set_t_full), car] * (1 - eta_lambda) ** nr_timesteps_averaged - \
                           b_tec.var_storage_level[max(set_t_full), car] * ambient_loss_factor[
                               max(set_t_full) - 1] ** nr_timesteps_averaged + \
                           (eta_in * self.input[self.sequence[t - 1], car] - 1 / eta_out * self.output[self.sequence[t - 1], car]) * \
                           sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged))
                else:  # all other time intervalls
                    return b_tec.var_storage_level[t, car] == \
                           b_tec.var_storage_level[t - 1, car] * (1 - eta_lambda) ** nr_timesteps_averaged - \
                           b_tec.var_storage_level[t, car] * ambient_loss_factor[t - 1] ** nr_timesteps_averaged + \
                           (eta_in * self.input[self.sequence[t - 1], car] - 1 / eta_out * self.output[self.sequence[t - 1], car]) * \
                           sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged))

            b_tec.const_storage_level = Constraint(set_t_full, b_tec.set_input_carriers, rule=init_storage_level)
        else:
            def init_storage_level(const, t, car):
                if t == 1:  # couple first and last time interval
                    return b_tec.var_storage_level[t, car] == \
                           b_tec.var_storage_level[max(set_t_full), car] * (1 - eta_lambda) ** nr_timesteps_averaged - \
                           b_tec.var_storage_level[max(set_t_full), car] * ambient_loss_factor[
                               max(set_t_full) - 1] ** nr_timesteps_averaged + \
                           (eta_in * self.input[t, car] - 1 / eta_out * self.output[
                               t, car]) * \
                           sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged))
                else:  # all other time intervalls
                    return b_tec.var_storage_level[t, car] == \
                           b_tec.var_storage_level[t - 1, car] * (1 - eta_lambda) ** nr_timesteps_averaged - \
                           b_tec.var_storage_level[t, car] * ambient_loss_factor[t - 1] ** nr_timesteps_averaged + \
                           (eta_in * self.input[t, car] - 1 / eta_out * self.output[t, car]) * \
                           sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged))
            b_tec.const_storage_level = Constraint(set_t_full, b_tec.set_input_carriers, rule=init_storage_level)

        # This makes sure that only either input or output is larger zero.
        if allow_only_one_direction == 1:
            self.big_m_transformation_required = 1
            s_indicators = range(0, 2)

            # Cut according to Germans work
            def init_cut_bidirectional(const, t, car):
                return (self.output[t, car] / b_tec.var_capacity_discharge + self.input[t, car] /
                        b_tec.var_capacity_charge <= b_tec.var_size)
            b_tec.const_cut_bidirectional = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_cut_bidirectional)

            def init_input_output(dis, t, ind):
                if ind == 0:  # input only
                    def init_output_to_zero(const, car_input):
                        return self.output[t, car_input] == 0
                    dis.const_output_to_zero = Constraint(b_tec.set_input_carriers, rule=init_output_to_zero)

                elif ind == 1:  # output only
                    def init_input_to_zero(const, car_input):
                        return self.input[t, car_input] == 0
                    dis.const_input_to_zero = Constraint(b_tec.set_input_carriers, rule=init_input_to_zero)

            b_tec.dis_input_output = Disjunct(self.set_t, s_indicators, rule=init_input_output)

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_input_output[t, i] for i in s_indicators]
            b_tec.disjunction_input_output = Disjunction(self.set_t, rule=bind_disjunctions)

        # Maximal charging and discharging rates
        def init_maximal_charge(const, t, car):
            return self.input[t, car] <= b_tec.var_capacity_charge
        b_tec.const_max_charge = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_maximal_charge)

        def init_maximal_discharge(const, t, car):
            return self.output[t, car] <= b_tec.var_capacity_discharge

        b_tec.const_max_discharge = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_maximal_discharge)

        # if the charging / discharging rates are fixed as a ratio of the energy capacity:
        if self.flexibility_data["power_energy_ratio"] == "fixed":

            def init_max_capacity_charge(const, t, car):
                return b_tec.var_capacity_charge == charge_rate * b_tec.var_size
            b_tec.const_max_cap_charge = Constraint(self.set_t, b_tec.set_input_carriers,
                                                      rule=init_max_capacity_charge)

            def init_max_capacity_discharge(const, t, car):
                return b_tec.var_capacity_discharge == discharge_rate * b_tec.var_size
            b_tec.const_max_cap_discharge = Constraint(self.set_t, b_tec.set_input_carriers,
                                                      rule=init_max_capacity_discharge)

        # if the charging / discharging rates can be chosen , and determined by the installed
        # charging and discharging power capacity.
        elif self.flexibility_data["power_energy_ratio"] == "flex":

            def init_maximal_capacity_charge(const, t, car):
                return b_tec.var_capacity_charge <= charge_rate * b_tec.var_size
            b_tec.const_max_cap_charge = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_maximal_capacity_charge)

            def init_maximal_capacity_discharge(const, t, car):
                return b_tec.var_capacity_discharge <= discharge_rate * b_tec.var_size
            b_tec.const_max_cap_discharge = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_maximal_capacity_discharge)

        else:
            return warn("the 'power_energy_ratio' parameter should be equal to 'flex' (is flexible) "
                        "or 'fixed' (is constant)")

        # RAMPING RATES
        if "ramping_rate" in self.performance_data:
            if not self.performance_data['ramping_rate'] == -1:
                b_tec = self.__define_ramping_rates(b_tec)

        return b_tec

    def report_results(self, b_tec):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        # TODO add the charging and discharging power capacities to the results
        super(Stor, self).report_results(b_tec)

        for car in b_tec.set_input_carriers:
            self.results['time_dependent']['storagelevel_' + car] = [b_tec.var_storage_level[t, car].value for t in self.set_t_full]

        return self.results

    def _define_stor_capex(self, b_tec, energyhub):

        flexibility = self.flexibility_data
        configuration = energyhub.configuration
        economics = self.economics
        discount_rate = set_discount_rate(configuration, economics)
        fraction_of_year_modelled = energyhub.topology.fraction_of_year_modelled
        annualization_factor = annualize(discount_rate, economics.lifetime, fraction_of_year_modelled)

        if flexibility["power_energy_ratio"] == "fixed":
            pass

        elif flexibility["power_energy_ratio"] == "flex":

            b_tec.para_unit_capex_charging_cap = Param(domain=Reals, initialize=flexibility["capex_charging_power"], mutable=True)
            b_tec.para_unit_capex_discharging_cap = Param(domain=Reals, initialize=flexibility["capex_discharging_power"], mutable=True)
            b_tec.para_unit_capex_energy_cap = Param(domain=Reals, initialize=economics.capex_data['unit_capex'], mutable=True)

            b_tec.para_unit_capex_charging_cap_annual = Param(domain=Reals, initialize=(annualization_factor * b_tec.para_unit_capex_charging_cap), mutable=True)
            b_tec.para_unit_capex_discharging_cap_annual = Param(domain=Reals, initialize=(annualization_factor * b_tec.para_unit_capex_discharging_cap), mutable=True)
            b_tec.para_unit_capex_energy_cap_annual = Param(domain=Reals, initialize=(annualization_factor * b_tec.para_unit_capex_energy_cap), mutable=True)

            max_capex_charging_cap = b_tec.para_unit_capex_charging_cap_annual * flexibility["charge_rate"] * b_tec.para_size_max
            max_capex_discharging_cap = b_tec.para_unit_capex_discharging_cap_annual * flexibility["discharge_rate"] * b_tec.para_size_max
            max_capex_energy_cap = b_tec.para_unit_capex_energy_cap_annual * b_tec.para_size_max

            b_tec.var_capex_charging_cap = Var(domain=NonNegativeReals, bounds=(0, max_capex_charging_cap))
            b_tec.var_capex_discharging_cap = Var(domain=NonNegativeReals, bounds=(0, max_capex_discharging_cap))
            b_tec.var_capex_energy_cap = Var(domain=NonNegativeReals, bounds=(0, max_capex_energy_cap))

            # CAPEX constraint
            b_tec.const_capex_charging_cap = Constraint(expr=b_tec.var_capacity_charge * b_tec.para_unit_capex_charging_cap_annual ==
                                                      b_tec.var_capex_charging_cap)
            b_tec.const_capex_discharging_cap = Constraint(expr=b_tec.var_capacity_discharge * b_tec.para_unit_capex_discharging_cap_annual ==
                                                      b_tec.var_capex_discharging_cap)
            b_tec.const_capex_energy_cap = Constraint(expr=b_tec.var_size * b_tec.para_unit_capex_energy_cap_annual ==
                                                      b_tec.var_capex_energy_cap)
            b_tec.const_capex_aux = Constraint(expr=b_tec.var_capex_charging_cap + b_tec.var_capex_discharging_cap +
                                                    b_tec.var_capex_energy_cap == b_tec.var_capex_aux)

        return b_tec

    def __define_ramping_rates(self, b_tec):
        """
        Constraints the inputs for a ramping rate. Implemented for input and output

        :param b_tec: technology model block
        :return:
        """
        ramping_rate = self.performance_data['ramping_rate']

        def init_ramping_down_rate_input(const, t):
            if t > 1:
                return -ramping_rate <= sum(self.input[t, car_input] - self.input[t - 1, car_input]
                                            for car_input in b_tec.set_input_carriers)
            else:
                return Constraint.Skip

        b_tec.const_ramping_down_rate_input = Constraint(self.set_t, rule=init_ramping_down_rate_input)

        def init_ramping_up_rate_input(const, t):
            if t > 1:
                return sum(self.input[t, car_input] - self.input[t - 1, car_input]
                           for car_input in b_tec.set_input_carriers) <= ramping_rate
            else:
                return Constraint.Skip

        b_tec.const_ramping_up_rate_input = Constraint(self.set_t, rule=init_ramping_up_rate_input)

        def init_ramping_down_rate_output(const, t):
            if t > 1:
                return -ramping_rate <= sum(self.output[t, car_output] - self.output[t - 1, car_output]
                                            for car_output in b_tec.set_ouput_carriers)
            else:
                return Constraint.Skip

        b_tec.const_ramping_down_rate_output = Constraint(self.set_t, rule=init_ramping_down_rate_output)

        def init_ramping_down_rate_output(const, t):
            if t > 1:
                return sum(self.output[t, car_output] - self.output[t - 1, car_output]
                           for car_output in b_tec.set_ouput_carriers) <= ramping_rate
            else:
                return Constraint.Skip

        b_tec.const_ramping_up_rate_output = Constraint(self.set_t, rule=init_ramping_down_rate_output)

        return b_tec
