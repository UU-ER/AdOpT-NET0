from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import numpy as np

from ..utilities import FittedPerformance
from src.components.technologies.technology import Technology
import src.global_variables as global_variables


class Stor(Technology):

    def __init__(self,
                 tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()

    def fit_technology_performance(self, node_data):
        """
        Fits conversion technology type 1 and returns fitted parameters as a dict
        :param node_data: contains data on demand, climate data, etc.
        :param performance_data: contains X and y data of technology performance
        :param performance_function_type: options for type of performance function (linear, piecewise,...)
        :param nr_seg: number of segments on piecewise defined function
        """

        climate_data = node_data.data['climate_data']

        time_steps = len(climate_data)

        # Calculate ambient loss factor
        theta = self.performance_data['performance']['theta']
        ambient_loss_factor = (65 - climate_data['temp_air']) / (90 - 65) * theta

        # Output Bounds
        for car in self.performance_data['output_carrier']:
            self.fitted_performance.bounds['output'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                             np.ones(shape=(time_steps)) * self.performance_data['performance'][
                                                                 'discharge_max']))
        # Input Bounds
        for car in self.performance_data['input_carrier']:
            self.fitted_performance.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                            np.ones(shape=(time_steps)) * self.performance_data['performance'][
                                                                'charge_max']))
        # Coefficients
        self.fitted_performance.coefficients['ambient_loss_factor'] = ambient_loss_factor.to_numpy()
        for par in self.performance_data['performance']:
            if not par == 'theta':
                self.fitted_performance.coefficients[par] = self.performance_data['performance'][par]

        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 1


    def construct_specific_constraints(self, b_tec, energyhub):
        """
        Adds constraints to technology blocks for tec_type STOR, resembling a storage technology

        The performance
        functions are fitted in ``src.model_construction.technology_performance_fitting``.
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

        :param obj model: instance of a pyomo model
        :param obj b_tec: technology block
        :param tec_data: technology data
        :return: technology block
        """

        # Full or reduced resolution
        self.input = b_tec.var_input
        self.output = b_tec.var_output
        self.set_t = energyhub.model.set_t_full

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        coeff = self.fitted_performance.coefficients
        rated_power = self.fitted_performance.rated_power

        if 'allow_only_one_direction' in performance_data:
            allow_only_one_direction = performance_data['allow_only_one_direction']
        else:
            allow_only_one_direction = 0

        nr_timesteps_averaged = global_variables.averaged_data_specs.nr_timesteps_averaged

        # Additional decision variables
        b_tec.var_storage_level = Var(self.set_t, b_tec.set_input_carriers,
                                      domain=NonNegativeReals,
                                      bounds=(b_tec.para_size_min, b_tec.para_size_max))

        # Abdditional parameters
        eta_in = coeff['eta_in']
        eta_out = coeff['eta_out']
        eta_lambda = coeff['lambda']
        charge_max = coeff['charge_max']
        discharge_max = coeff['discharge_max']
        ambient_loss_factor = coeff['ambient_loss_factor']

        # Size constraint
        def init_size_constraint(const, t, car):
            return b_tec.var_storage_level[t, car] <= b_tec.var_size

        b_tec.const_size = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_size_constraint)

        # Storage level calculation
        def init_storage_level(const, t, car):
            if t == 1:  # couple first and last time interval
                return b_tec.var_storage_level[t, car] == \
                       b_tec.var_storage_level[max(self.set_t), car] * (1 - eta_lambda) ** nr_timesteps_averaged - \
                       b_tec.var_storage_level[max(self.set_t), car] * ambient_loss_factor[
                           max(self.set_t) - 1] ** nr_timesteps_averaged + \
                       (eta_in * self.input[t, car] - 1 / eta_out * self.output[t, car]) * \
                       sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged))
            else:  # all other time intervalls
                return b_tec.var_storage_level[t, car] == \
                       b_tec.var_storage_level[t - 1, car] * (1 - eta_lambda) ** nr_timesteps_averaged - \
                       b_tec.var_storage_level[t, car] * ambient_loss_factor[t - 1] ** nr_timesteps_averaged + \
                       (eta_in * self.input[t, car] - 1 / eta_out * self.output[t, car]) * \
                       sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged))

        b_tec.const_storage_level = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_storage_level)

        # This makes sure that only either input or output is larger zero.
        if allow_only_one_direction == 1:
            self.big_m_transformation_required = 1
            s_indicators = range(0, 2)

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
            return self.input[t, car] <= charge_max * b_tec.var_size
        b_tec.const_max_charge = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_maximal_charge)

        def init_maximal_discharge(const, t, car):
            return self.output[t, car] <= discharge_max * b_tec.var_size

        b_tec.const_max_discharge = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_maximal_discharge)

        return b_tec
