"""
TODO:
- Lets change the order of functions to match the call order from construct_tech_model
- We need to be careful with inflow and input max/mins
- Adopt documentation
"""


from pyomo.environ import *
from pyomo.gdp import *
import pandas as pd
import numpy as np
from pathlib import Path

from src.components.technologies.utilities import FittedPerformance
from src.components.technologies.technology import Technology
from src.components.utilities import perform_disjunct_relaxation
from src.components.utilities import annualize, set_discount_rate
from src.components.technologies.specificTechnologies.utilities import fit_turbomachinery, fit_turbomachinery_capex, fit_turbomachinery_general, fit_turbomachinery_capex_general


class OceanBattery(Technology):
    """
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
        E_{t} = E_{t-1} * (1 - \\lambda_1) - \\lambda_2(\\Theta) * E_{t-1} + {\\eta}_{in} * Input_{t} - 1 / {\\eta}_{out} * Output_{t}

    - If ``allow_only_one_direction == 1``, then only input or output can be unequal to zero in each respective time
      step (otherwise, simultanous charging and discharging can lead to unwanted 'waste' of energy/material).

    :param obj model: instance of a pyomo model
    :param obj b_tec: technology block
    :param tec_data: technology data
    :return: technology block
    """
    def __init__(self,
                 tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()
        self.save_specific_design = None
        self.bounds = {}

    def fit_technology_performance(self, node_data):
        """
        Fits ocean battery

        :param node_data: data on node
        """
        # Coefficients
        for par in self.performance_data['performance']:
            self.fitted_performance.coefficients[par] = self.performance_data['performance'][par]

        # Time dependent coefficients
        self.fitted_performance.time_dependent_coefficients = 1

        # Fit pump
        pump_data = {}
        pump_data['type'] = 'pump'
        pump_data['subtype'] = self.fitted_performance.coefficients['pump_type']
        pump_data['min_power'] = 0
        pump_data['nominal_head'] = self.fitted_performance.coefficients['nominal_head']
        pump_data['frequency'] = self.fitted_performance.coefficients['frequency']
        pump_data['pole_pairs'] = self.fitted_performance.coefficients['pole_pairs']
        self.performance_data['pump'] = fit_turbomachinery_general(pump_data)

        pump_data['P_design'] = self.performance_data['pump']['design']['P_design']
        pump_data['capex_constant_a'] = 1753
        pump_data['capex_constant_b'] = 0.9623
        pump_data['capex_constant_c'] = -0.3566
        self.economics.capex_data['pump'] = fit_turbomachinery_capex_general(pump_data)

        # Fit turbine
        turbine_data = {}
        turbine_data['type'] = 'turbine'
        turbine_data['subtype'] = self.fitted_performance.coefficients['turbine_type']
        turbine_data['min_power'] = 0
        turbine_data['nominal_head'] = self.fitted_performance.coefficients['nominal_head']
        turbine_data['frequency'] = self.fitted_performance.coefficients['frequency']
        turbine_data['pole_pairs'] = self.fitted_performance.coefficients['pole_pairs']
        self.performance_data['turbine'] = fit_turbomachinery_general(turbine_data)

        turbine_data['P_design'] = self.performance_data['turbine']['design']['P_design']
        turbine_data['capex_constant_a'] = 2.927
        turbine_data['capex_constant_b'] = 1.174
        turbine_data['capex_constant_c'] = -0.4933
        self.economics.capex_data['turbine'] = fit_turbomachinery_capex_general(turbine_data)

        # Derive bounds
        climate_data = node_data.data['climate_data']
        time_steps = len(climate_data)
        # pump_slots = self.fitted_performance.coefficients['pump_slots']
        # turbine_slots = self.fitted_performance.coefficients['turbine_slots']

        # Input bounds
        for car in self.performance_data['input_carrier']:
            self.fitted_performance.bounds['input'][car] = np.column_stack((np.zeros(shape=time_steps),
                                                            np.ones(shape=time_steps) * 1000))

        # Output bounds
        for car in self.performance_data['output_carrier']:
            self.fitted_performance.bounds['output'][car] = np.column_stack((np.zeros(shape=time_steps),
                                                             np.ones(shape=time_steps) * 1000))

    def construct_tech_model(self, b_tec, energyhub):
        """
        Adds constraints to technology blocks for tec_type Ocean_Batteru, resembling a storage technology
        """
        super(OceanBattery, self).construct_tech_model(b_tec, energyhub)

        self.save_specific_design = energyhub.configuration.reporting.save_path

        nr_timesteps_averaged = energyhub.model_information.averaged_data_specs.nr_timesteps_averaged

        # Global parameters
        coeff = self.fitted_performance.coefficients
        configuration = energyhub.configuration
        economics = self.economics
        discount_rate = set_discount_rate(configuration, economics)
        fraction_of_year_modelled = energyhub.topology.fraction_of_year_modelled
        annualization_factor = annualize(discount_rate, economics.lifetime, fraction_of_year_modelled)

        # Method sections
        b_tec = self._define_vars(b_tec)
        b_tec = self._define_storage_level(b_tec, nr_timesteps_averaged)
        b_tec = self._define_turbine_performance(b_tec, energyhub)
        b_tec = self._define_pump_performance(b_tec, energyhub)
        #
        # # Aggregate Input/Output
        # def init_total_input(const, t, car):
        #     return b_tec.var_input[t, car] == \
        #            sum(b_tec.var_input_pump[t, pump] for pump in b_tec.set_pump_slots)
        # b_tec.const_total_input = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_total_input)
        #
        # def init_total_output(const, t, car):
        #     return b_tec.var_output[t, car] == \
        #            sum(b_tec.var_output_turbine[t, turbine] for turbine in b_tec.set_turbine_slots)
        # b_tec.const_total_output = Constraint(self.set_t, b_tec.set_output_carriers, rule=init_total_output)
        #
        # # CAPEX Calculation
        # b_tec.const_capex_aux = Constraint(expr=b_tec.para_unit_capex_reservoir_annual * b_tec.var_size +
        #                                         sum(b_tec.var_capex_turbine[turbine] for
        #                                                              turbine in b_tec.set_turbine_slots) +
        #                                         sum(b_tec.var_capex_pump[pump] for pump in b_tec.set_pump_slots) ==
        #                                         b_tec.var_capex_aux)

        return b_tec

    def _define_vars(self, b_tec):

        # Additional parameters
        coeff = self.fitted_performance.coefficients

        # TODO: DEFINE
        max_size = 200
        max_flow_pump = 8
        max_flow_turbine = 8

        # Additional decision variables
        b_tec.var_storage_level = Var(self.set_t_full, domain=NonNegativeReals,
                                      bounds=(self.size_min, self.size_max))
        b_tec.var_total_inflow = Var(self.set_t_full, domain=NonNegativeReals,
                                     bounds=(0, max_flow_pump * max_size))
        b_tec.var_total_outflow = Var(self.set_t_full, domain=NonNegativeReals,
                                      bounds=(0, max_flow_turbine * max_size))

        return b_tec


    def _define_storage_level(self, b_tec, nr_timesteps_averaged):

        coeff = self.fitted_performance.coefficients

        # Additional parameters
        eta_lambda = coeff['lambda']
        reservoir_size = 250000

        # Fill constraints
        def init_fill_constraint_up(const, t):
            return b_tec.var_storage_level[t] <= b_tec.var_size * reservoir_size
        b_tec.const_size_up = Constraint(self.set_t_full, rule=init_fill_constraint_up)

        # Storage level calculation
        def init_storage_level(const, t):
            if t == 1:  # couple first and last time interval
                return b_tec.var_storage_level[t] == \
                       b_tec.var_storage_level[max(self.set_t_full)] * (1 - eta_lambda) ** nr_timesteps_averaged + \
                       (b_tec.var_total_inflow[t] - b_tec.var_total_outflow[t]) * \
                       sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged))
            else:  # all other time intervals
                return b_tec.var_storage_level[t] == \
                       b_tec.var_storage_level[t - 1] * (1 - eta_lambda) ** nr_timesteps_averaged + \
                       (b_tec.var_total_inflow[t]- b_tec.var_total_outflow[t]) * \
                       sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged))

        b_tec.const_storage_level = Constraint(self.set_t_full, rule=init_storage_level)

        return b_tec

    def _define_turbine_performance(self, b_tec, energyhub):
        """
        Defines turbine performance
        """

        # eta_turbine = coeff['eta_turbine']
        eta_turbine = 0.8
        head_correction = 1/3.6 * 9.81 * self.fitted_performance.coefficients['nominal_head'] * 10 ** -6
        q_up = 10

        def init_output(const, t, car):
            return b_tec.var_output[t, car] == b_tec.var_total_outflow[t] * eta_turbine * head_correction
        b_tec.const_output = Constraint(self.set_t_full, b_tec.set_output_carriers, rule=init_output)

        def init_outflow_up(const, t):
            return b_tec.var_total_outflow[t] <= b_tec.var_size * q_up
        b_tec.const_outflow_up = Constraint(self.set_t_full, rule=init_outflow_up)

        return b_tec

    def _define_pump_performance(self, b_tec, energyhub):
        """
        Defines turbine performance
        """
        # eta_turbine = coeff['eta_turbine']
        eta_pump = 0.8
        head_correction = 1/3.6 * 9.81 * self.fitted_performance.coefficients['nominal_head'] * 10 ** -6
        q_up = 10

        def init_input(const, t, car):
            return b_tec.var_total_inflow[t] == b_tec.var_input[t, car] * eta_pump * head_correction
        b_tec.const_input = Constraint(self.set_t_full, b_tec.set_input_carriers, rule=init_input)

        def init_inflow_up(const, t):
            return b_tec.var_total_outflow[t] <= b_tec.var_size * q_up
        b_tec.const_inflow_up = Constraint(self.set_t_full, rule=init_inflow_up)

        return b_tec

    def report_results(self, b_tec):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        super(OceanBattery, self).report_results(b_tec)

        self.results['time_dependent']['storagelevel'] = [b_tec.var_storage_level[t].value for t in self.set_t_full]
        self.results['time_dependent']['total_inflow'] = [b_tec.var_total_inflow[t].value for t in self.set_t_full]
        self.results['time_dependent']['total_outflow'] = [b_tec.var_total_outflow[t].value for t in self.set_t_full]

        return self.results
