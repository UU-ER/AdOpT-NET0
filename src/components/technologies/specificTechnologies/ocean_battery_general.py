from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

import components.utilities
from ..utilities import FittedPerformance, fit_piecewise_function
from ..technology import Technology
from ...utilities import perform_disjunct_relaxation


class OceanBattery(Technology):

    def __init__(self,
                 tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()
        self.save_specific_design = None

    def fit_technology_performance(self, node_data):
        """
        Fits ocean battery

        :param node_data: data on node
        """
        # Climate data & Number of timesteps
        climate_data = node_data.data['climate_data']
        time_steps = len(climate_data)

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
        for par in self.performance_data['performance']:
            self.fitted_performance.coefficients[par] = self.performance_data['performance'][par]

        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 1

        # Number of segments
        nr_segments = self.performance_data['nr_segments']

        # Additional parameters needed for pump / turbine performance calculations
        frequency = self.performance_data['frequency']
        pole_pairs = self.performance_data['pole_pairs']
        N = (120 * frequency) / (pole_pairs * 2)  # RPM
        omega = 2 * np.pi * N / 60  # rad/s

        # calculating the nominal head based on OB type
        water_depth = self.performance_data['performance']['water_depth']
        if water_depth < 22.5:
            raise Warning("Ocean Battery might not be suitable for this depth")
        elif water_depth > 60:
            nominal_head = water_depth - 7.5
        else:
            nominal_head = water_depth + 17.5

        #TODO change this according to new performance data (possibly only one type left?)

        # PUMPS: fitting performance

        # get performance data
        performance_pumps = pd.read_csv('data/technology_data/HydroTechnologies/Pump_performance.csv',
                                        delimiter=";")

        # convert performance data from (omega s, eta) to (Qin, Pin)
        performance_pumps['Q_in'] = performance_pumps.apply(lambda row: ((row['Specific_rotational_speed'] *
                                                                          ((9.81 * nominal_head) ** 0.75)) / omega) ** 2,
                                                            axis=1)

        performance_pumps['P_in'] = performance_pumps.apply(lambda row: (9.81 * 1000 * nominal_head * row['Q_in']) /
                                                                        (row['Efficiency'] / 100) * 10 ** -6, axis=1)

        # group performance by type
        pumps = performance_pumps.groupby('Pump_type')
        radial_data = pumps.get_group('Radial')
        mixedflow_data = pumps.get_group('Mixed_flow')
        axial_data = pumps.get_group('Axial')

        # perform fitting of appropriate performance curve
        if self.performance_data['pump_type'] == "Radial":
            x = radial_data['Q_in']
            y = {}
            y['P_in'] = radial_data['P_in']
            fit_pump = fit_piecewise_function(x, y, nr_segments)
        elif self.performance_data['pump_type'] == "Mixed_flow":
            x = mixedflow_data['Q_in']
            y = {}
            y['P_in'] = mixedflow_data['P_in']
            fit_pump = fit_piecewise_function(x, y, nr_segments)
        elif self.performance_data['pump_type'] == "Axial":
            x = axial_data['Q_in']
            y = {}
            y['P_in'] = axial_data['P_in']
            fit_pump = fit_piecewise_function(x, y, nr_segments)
        else:
            raise Warning("Pump type not defined")

        # TURBINES: fitting performance

        # get performance data
        performance_turbines = pd.read_csv('data/technology_data/Turbine_performance.csv', delimiter=";")

        # convert performance data from (omega s, eta) to (Qout, Pout)
        performance_turbines['Q_out'] = performance_turbines.apply(lambda row: ((row['Specific_rotational_speed'] *
                                                                                 ((9.81 * nominal_head) ** 0.75)) /
                                                                                omega) ** 2,
                                                                   axis=1)

        performance_turbines['P_out'] = performance_turbines.apply(lambda row: 9.81 * 1000 * nominal_head * row['Q_out']
                                                                               * row['Efficiency'] * 10 ** -6, axis=1)

        # group performance by type
        turbines = performance_turbines.groupby('Turbine_type')
        pelton_data = turbines.get_group('Pelton')
        francis_data = turbines.get_group('Francis')
        kaplan_data = turbines.get_group('Kaplan')

        # perform fitting of appropriate performance curve

        if self.performance_data['turbine_type'] == "Pelton":
            x = pelton_data['Q_out']
            y = {}
            y['P_out'] = pelton_data['P_out']
            fit_turbine = fit_piecewise_function(x, y, nr_segments)
        elif self.performance_data['turbine_type'] == "Kaplan":
            x = kaplan_data['Q_out']
            y = {}
            y['P_out'] = kaplan_data['P_out']
            fit_turbine = fit_piecewise_function(x, y, nr_segments)
        elif self.performance_data['turbine_type'] == "Francis":
            x = francis_data['Q_out']
            y = {}
            y['P_out'] = francis_data['P_out']
            fit_turbine = fit_piecewise_function(x, y, nr_segments)
        else:
            raise Warning("turbine type not defined")

        alpha1_pump = fit_pump['P_in']['alpha1']
        alpha2_pump = fit_pump['P_in']['alpha2']
        bp_x_pump = fit_pump['P_in']['bp_x']
        alpha1_turbine = fit_turbine['P_out']['alpha1']
        alpha2_turbine = fit_turbine['P_out']['alpha2']
        bp_x_turbine = fit_turbine['P_out']['bp_x']

        # Output Bounds
        for car in self.performance_data['output_carrier']:
            self.fitted_performance.bounds['output'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                             np.ones(shape=(time_steps)) * self.performance_data
                                                                             ['performance']['discharge_max']))
        # Input Bounds
        for car in self.performance_data['input_carrier']:
            self.fitted_performance.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                            np.ones(shape=(time_steps)) * self.performance_data
                                                                            ['performance']['charge_max']))

        # Coefficients
        for par in self.performance_data['performance']:
            self.fitted_performance.coefficients[par] = self.performance_data['performance'][par]

        self.fitted_performance.coefficients['nominal_head'] = nominal_head
        self.fitted_performance.coefficients['alpha1_pump'] = alpha1_pump
        self.fitted_performance.coefficients['alpha2_pump'] = alpha2_pump
        self.fitted_performance.coefficients['bp_x_pump'] = bp_x_pump
        self.fitted_performance.coefficients['alpha1_turbine'] = alpha1_turbine
        self.fitted_performance.coefficients['alpha2_turbine'] = alpha2_turbine
        self.fitted_performance.coefficients['bp_x_turbine'] = bp_x_turbine
        self.fitted_performance.coefficients['Kaplan']['bp_x_turbine'] = bp_x_turbine
        self.fitted_performance.coefficients['Pelton']['bp_x_turbine'] = bp_x_turbine

        # Time dependent coefficients
        self.fitted_performance.coefficients.time_dependent_coefficients = 0

    def construct_tech_model(self, b_tec, energyhub):
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
        super(OceanBattery, self).construct_tech_model(b_tec, energyhub)

        self.save_specific_design = energyhub.configuration.reporting.save_path

        set_t_full = energyhub.model.set_t_full

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        coeff = self.fitted_performance.coefficients

        # Additional parameters
        eta_in = coeff['eta_in']
        eta_out = coeff['eta_out']
        eta_lambda = coeff['lambda']
        min_fill = coeff['min_fill']

        # Additional parameters for general model: dependent on outcome of specific model = based on pump and turbine
        # sizes. Should be written in terms of volume rather than power in the json file.

        charge_min = coeff['charge_min']
        discharge_min = coeff['discharge_min']
        charge_max = coeff['charge_max']
        discharge_max = coeff['discharge_max']

        nr_timesteps_averaged = energyhub.model_information.averaged_data_specs.nr_timesteps_averaged

        # Additional decision variables
        b_tec.var_storage_level = Var(set_t_full,
                                      domain=NonNegativeReals,
                                      bounds=(b_tec.para_size_min, b_tec.para_size_max))
        b_tec.var_total_inflow = Var(set_t_full,
                                     domain=NonNegativeReals,
                                     bounds=(b_tec.para_size_min, b_tec.para_size_max))
        b_tec.var_total_outflow = Var(set_t_full,
                                      domain=NonNegativeReals,
                                      bounds=(b_tec.para_size_min, b_tec.para_size_max))

        # Fill constraints
        def init_size_constraint_up(const, t):
            return b_tec.var_storage_level[t] <= b_tec.var_size

        b_tec.const_size_up = Constraint(set_t_full, rule=init_size_constraint_up)

        def init_size_constrain_low(const, t):
            return b_tec.var_storage_level[t] >= min_fill * b_tec.var_size

        b_tec.const_size_low = Constraint(set_t_full, rule=init_size_constrain_low)

        # Storage level calculation
        def init_storage_level(const, t, car):
            if t == 1:  # couple first and last time interval
                return b_tec.var_storage_level[t] == \
                       b_tec.var_storage_level[max(set_t_full)] * (1 - eta_lambda) ** nr_timesteps_averaged + \
                       (b_tec.var_total_inflow[t] - b_tec.var_total_outflow[t]) * \
                       sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged))
            else:  # all other time intervals
                return b_tec.var_storage_level[t] == \
                       b_tec.var_storage_level[t - 1] * (1 - eta_lambda) ** nr_timesteps_averaged + \
                       (b_tec.var_total_inflow[t] - b_tec.var_total_outflow[t]) * \
                       sum((1 - eta_lambda) ** i for i in range(0, nr_timesteps_averaged))

        b_tec.const_storage_level = Constraint(set_t_full, b_tec.set_input_carriers, rule=init_storage_level)

        # Charging: disjuncts because either not or within bounds
        s_indicators_charge = range(0, 2)

        def init_charging(dis, t, ind):
            if ind == 0:  # not charging
                def init_input_to_zero(const):
                    return b_tec.var_total_inflow[t] == 0

                dis.const_input_to_zero = Constraint(rule=init_input_to_zero)

            elif ind == 1:  # charging
                def init_input_low(const):
                    return charge_min * b_tec.var_size <= b_tec.var_total_inflow[t]

                dis.const_input_low = Constraint(rule=init_input_low)

                def init_input_up(const):
                    return b_tec.var_total_inflow[t] <= charge_max * b_tec.var_size

                dis.const_input_up = Constraint(rule=init_input_up)

        b_tec.dis_charging = Disjunct(set_t_full, s_indicators_charge, rule=init_charging)

        # Bind disjuncts
        def bind_disjunctions_charge(dis, t):
            return [b_tec.dis_charging[t, i] for i in s_indicators_charge]

        b_tec.disjunction_charging = Disjunction(set_t_full, rule=bind_disjunctions_charge)


        # INPUT FORMULATIONS (Pin, Qin): PUMP PERFORMANCE
        alpha1_pump = coeff['alpha1_pump']
        alpha2_pump = coeff['alpha2_pump']
        bp_x_pump = coeff['bp_x_pump']

        s_indicators_inputs = range(0, len(bp_x_pump))

        # note: input power = input[t,car]. input volume = b_tec.var_total_inflow[t] (no car, because water)

        def init_power_inflow(dis, t, ind):
            if ind == 0:  # technology off
                def init_inflow_zero(const):
                    return b_tec.var_total_inflow[t] == 0

                dis.const_inflow_zero = Constraint(rule=init_inflow_zero)

                def init_power_in_zero(const, car_input):
                    return self.input[t, car_input] == 0

                dis.const_power_in_zero = Constraint(b_tec.set_input_carriers, rule=init_power_in_zero)

            else:
                def init_inflow_above1(const):  # making sure value is higher than first breakpoint of segment
                    return b_tec.var_total_inflow[t] >= bp_x_pump[ind - 1] * 3600 * b_tec.var_size

                dis.const_inflow_above1 = Constraint(rule=init_inflow_above1)

                def init_inflow_below2(const):  # making sure value is lower than second breakpoint of segment
                    return b_tec.var_input_volume[t] <= bp_x_pump[ind] * 3600 * b_tec.var_size

                dis.const_inflow_below2 = Constraint(rule=init_inflow_below2)

                def init_power_in_on(const, car_input):
                    return (self.input[t, car_input] == alpha1_pump[ind - 1] * b_tec.var_input_volume[t] / 3600 +
                            alpha2_pump[ind - 1])

                dis.const_power_in_on = Constraint(b_tec.set_input_carriers, rule=init_power_in_on)

        b_tec.dis_power_inflow = Disjunct(set_t_full, s_indicators_inputs, rule=init_power_inflow)

        def bind_disjunctions_inputs(dis, t):
            return [b_tec.dis_power_inflow[t, i] for i in s_indicators_inputs]

        b_tec.disjunction_power_inflow = Disjunction(set_t_full, rule=bind_disjunctions_inputs)

        #TODO check (update discharging / turbine disjunction to new code style)

        # define disjuncts for discharging: either not or within bounds
        s_indicators_discharge = range(0, 2)

        def init_discharging(dis, t, ind):
            if ind == 0:  # not discharging
                def init_output_to_zero(const):
                    return b_tec.var_total_outflow[t] == 0

                dis.const_output_to_zero = Constraint(rule=init_output_to_zero)

            elif ind == 1:  # discharging
                def init_output_low(const):
                    return discharge_min * b_tec.var_size <= b_tec.var_total_outflow[t]

                dis.const_output_low = Constraint(rule=init_output_low)

                def init_output_up(const):
                    return b_tec.var_total_outflow[t] <= discharge_max * b_tec.var_size

                dis.const_output_up = Constraint(rule=init_output_up)

        b_tec.dis_discharging = Disjunct(set_t_full, s_indicators_discharge, rule=init_discharging)

        # Bind disjuncts
        def bind_disjunctions_discharge(dis, t):
            return [b_tec.dis_discharging[t, i] for i in s_indicators_discharge]

        b_tec.disjunction_discharging = Disjunction(set_t_full, rule=bind_disjunctions_discharge)

        #TODO add output/outflow relations = turbine performance disjunctions + update to new code style

        # OUTPUT FORMULATIONS (Pout, Qout): TURBINE PERFORMANCE

        alpha1_turbine = coeff['alpha1_turbine']
        alpha2_turbine = coeff['alpha2_turbine']
        bp_x_turbine = coeff['bp_x_turbine']

        s_indicators_outputs = range(0, len(bp_x_turbine))

        def init_power_outflow(dis, t, ind):
            if ind == 0:  # technology off
                def init_outflow_zero(const):
                    return b_tec.var_total_outflow[t] == 0

                dis.const_outflow_zero = Constraint(rule=init_outflow_zero)

                def init_power_out_zero(const, car_output):
                    return self.output[t, car_output] == 0

                dis.const_power_out_zero = Constraint(b_tec.set_output_carriers, rule=init_power_out_zero)

            else:
                def init_outflow_above1(const):  # making sure value is higher than first breakpoint of segment
                    return b_tec.var_total_outflow[t] >= bp_x_turbine[ind - 1] * 3600 * b_tec.var_size

                dis.const_outflow_above1 = Constraint(rule=init_outflow_above1)

                def init_outflow_below2(const):  # making sure value is lower than second breakpoint of segment
                    return b_tec.var_total_outflow[t] <= bp_x_turbine[ind] * 3600 * b_tec.var_size

                dis.const_outflow_below2 = Constraint(rule=init_outflow_below2)

                def init_power_out_on(const, car_output):
                    return (self.output[t, car_output] == alpha1_turbine[ind - 1] * b_tec.var_total_outflow[t] / 3600 +
                            alpha2_turbine[ind - 1])

                dis.const_power_out_on = Constraint(b_tec.set_output_carriers, rule=init_power_out_on)

        b_tec.dis_power_outflow = Disjunct(set_t_full, s_indicators_outputs, rule=init_power_outflow)

        def bind_disjunctions_outputs(dis, t):
            return [b_tec.dis_power_outflow[t, i] for i in s_indicators_outputs]

        b_tec.disjunction_power_outflow = Disjunction(set_t_full, rule=bind_disjunctions_outputs)

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
