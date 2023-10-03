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

        # Abdditional parameters
        eta_in = coeff['eta_in']
        eta_out = coeff['eta_out']
        eta_lambda = coeff['lambda']
        charge_max = coeff['charge_max']
        discharge_max = coeff['discharge_max']
        min_fill = coeff['min_fill']
        pump_slots = coeff['pump_slots']
        turbine_slots = coeff['turbine_slots']
        capex_pumps = {}
        capex_pumps[str(1)] = 5
        capex_pumps[str(2)] = 7


        nr_timesteps_averaged = energyhub.model_information.averaged_data_specs.nr_timesteps_averaged

        # Additional sets
        b_tec.set_pump_slots = RangeSet(pump_slots)
        b_tec.set_turbine_slots = RangeSet(turbine_slots)

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

        def pumps_block_init(b_pump):
            """

            """
            pump_types = range(0, 3)
            b_pump.set_pump_types = RangeSet(max(pump_types))

            # Parameters
            b_pump.para_size_min = Param(initialize=0)
            b_pump.para_size_max = Param(initialize=10)
            def init_para_capex(para, pt):
                return capex_pumps[str(pt)]
            b_pump.para_capex = Param(b_pump.set_pump_types, rule=init_para_capex)

            # Decision Variables
            b_pump.var_size = Var(domain=NonNegativeReals,
                                  bounds=(b_pump.para_size_min, b_pump.para_size_max))
            b_pump.var_input = Var(set_t_full, domain=NonNegativeReals,
                                   bounds=(b_pump.para_size_min, b_pump.para_size_max))
            b_pump.var_inflow = Var(set_t_full, domain=NonNegativeReals,
                                    bounds=(b_pump.para_size_min, b_pump.para_size_max))
            b_pump.var_pump_type = Var(domain=NonNegativeReals,
                                       bounds=(min(pump_types), max(pump_types)))

            # THIS NEEDS TO BE FIXED
            b_pump.var_capex = Var(domain=NonNegativeReals, bounds=(0, b_pump.para_size_max * 100))


            def init_pump_types(dis, type):
                if type == 0:  # slot not used

                    dis.const_pump_type = Constraint(expr= b_pump.var_pump_type == 0)

                    def init_inflow(const, t):
                        return b_pump.var_input[t] == 0

                    dis.const_inflow = Constraint(self.set_t, rule=init_inflow)

                    def init_pump_efficiency(const, t):
                        return b_pump.var_inflow[t] == 0

                    dis.const_pump_efficiency = Constraint(self.set_t, rule=init_pump_efficiency)

                    dis.const_size = Constraint(expr=b_pump.var_size == 0)
                    dis.const_capex = Constraint(expr=b_pump.var_capex == 0)

                elif type == 1:  # type 1

                    dis.const_pump_type = Constraint(expr= b_pump.var_pump_type == 1)

                    def init_inflow(const, t):
                        return b_pump.var_inflow[t] <= b_pump.var_size

                    dis.const_inflow = Constraint(self.set_t, rule=init_inflow)

                    def init_pump_efficiency(const, t, car):
                        return b_pump.var_inflow[t] / 0.99 == b_pump.var_input[t]


                    dis.const_pump_efficiency = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_pump_efficiency)

                    dis.const_size = Constraint(expr= 1 <= b_pump.var_size)
                    dis.const_capex = Constraint(expr=b_pump.var_size * b_pump.para_capex[type] == b_pump.var_capex)


                elif type == 2:  # type 2

                    dis.const_pump_type = Constraint(expr= b_pump.var_pump_type == 2)

                    def init_inflow(const, t):
                        return b_pump.var_inflow[t] <= b_pump.var_size

                    dis.const_inflow = Constraint(self.set_t, rule=init_inflow)

                    def init_pump_efficiency(const, t, car):
                        return b_pump.var_inflow[t] / 0.7 == b_pump.var_input[t]

                    dis.const_pump_efficiency = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_pump_efficiency)

                    dis.const_size = Constraint(expr= 1 <= b_pump.var_size)

                    dis.const_capex = Constraint(expr=b_pump.var_size * b_pump.para_capex[type] == b_pump.var_capex)

            b_pump.dis_pump_types = Disjunct(pump_types, rule=init_pump_types)

            # Bind disjuncts
            def bind_disjunctions(dis):
                return [b_pump.dis_pump_types[i] for i in pump_types]

            b_pump.disjunction_pump_types = Disjunction(rule=bind_disjunctions)

            b_pump = perform_disjunct_relaxation(b_pump)

            return b_pump

        b_tec.pump_block = Block(b_tec.set_pump_slots, rule=pumps_block_init)

        def turbines_block_init(b_turbine):
            """

            """
            # Parameters
            b_turbine.para_size_min = Param(initialize=0)
            b_turbine.para_size_max = Param(initialize=10)
            b_turbine.para_capex = Param(initialize=2)

            # Decision Variables
            b_turbine.var_size = Var(domain=NonNegativeReals,
                                     bounds=(b_turbine.para_size_min, b_turbine.para_size_max))
            b_turbine.var_output = Var(self.set_t, domain=NonNegativeReals,
                                       bounds=(b_turbine.para_size_min, b_turbine.para_size_max))
            b_turbine.var_outflow = Var(self.set_t, domain=NonNegativeReals,
                                        bounds=(b_turbine.para_size_min, b_turbine.para_size_max))
            b_turbine.var_capex = Var(domain=NonNegativeReals)

            def init_inflow(const, t):
                return b_turbine.var_outflow[t] <= b_turbine.var_size

            b_turbine.const_inflow = Constraint(self.set_t, rule=init_inflow)

            def init_turbine_efficiency(const, t):
                return 0.8 * b_turbine.var_outflow[t] == b_turbine.var_output[t]

            b_turbine.const_turbine_efficiency = Constraint(self.set_t, rule=init_turbine_efficiency)

            b_turbine.const_capex = Constraint(expr=b_turbine.var_size * b_turbine.para_capex == b_turbine.var_capex)

            #     mc.perform_disjunct_relaxation(b_turbine)
            return b_turbine

        b_tec.turbine_block = Block(b_tec.set_turbine_slots, rule=turbines_block_init)

        # Aggregate Input/Output
        def init_total_input(const, t, car):
            return b_tec.var_input[t, car] == \
                   sum(b_tec.pump_block[pump].var_input[t] for pump in b_tec.set_pump_slots)

        b_tec.const_total_input = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_total_input)

        def init_total_output(const, t, car):
            return b_tec.var_output[t, car] == \
                   sum(b_tec.turbine_block[turbine].var_output[t] for turbine in b_tec.set_turbine_slots)

        b_tec.const_total_output = Constraint(self.set_t, b_tec.set_input_carriers, rule=init_total_output)

        def init_total_inflow(const, t):
            return b_tec.var_total_inflow[t] == \
                   sum(b_tec.pump_block[pump].var_inflow[t] for pump in b_tec.set_pump_slots)

        b_tec.const_total_inflow = Constraint(self.set_t, rule=init_total_inflow)

        def init_total_outflow(const, t):
            return b_tec.var_total_outflow[t] == \
                   sum(b_tec.turbine_block[turbine].var_outflow[t] for turbine in b_tec.set_turbine_slots)

        b_tec.const_total_outflow = Constraint(self.set_t, rule=init_total_outflow)

        # CAPEX Calculation
        b_tec.const_capex_aux = Constraint(expr=10 * b_tec.var_size +
                                                sum(b_tec.turbine_block[turbine].var_capex for turbine in
                                                    b_tec.set_turbine_slots) +
                                                sum(b_tec.pump_block[pump].var_capex for pump in
                                                    b_tec.set_pump_slots) ==
                                                b_tec.var_capex_aux)

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
        for pump in b_tec.set_pump_slots:
            self.results['time_dependent']['var_inflow' + str(pump)] = [b_tec.pump_block[pump].var_inflow[t].value for t in self.set_t]
            self.results['time_dependent']['var_input' + str(pump)] = [b_tec.pump_block[pump].var_input[t].value for t in self.set_t]

        for turb in b_tec.set_pump_slots:
            self.results['time_dependent']['var_outflow' + str(turb)] = [b_tec.turbine_block[turb].var_outflow[t].value for t in self.set_t]
            self.results['time_dependent']['var_output' + str(turb)] = [b_tec.turbine_block[turb].var_output[t].value for t in self.set_t]

        design = {}
        design['reservoir_size'] = b_tec.var_size.value
        for pump in b_tec.set_pump_slots:
            design['pump_' + str(pump) + '_type'] = b_tec.pump_block[pump].var_pump_type.value
            design['pump_' + str(pump) + '_size'] = b_tec.pump_block[pump].var_size.value
            design['pump_' + str(pump) + '_capex'] = b_tec.pump_block[pump].var_capex.value

        for turb in b_tec.set_pump_slots:
            # design['turbine_' + str(pump) + '_type'] = b_tec.turbine_block[turb].var_pump_type.value
            design['turbine_' + str(turb) + '_size'] = b_tec.turbine_block[turb].var_size.value
            design['turbine_' + str(turb) + '_capex'] = b_tec.turbine_block[turb].var_capex.value

        design_df = pd.DataFrame(data=design, index=[0]).T

        self.results['specific_design'] = design_df

        return self.results
