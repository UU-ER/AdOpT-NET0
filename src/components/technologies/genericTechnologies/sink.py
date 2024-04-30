from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import numpy as np

from ..utilities import FittedPerformance
from ..technology import Technology


class Sink(Technology):
    """
    This model resembles a permanent storage technology (sink). It takes energy and a main carrier (e.g. CO2, H2 etc)
    as inputs, and it has no output.


    **Variable declarations:**

    - Storage level in :math:`t`: :math:`E_t`

    **Constraint declarations:**

    - Maximal injection rate:

      .. math::
        Input_{t} \leq Inj_rate

    - Size constraint:

      .. math::
        E_{t} \leq Size

    - Storage level calculation:

      .. math::
        E_{t} = E_{t-1} + Input_{t}

    - If an energy consumption for the injection is given, the respective carrier input is:

      .. math::
        Input_{t, car} = cons_{car, in} Input_{t}

    """

    def __init__(self, tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()

    def fit_technology_performance(self, node_data):
        """
        Fits conversion technology type SINK and returns fitted parameters as a dict

        :param node_data: contains data on demand, climate data, etc.
        """

        climate_data = node_data.data['climate_data']

        time_steps = len(climate_data)

        # Main carrier (carrier to be stored)
        self.main_car = self.performance_data['main_input_carrier']

        # Input Bounds
        for car in self.performance_data['input_carrier']:
            if car == self.performance_data['main_input_carrier']:
                self.fitted_performance.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                            np.ones(shape=(time_steps))))
            else:
                if 'energy_consumption' in self.performance_data['performance']:
                    energy_consumption = self.performance_data['performance']['energy_consumption']
                    self.fitted_performance.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                                                np.ones(shape=(time_steps)) *
                                                                                energy_consumption['in'][car]))

        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 0


    def construct_tech_model(self, b_tec, energyhub):
        """
        Adds constraints to technology blocks for tec_type SINK, resembling a permanent storage technology

        :param b_tec:
        :param energyhub:
        :return: b_tec
        """

        super(Sink, self).construct_tech_model(b_tec, energyhub)

        set_t_full = energyhub.model.set_t_full

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        coeff = self.fitted_performance.coefficients

        nr_timesteps_averaged = energyhub.model_information.averaged_data_specs.nr_timesteps_averaged

        # Additional decision variables
        b_tec.var_storage_level = Var(set_t_full,
                                      domain=NonNegativeReals,
                                      bounds=(b_tec.para_size_min, b_tec.para_size_max))


        # Size constraint
        def init_size_constraint(const, t):
            return b_tec.var_storage_level[t] <= b_tec.var_size
        b_tec.const_size = Constraint(set_t_full, rule=init_size_constraint)

        # Constraint storage level
        if energyhub.model_information.clustered_data and not self.modelled_with_full_res:
            def init_storage_level(const, t):
                if t == 1:
                    return b_tec.var_storage_level[t] == self.input[self.sequence[t - 1], self.main_car]
                else:
                    return b_tec.var_storage_level[t] == \
                           b_tec.var_storage_level[t - 1] + \
                           self.input[self.sequence[t - 1], self.main_car]
        else:
            def init_storage_level(const, t):
                if t == 1:
                    return b_tec.var_storage_level[t] == self.input[t, self.main_car]
                else:
                    return b_tec.var_storage_level[t] == \
                           b_tec.var_storage_level[t - 1]  + \
                           self.input[t, self.main_car]

            b_tec.const_storage_level = Constraint(set_t_full, rule=init_storage_level)

        # Maximal injection rate
        def init_maximal_injection(const, t):
            return self.input[t, self.main_car] <= self.performance_data['injection_rate_max']
        b_tec.const_max_charge = Constraint(self.set_t, rule=init_maximal_injection)

        # Energy consumption for injection
        if 'energy_consumption' in self.performance_data['performance']:
            energy_consumption = self.performance_data['performance']['energy_consumption']
            if 'in' in energy_consumption:
                b_tec.set_energyconsumption_carriers_in = Set(initialize=energy_consumption['in'].keys())

                def init_energyconsumption_in(const, t, car):
                    return self.input[t, car] == self.input[t, self.main_car] * energy_consumption['in'][car]
                b_tec.const_energyconsumption_in = Constraint(self.set_t, b_tec.set_energyconsumption_carriers_in,
                                                       rule=init_energyconsumption_in)


        # RAMPING RATES
        if "ramping_rate" in self.performance_data:
            if not self.performance_data['ramping_rate']   == -1:
                b_tec = self._define_ramping_rates(b_tec)

        return b_tec

    def write_tec_operation_results_to_group(self, h5_group, model_block):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        super(Sink, self).write_tec_operation_results_to_group(h5_group, model_block)

        h5_group.create_dataset("storage_level", data=[model_block.var_storage_level[t].value for t in self.set_t_full])

    def _define_ramping_rates(self, b_tec):
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
