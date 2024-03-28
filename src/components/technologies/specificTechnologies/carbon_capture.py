from pyomo.environ import *
from pyomo.gdp import *
import copy
from warnings import warn
import numpy as np
from src.components.technologies.genericTechnologies.utilities import fit_performance_generic_tecs
from src.components.technologies.technology import Technology
from src.components.technologies.utilities import FittedPerformance, fit_linear_function, fit_piecewise_function, sig_figs



class CarbonCapture(Technology):
    """
    Generic carbon capture (CC) object that takes the mass flow rate of the CO2 in a fluegas as input and returns
    CO2 captured as output according to

    :math:`CO2fluegas * capture\_rate = CO2captured`

    For every unit of 'CO2fluegas' (main carrier), the heat and electricity demand are calculated with fixed input ratios.
    To account for the CO2 captured in the emission balance, the 'CO2captured' needs to be either exported
    with an emission factor at the export of -1, or sent to a CO2 sink object (which has -1 as emission factor). To add
    the option of using CC on a technology (i.e. gas turbine), the technology needs to be modified so that it has also
    the carrier 'CO2fluegas' as output.
    The fit performance fit depends on the type of capture technology, which needs to be specified in the json file.

    So far, only post-combustion MEA has been modelled (based on Weimann et Al. (2023),
    A thermodynamic-based mixed-integer linear model of post-combustion carbon capture
    for reliable use in energy system optimisation https://doi.org/10.1016/j.apenergy.2023.120738).
    The range of sizes of the CC object needs to be specified in the json file by selecting

    The constraints implemented are the following


    """

    def __init__(self,
                 tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance()
        self.main_car = self.performance_data['main_input_carrier']
        self.capture_type = self.performance_data['capture_type']

    def fit_technology_performance(self, node_data):
        """
        Performing fitting for the technology type CarbonCapture and for the specific type of CC.
        The input ratios correspond to the energy consumption per unit of CO2 entering the CC (main carrier).
        In the case of MEA, eta and omega are the equivalent of alpha and beta in the paper of Weimann et Al. 2023.
        Size_min and size_max are given in tCO2/h. In addition, the capex (fixed and unitary)
        are adjusted for the CO2 concentration and capture rate.


        :param tec_data: technology data
        :param climate_data: climate data

        """

        climate_data = node_data.data['climate_data']
        time_steps = len(climate_data)

        if self.capture_type == 'MEA':

            self.input_ratios = {}
            CO2_concentration = self.performance_data['CO2_concentration']
            molar_mass_CO2 = 44.01
            carbon_capture_rate = self.performance_data['capture_rate']
            convert2t_per_h = molar_mass_CO2 * CO2_concentration * 3.6 # convert kmol/s of fluegas to ton/h of CO2
            if self.performance_data['plant_size'] == 'small':

                size_min = 14.18 * CO2_concentration
                size_max = 158.436 * CO2_concentration
                self.size_min = np.max([size_min, self.size_min])
                self.size_max = np.min([size_max, self.size_max])

                eta = {
                    'electricity': 0.937,
                    'heat': -0.6068
                }
                omega = {
                    'electricity': 0.2719,
                    'heat': 158.71
                }

                self.economics.capex_data['unit_capex'] = 3.44/convert2t_per_h + 185 * carbon_capture_rate * CO2_concentration / convert2t_per_h
                self.economics.capex_data['fix_capex'] = 2.17

            elif self.performance_data['plant_size'] == 'medium':

                size_min = 158.436 * CO2_concentration
                size_max = 792.18 * CO2_concentration
                self.size_min = np.max([size_min, self.size_min])
                self.size_max = np.min([size_max, self.size_max])

                eta = {
                    'electricity': 0.0945,
                    'heat': -1.240
                }
                omega = {
                    'electricity': 0.2787,
                    'heat': 175.32
                }

                self.economics.capex_data['unit_capex'] = 2.83/convert2t_per_h + 125 * carbon_capture_rate * CO2_concentration / convert2t_per_h
                self.economics.capex_data['fix_capex'] = 11.1

            elif self.performance_data['plant_size'] == 'large':

                size_min = 792.18 * CO2_concentration
                size_max = 1985.203 * CO2_concentration
                self.size_min = np.max([size_min, self.size_min])
                self.size_max = np.min([size_max, self.size_max])

                eta = {
                    'electricity': 0.0958,
                    'heat': 0.2684
                }
                omega = {
                    'electricity': 0.2885,
                    'heat': 150.22
                }

                self.economics.capex_data['unit_capex'] = 3.11/convert2t_per_h + 123 * carbon_capture_rate * CO2_concentration / convert2t_per_h
                self.economics.capex_data['fix_capex'] = 10.8

            for car in self.performance_data['input_carrier']:
                if not car == self.performance_data['main_input_carrier']:
                    self.input_ratios[car] = (eta[car] + omega[car] * CO2_concentration) / (CO2_concentration * molar_mass_CO2 * 3.6)

        # Input bounds
        self.fitted_performance.bounds['input'][self.performance_data['main_input_carrier']] = np.column_stack((np.zeros(shape=(time_steps)),
                                                             np.ones(shape=(time_steps))))
        for car in self.performance_data['input_carrier']:
            if not car == self.performance_data['main_input_carrier']:
                self.fitted_performance.bounds['input'][car] = self.fitted_performance.bounds['input'][
                                                            self.performance_data['main_input_carrier']] \
                                                            * self.input_ratios[car]

        # Output bounds
        for car in self.performance_data['output_carrier']:
            # TODO add possibility of having another output carrier (e.g. heat)

            if 'CO2' in car:
                self.fitted_performance.bounds['output'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                                    np.ones(shape=(time_steps)) *
                                                                    (self.performance_data['capture_rate'])))

    def construct_tech_model(self, b_tec, energyhub):
        """
        Adds constraints to technology blocks for tec_type CARBONCAPTURE

        :param obj b_tec: technology block
        :param Energyhub energyhub: energyhub instance
        :return: technology block
        """
        super(CarbonCapture, self).construct_tech_model(b_tec, energyhub)

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        rated_power = self.fitted_performance.rated_power
        standby_power = self.performance_data['standby_power']
        phi = self.input_ratios

        # Technology Constraints
        # performance parameter:
        carbon_capture_rate = self.performance_data['capture_rate']
        rated_power = self.fitted_performance.rated_power
        min_part_load = self.performance_data['min_part_load']

        # Input-output correlation
        def init_input_output(const, t, car_output):
            return self.output[t, car_output] == \
                   carbon_capture_rate * self.input[t, self.main_car]

        b_tec.const_input_output = Constraint(self.set_t, b_tec.set_output_carriers,
                                              rule=init_input_output)

        if min_part_load > 0:
            def init_min_part_load(const, t):
                return min_part_load * b_tec.var_size * rated_power <= self.input[t, self.main_car]

            b_tec.const_min_part_load = Constraint(self.set_t, rule=init_min_part_load)

        # Size constraints
        # constraint on input ratios
        if standby_power == -1:
            def init_input_input(const, t, car_input):
                if car_input == self.main_car:
                    return Constraint.Skip
                else:
                    return self.input[t, car_input] == phi[car_input] * self.input[t, self.main_car]

            b_tec.const_input_input = Constraint(self.set_t_full, b_tec.set_input_carriers, rule=init_input_input)
        else:
            s_indicators = range(0, 2)

            def init_input_input(dis, t, ind):
                if ind == 0:  # technology off
                    dis.const_x_off = Constraint(expr=b_tec.var_x[t] == 0)

                    def init_input_off(const, car_input):
                        if car_input == self.main_car:
                            return Constraint.Skip
                        else:
                            return self.input[t, car_input] == 0

                    dis.const_input_off = Constraint(b_tec.set_input_carriers, rule=init_input_off)

                else:  # technology on
                    dis.const_x_on = Constraint(expr=b_tec.var_x[t] == 1)

                    def init_input_on(const, car_input):
                        if car_input == self.main_car:
                            return Constraint.Skip
                        else:
                            return self.input[t, car_input] == phi[car_input] * self.input[t, self.main_car]

                    dis.const_input_on = Constraint(b_tec.set_input_carriers, rule=init_input_on)

            b_tec.dis_input_input = Disjunct(self.set_t, s_indicators, rule=init_input_input)

            # Bind disjuncts
            def bind_disjunctions(dis, t):
                return [b_tec.dis_input_input[t, i] for i in s_indicators]

            b_tec.disjunction_input_input = Disjunction(self.set_t, rule=bind_disjunctions)

        # size constraint based main carrier input
        def init_size_constraint(const, t):
            return self.input[t, self.main_car] <= b_tec.var_size * rated_power

        b_tec.const_size = Constraint(self.set_t, rule=init_size_constraint)

        # RAMPING RATES
        if "ramping_rate" in self.performance_data:
            if not self.performance_data['ramping_rate'] == -1:
                b_tec = self.__define_ramping_rates(b_tec)

        return b_tec

    def __define_ramping_rates(self, b_tec):
        """
        Constraints the inputs for a ramping rate

        :param b_tec: technology model block
        :return:
        """
        ramping_rate = self.performance_data['ramping_rate']

        def init_ramping_down_rate(const, t):
            if t > 1:
                return -ramping_rate <= sum(self.input[t, car_input] - self.input[t-1, car_input]
                                                for car_input in b_tec.set_input_carriers)
            else:
                return Constraint.Skip
        b_tec.const_ramping_down_rate = Constraint(self.set_t, rule=init_ramping_down_rate)

        def init_ramping_up_rate(const, t):
            if t > 1:
                return sum(self.input[t, car_input] - self.input[t-1, car_input]
                               for car_input in b_tec.set_input_carriers) <= ramping_rate
            else:
                return Constraint.Skip
        b_tec.const_ramping_up_rate = Constraint(self.set_t, rule=init_ramping_up_rate)

        return b_tec
