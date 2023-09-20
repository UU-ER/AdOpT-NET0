from src.model_construction.technology_constraints import *
import src.global_variables as global_variables
from src.data_management.components.utilities import Economics
from src.data_management.components.fit_technology_performance import *

# Technology Class
    # Attributes (Data)
    # construct general tec model
    # reporting

# One subclass per tec type
    # fit_technology performance
    # construct specific technology constraints
    # reporting


class Technology:
    """
    Class to read and manage data for technologies
    """
    def __init__(self, tec_data):
        """
        Initializes technology class from technology name

        The technology name needs to correspond to the name of a JSON file in ./data/technology_data.

        :param str technology: name of technology to read data
        """
        # General information
        self.name = tec_data['name']
        self.existing = 0
        self.technology_model = tec_data['tec_type']
        self.size_initial = []
        self.size_is_int = tec_data['size_is_int']
        self.size_min = tec_data['size_min']
        self.size_max = tec_data['size_max']
        self.decommission = tec_data['decommission']
        self.modelled_with_full_res = []

        # Economics
        self.economics = Economics(tec_data['Economics'])

        # Technology Performance
        self.performance_data = tec_data['TechnologyPerf']

        # Size-input/output constraints
        if self.technology_model == 'CONV1':
            self.performance_data['size_based_on'] = tec_data['size_based_on']
        else:
            self.performance_data['size_based_on'] = 'input'

        # Emissions are based on...
        if (self.technology_model == 'DAC_Adsorption') or \
                (self.technology_model == 'CONV4'):
            self.emissions_based_on = 'output'
        else:
            self.emissions_based_on = 'input'

        self.fitted_performance = None


    def construct_model(self, b_tec, energyhub):

        print('\t - Adding Technology ' + self.name)

        # TECHNOLOGY DATA
        configuration = energyhub.configuration

        # MODELING TYPICAL DAYS
        if global_variables.clustered_data:
            if configuration.optimization.typicaldays.method == 2:
                technologies_modelled_with_full_res = ['RES', 'STOR', 'Hydro_Open']
                if self.technology_model in technologies_modelled_with_full_res:
                    self.modelled_with_full_res = 1
                else:
                    self.modelled_with_full_res = 0
            else:
                raise KeyError('The clustering method specified in the configuration file does not exist.')
        else:
            self.modelled_with_full_res = 1

        # SIZE
        b_tec = self.__define_size(b_tec)

        # CAPEX
        b_tec = self.__define_capex(b_tec, energyhub)

        # INPUT AND OUTPUT
        b_tec = self.__define_input(b_tec, energyhub)
        b_tec = self.__define_output(b_tec, energyhub)

        # OPEX
        b_tec = self.__define_opex(b_tec, energyhub)

        # EMISSIONS
        b_tec = self.__define_emissions(b_tec, energyhub)

        # DEFINE AUXILIARY VARIABLES FOR CLUSTERED DATA
        if global_variables.clustered_data and not self.modelled_with_full_res:
            b_tec = self.__define_auxiliary_vars(b_tec, energyhub)

        # GENERIC TECHNOLOGY CONSTRAINTS
        b_tec = self.__constraints_tec_RES(b_tec, energyhub)

        return b_tec

    def __define_size(self, b_tec):
        """
        Defines variables and parameters related to technology size.

        Parameters defined:
        - size min
        - size max
        - size initial (for existing technologies)

        Variables defined:
        - size
        """
        size_is_int = self.size_is_int
        size_min = self.size_min
        existing = self.existing
        decommission = self.decommission
        if existing:
            size_initial = self.size_initial
            size_max = size_initial
        else:
            size_max = self.size_max

        if size_is_int:
            size_domain = NonNegativeIntegers
        else:
            size_domain = NonNegativeReals

        b_tec.para_size_min = Param(domain=NonNegativeReals, initialize=size_min, mutable=True)
        b_tec.para_size_max = Param(domain=NonNegativeReals, initialize=size_max, mutable=True)
        if existing:
            b_tec.para_size_initial = Param(within=size_domain, initialize=size_initial)
        if existing and not decommission:
            # Decommissioning is not possible, size fixed
            b_tec.var_size = Param(within=size_domain, initialize=b_tec.para_size_initial)
        else:
            # Decommissioning is possible, size variable
            b_tec.var_size = Var(within=size_domain, bounds=(b_tec.para_size_min, b_tec.para_size_max))

        return b_tec

    def __define_capex(self, b_tec, energyhub):
        """
        Defines variables and parameters related to technology capex.

        Parameters defined:
        - unit capex/ breakpoints for capex function

        Variables defined:
        - capex_aux (theoretical CAPEX for existing technologies)
        - CAPEX (actual CAPEX)
        - Decommissioning Costs (for existing technologies)
        """
        configuration = energyhub.configuration

        size_is_int = self.size_is_int
        existing = self.existing
        decommission = self.decommission
        economics = self.economics
        discount_rate = mc.set_discount_rate(configuration, economics)
        capex_model = mc.set_capex_model(configuration, economics)

        # CAPEX auxiliary (used to calculate theoretical CAPEX)
        # For new technologies, this is equal to actual CAPEX
        # For existing technologies it is used to calculate fixed OPEX
        b_tec.var_capex_aux = Var()
        annualization_factor = mc.annualize(discount_rate, economics.lifetime)
        if capex_model == 1:
            b_tec.para_unit_capex = Param(domain=Reals, initialize=economics.capex_data['unit_capex'], mutable=True)
            b_tec.para_unit_capex_annual = Param(domain=Reals,
                                                 initialize=annualization_factor * economics.capex_data['unit_capex'],
                                                 mutable=True)
            b_tec.const_capex_aux = Constraint(
                expr=b_tec.var_size * b_tec.para_unit_capex_annual == b_tec.var_capex_aux)
        elif capex_model == 2:
            b_tec.para_bp_x = Param(domain=Reals, initialize=economics.capex_data['piecewise_capex']['bp_x'])
            b_tec.para_bp_y = Param(domain=Reals, initialize=economics.capex_data['piecewise_capex']['bp_y'])
            b_tec.para_bp_y_annual = Param(domain=Reals, initialize=annualization_factor *
                                                                    economics.capex_data['piecewise_capex']['bp_y'])
            global_variables.big_m_transformation_required = 1
            b_tec.const_capex_aux = Piecewise(b_tec.var_capex_aux, b_tec.var_size,
                                              pw_pts=b_tec.para_bp_x,
                                              pw_constr_type='EQ',
                                              f_rule=b_tec.para_bp_y_annual,
                                              pw_repn='SOS2')
        # CAPEX
        if existing and not decommission:
            b_tec.var_capex = Param(domain=Reals, initialize=0)
        else:
            b_tec.var_capex = Var()
            if existing:
                b_tec.para_decommissioning_cost = Param(domain=Reals, initialize=economics.decommission_cost,
                                                        mutable=True)
                b_tec.const_capex = Constraint(
                    expr=b_tec.var_capex == (
                                b_tec.para_size_initial - b_tec.var_size) * b_tec.para_decommissioning_cost)
            else:
                b_tec.const_capex = Constraint(expr=b_tec.var_capex == b_tec.var_capex_aux)
        return b_tec


    def __define_input(self, b_tec, energyhub):
        """
        Defines input to a technology

        var_input is always in full resolution
        var_input_aux can be in reduced resolution
        """
        # Technology related options
        existing = self.existing
        performance_data = self.performance_data
        fitted_performance = self.fitted_performance
        technology_model = self.technology_model
        modelled_with_full_res = self.modelled_with_full_res

        # set_t and sequence
        set_t = energyhub.model.set_t_full
        if global_variables.clustered_data and not modelled_with_full_res:
            sequence = energyhub.data.k_means_specs.full_resolution['sequence']

        if existing:
            size_initial = self.size_initial
            size_max = size_initial
        else:
            size_max = self.size_max

        rated_power = fitted_performance.rated_power

        if (technology_model == 'RES') or (technology_model == 'CONV4'):
            b_tec.set_input_carriers = Set(initialize=[])
        else:
            b_tec.set_input_carriers = Set(initialize=performance_data['input_carrier'])

            def init_input_bounds(bounds, t, car):
                if global_variables.clustered_data and not modelled_with_full_res:
                    return tuple(
                        fitted_performance.bounds['input'][car][sequence[t - 1] - 1, :] * size_max * rated_power)
                else:
                    return tuple(fitted_performance.bounds['input'][car][t - 1, :] * size_max * rated_power)

            b_tec.var_input = Var(set_t, b_tec.set_input_carriers, within=NonNegativeReals,
                                  bounds=init_input_bounds)
        return b_tec

    def __define_output(self, b_tec, energyhub):
        """
        Defines output to a technology

        var_output is always in full resolution
        """
        # Technology related options
        existing = self.existing
        performance_data = self.performance_data
        fitted_performance = self.fitted_performance
        modelled_with_full_res = self.modelled_with_full_res

        rated_power = fitted_performance.rated_power

        # set_t
        set_t = energyhub.model.set_t_full
        if global_variables.clustered_data and not modelled_with_full_res:
            sequence = energyhub.data.k_means_specs.full_resolution['sequence']

        if existing:
            size_initial = self.size_initial
            size_max = size_initial
        else:
            size_max = self.size_max

        b_tec.set_output_carriers = Set(initialize=performance_data['output_carrier'])

        def init_output_bounds(bounds, t, car):
            if global_variables.clustered_data and not modelled_with_full_res:
                return tuple(fitted_performance.bounds['output'][car][sequence[t - 1] - 1, :] * size_max * rated_power)
            else:
                return tuple(fitted_performance.bounds['output'][car][t - 1, :] * size_max * rated_power)

        b_tec.var_output = Var(set_t, b_tec.set_output_carriers, within=NonNegativeReals,
                               bounds=init_output_bounds)
        return b_tec

    def __define_opex(self, b_tec, energyhub):
        """
        Defines variable and fixed OPEX
        """
        economics = self.economics
        set_t = energyhub.model.set_t_full

        # VARIABLE OPEX
        b_tec.para_opex_variable = Param(domain=Reals, initialize=economics.opex_variable, mutable=True)
        b_tec.var_opex_variable = Var(set_t)

        def init_opex_variable(const, t):
            return sum(b_tec.var_output[t, car] for car in b_tec.set_output_carriers) * b_tec.para_opex_variable == \
                   b_tec.var_opex_variable[t]

        b_tec.const_opex_variable = Constraint(set_t, rule=init_opex_variable)

        # FIXED OPEX
        b_tec.para_opex_fixed = Param(domain=Reals, initialize=economics.opex_fixed, mutable=True)
        b_tec.var_opex_fixed = Var()
        b_tec.const_opex_fixed = Constraint(expr=b_tec.var_capex_aux * b_tec.para_opex_fixed == b_tec.var_opex_fixed)
        return b_tec

    def __define_emissions(self, b_tec, energyhub):
        """
        Defines Emissions
        """

        set_t = energyhub.model.set_t_full
        performance_data = self.performance_data
        technology_model = self.technology_model
        emissions_based_on = self.emissions_based_on

        b_tec.para_tec_emissionfactor = Param(domain=Reals, initialize=performance_data['emission_factor'])
        b_tec.var_tec_emissions_pos = Var(set_t, within=NonNegativeReals)
        b_tec.var_tec_emissions_neg = Var(set_t, within=NonNegativeReals)

        if technology_model == 'RES':
            # Set emissions to zero
            def init_tec_emissions_pos(const, t):
                return b_tec.var_tec_emissions_pos[t] == 0

            b_tec.const_tec_emissions_pos = Constraint(set_t, rule=init_tec_emissions_pos)

            def init_tec_emissions_neg(const, t):
                return b_tec.var_tec_emissions_neg[t] == 0

            b_tec.const_tec_emissions_neg = Constraint(set_t, rule=init_tec_emissions_neg)

        else:

            if emissions_based_on == 'output':
                def init_tec_emissions_pos(const, t):
                    if performance_data['emission_factor'] >= 0:
                        return b_tec.var_output[t, performance_data['main_output_carrier']] * \
                               b_tec.para_tec_emissionfactor == \
                               b_tec.var_tec_emissions_pos[t]
                    else:
                        return b_tec.var_tec_emissions_pos[t] == 0

                b_tec.const_tec_emissions_pos = Constraint(set_t, rule=init_tec_emissions_pos)

                def init_tec_emissions_neg(const, t):
                    if performance_data['emission_factor'] < 0:
                        return b_tec.var_output[t, performance_data['main_output_carrier']] * \
                               (-b_tec.para_tec_emissionfactor) == \
                               b_tec.var_tec_emissions_neg[t]
                    else:
                        return b_tec.var_tec_emissions_neg[t] == 0

                b_tec.const_tec_emissions_neg = Constraint(set_t, rule=init_tec_emissions_neg)

            elif emissions_based_on == 'input':
                def init_tec_emissions_pos(const, t):
                    if performance_data['emission_factor'] >= 0:
                        return b_tec.var_input[t, performance_data['main_input_carrier']] \
                               * b_tec.para_tec_emissionfactor \
                               == b_tec.var_tec_emissions_pos[t]
                    else:
                        return b_tec.var_tec_emissions_pos[t] == 0

                b_tec.const_tec_emissions_pos = Constraint(set_t, rule=init_tec_emissions_pos)

                def init_tec_emissions_neg(const, t):
                    if performance_data['emission_factor'] < 0:
                        return b_tec.var_input[t, performance_data['main_input_carrier']] \
                                   (-b_tec.para_tec_emissionfactor) == \
                               b_tec.var_tec_emissions_neg[t]
                    else:
                        return b_tec.var_tec_emissions_neg[t] == 0

                b_tec.const_tec_emissions_neg = Constraint(set_t, rule=init_tec_emissions_neg)

        return b_tec

    def __define_auxiliary_vars(self, b_tec, energyhub):
        """
        Defines auxiliary variables, that are required for the modelling of clustered data
        """
        set_t_clustered = energyhub.model.set_t_clustered
        set_t_full = energyhub.model.set_t_full
        fitted_performance = self.fitted_performance
        technology_model = self.technology_model
        existing = self.existing
        if existing:
            size_initial = self.size_initial
            size_max = size_initial
        else:
            size_max = self.size_max

        rated_power = fitted_performance.rated_power

        sequence = energyhub.data.k_means_specs.full_resolution['sequence']

        def init_input_bounds(bounds, t, car):
            return tuple(fitted_performance.bounds['input'][car][t - 1, :] * size_max * rated_power)

        b_tec.var_input_aux = Var(set_t_clustered, b_tec.set_input_carriers, within=NonNegativeReals,
                                  bounds=init_input_bounds)

        b_tec.const_link_full_resolution_input = mc.link_full_resolution_to_clustered(b_tec.var_input_aux,
                                                                                      b_tec.var_input,
                                                                                      set_t_full,
                                                                                      sequence,
                                                                                      b_tec.set_input_carriers)

        def init_output_bounds(bounds, t, car):
            return tuple(fitted_performance.bounds['output'][car][t - 1, :] * size_max * rated_power)

        b_tec.var_output_aux = Var(set_t_clustered, b_tec.set_output_carriers, within=NonNegativeReals,
                                   bounds=init_output_bounds)

        b_tec.const_link_full_resolution_output = mc.link_full_resolution_to_clustered(b_tec.var_output_aux,
                                                                                       b_tec.var_output,
                                                                                       set_t_full,
                                                                                       sequence,
                                                                                       b_tec.set_output_carriers)

        return b_tec

    def __constraints_tec_RES(self, b_tec, energyhub):
        b_tec = constraints_tec_RES(b_tec, self, energyhub)
        return b_tec

    # def fit_technology_performance(self, node_data):
    #
    #     """
    #     Fits performance to respective technology model
    #
    #     :param pd node_data: Dataframe of climate data
    #     """
    #     location = node_data.location
    #     climate_data = node_data.data['climate_data']
    #
    #     # Derive performance parameters for respective performance function type
    #     # GENERIC TECHNOLOGIES
    #     if self.technology_model == 'RES':  # Renewable technologies
    #         if self.name == 'Photovoltaic':
    #             if 'system_type' in self.performance_data:
    #                 self.fitted_performance = perform_fitting_PV(climate_data, location,
    #                                                        system_data=self.performance_data['system_type'])
    #             else:
    #                 self.fitted_performance = perform_fitting_PV(climate_data, location)
    #         elif self.name == 'SolarThermal':
    #             self.fitted_performance = perform_fitting_ST(climate_data)
    #         elif 'WindTurbine' in self.name:
    #             if 'hubheight' in self.performance_data:
    #                 hubheight = self.performance_data['hubheight']
    #             else:
    #                 hubheight = 120
    #             self.fitted_performance = perform_fitting_WT(climate_data, self.name, hubheight)
    #
    #     elif self.technology_model == 'CONV1':  # n inputs -> n output, fuel and output substitution
    #         self.fitted_performance = perform_fitting_tec_CONV1(self.performance_data, climate_data)
    #
    #     elif self.technology_model == 'CONV2':  # n inputs -> n output, fuel substitution
    #         self.fitted_performance = perform_fitting_tec_CONV2(self.performance_data, climate_data)
    #
    #     elif self.technology_model == 'CONV3':  # n inputs -> n output, fixed ratio between inputs and outputs
    #         self.fitted_performance = perform_fitting_tec_CONV3(self.performance_data, climate_data)
    #
    #     elif self.technology_model == 'CONV4':  # 0 inputs -> n outputs, fixed ratio between outputs
    #         self.fitted_performance = perform_fitting_tec_CONV4(self.performance_data, climate_data)
    #
    #     elif self.technology_model == 'STOR':  # storage technologies
    #         self.fitted_performance = perform_fitting_tec_STOR(self.performance_data, climate_data)
    #
    #     # SPECIFIC TECHNOLOGIES
    #     elif self.technology_model == 'DAC_Adsorption':  # DAC adsorption
    #         self.fitted_performance = perform_fitting_tec_DAC_adsorption(self.performance_data, climate_data)
    #
    #     elif self.technology_model.startswith('HeatPump_'):  # Heat Pump
    #         self.fitted_performance = perform_fitting_tec_HP(self.performance_data, climate_data, self.technology_model)
    #
    #     elif self.technology_model.startswith('GasTurbine_'):  # Gas Turbine
    #         self.fitted_performance = perform_fitting_tec_GT(self.performance_data, climate_data)
    #
    #     elif self.technology_model == 'Hydro_Open':  # Open Cycle Pumped Hydro
    #         self.fitted_performance = perform_fitting_tec_hydro_open(self.name, self.performance_data, climate_data)



class Res(Technology):

    def __init__(self,
                tec_data):
        super().__init__(tec_data)

    def fit_technology_performance(self, node_data):

        location = node_data.location
        climate_data = node_data.data['climate_data']

        if self.name == 'Photovoltaic':
            if 'system_type' in self.performance_data:
                self.fitted_performance = perform_fitting_PV(climate_data, location,
                                                             system_data=self.performance_data['system_type'])
            else:
                self.fitted_performance = perform_fitting_PV(climate_data, location)
        elif self.name == 'SolarThermal':
            self.fitted_performance = perform_fitting_ST(climate_data)
        elif 'WindTurbine' in self.name:
            if 'hubheight' in self.performance_data:
                hubheight = self.performance_data['hubheight']
            else:
                hubheight = 120
            self.fitted_performance = perform_fitting_WT(climate_data, self.name, hubheight)

    # def add_specific_constraints(self):