import pandas as pd
from pyomo.gdp import *
from warnings import warn
from pyomo.environ import *

from ..component import ModelComponent
from ..utilities import annualize, set_discount_rate, link_full_resolution_to_clustered
from .utilities import set_capex_model

"""
TODO
Suggestions:
- add sources to documentation
- Delete src/model_construction/generic_technology_constraints.py
"""

class Technology(ModelComponent):
    """
    Class to read and manage data for technologies
    """
    def __init__(self,tec_data):
        """
        Initializes technology class from technology name

        The technology name needs to correspond to the name of a JSON file in ./data/technology_data.

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)

        self.technology_model = tec_data['tec_type']
        self.modelled_with_full_res = []

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

        self.input = []
        self.output = []
        self.set_t = []
        self.set_t_full = []
        self.sequence = []

    def construct_tech_model(self, b_tec, energyhub):
        r"""
        This function adds Sets, Parameters, Variables and Constraints that are common for all technologies.
        For each technology type, individual parts are added.
        The following description is true for new technologies. For existing technologies a few adaptions are made
        (see below).

        **Set declarations:**

        - Set of input carriers
        - Set of output carriers

        **Parameter declarations:**

        - Min Size
        - Max Size
        - Output max (same as size max)
        - Unit CAPEX (annualized from given data on up-front CAPEX, lifetime and discount rate)
        - Variable OPEX
        - Fixed OPEX

        **Variable declarations:**

        - Size (can be integer or continuous)
        - Input for each input carrier
        - Output for each output carrier
        - CAPEX
        - Variable OPEX
        - Fixed OPEX

        **Constraint declarations**
        - CAPEX, can be linear (for ``capex_model == 1``), piecewise linear (for ``capex_model == 2``) or linear with
        a fixed cost when the technology is installed (for ``capex_model == 3``). Linear is defined as:

        .. math::
            CAPEX_{tec} = Size_{tec} * UnitCost_{tec}

        while linear with fixed installation costs is defined as:
                .. math::
            CAPEX_{tec} = Size_{tec} * UnitCost_{tec} + FixCost_{tec}

        - Variable OPEX: defined per unit of output for the main carrier:

        .. math::
            OPEXvar_{t, tec} = Output_{t, maincarrier} * opex_{var} \forall t \in T

        - Fixed OPEX: defined as a fraction of annual CAPEX:

        .. math::
            OPEXfix_{tec} = CAPEX_{tec} * opex_{fix}

        Existing technologies, i.e. existing = 1, can be decommissioned (decommission = 1) or not (decommission = 0).
        For technologies that cannot be decommissioned, the size is fixed to the size given in the technology data.
        For technologies that can be decommissioned, the size can be smaller or equal to the initial size. Reducing the
        size comes at the decommissioning costs specified in the economics of the technology.
        The fixed opex is calculated by determining the capex that the technology would have costed if newly build and
        then taking the respective opex_fixed share of this. This is done with the auxiliary variable var_capex_aux.

        :param str nodename: name of node for which technology is installed
        :param set set_tecsToAdd: list of technologies to add
        :param energyhub EnergyHub: instance of the energyhub
        :return: b_node
        """

        print('\t - Adding Technology ' + self.name)

        # TECHNOLOGY DATA
        configuration = energyhub.configuration

        # MODELING TYPICAL DAYS
        if energyhub.model_information.clustered_data:
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

        # GENERAL TECHNOLOGY CONSTRAINTS
        b_tec = self.__define_size(b_tec)
        b_tec = self.__define_capex(b_tec, energyhub)
        b_tec = self.__define_input(b_tec, energyhub)
        b_tec = self.__define_output(b_tec, energyhub)
        b_tec = self.__define_opex(b_tec, energyhub)
        b_tec = self.__define_emissions(b_tec, energyhub)

        # CLUSTERED DATA
        if energyhub.model_information.clustered_data and not self.modelled_with_full_res:
            b_tec = self.__define_auxiliary_vars(b_tec, energyhub)
        else:
            if not (self.technology_model == 'RES') and not (self.technology_model == 'CONV4'):
                self.input = b_tec.var_input
            self.output = b_tec.var_output
            self.set_t = energyhub.model.set_t_full
            self.sequence = list(self.set_t)
        self.set_t_full = energyhub.model.set_t_full

        # DYNAMICS
        if energyhub.configuration.performance.dynamics:
            technologies_modelled_with_dynamics = ['CONV1', 'CONV2', 'CONV3']
            if self.technology_model in technologies_modelled_with_dynamics:
                b_tec = self.__define_dynamics(b_tec)
            else:
                warn('Modeling dynamic constraints not enabled for technology type' + self.name)
        else:
            if hasattr(self.performance_data, "performance_function_type"):
                if self.performance_data.performance_function_type == 4:
                    self.performance_data.performance_function_type = 3
                    warn('Switching dynamics off for performance function type 4, type changed to 3 for ' + self.name)

        return b_tec

    def report_results(self, b_tec):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        self.results['time_independent']['technology'] = [self.name]
        self.results['time_independent']['size'] = [b_tec.var_size.value]
        self.results['time_independent']['existing'] = [self.existing]
        self.results['time_independent']['capex'] = [b_tec.var_capex.value]
        self.results['time_independent']['opex_variable'] = [sum(b_tec.var_opex_variable[t].value for t in self.set_t_full)]
        self.results['time_independent']['opex_fixed'] = [b_tec.var_opex_fixed.value]
        self.results['time_independent']['emissions_pos'] = [sum(b_tec.var_tec_emissions_pos[t].value for t in self.set_t_full)]
        self.results['time_independent']['emissions_neg'] = [sum(b_tec.var_tec_emissions_neg[t].value for t in self.set_t_full)]

        for car in b_tec.set_input_carriers:
            if b_tec.find_component('var_input'):
                self.results['time_dependent']['input_' + car] = [b_tec.var_input[t, car].value for t in self.set_t_full]
        for car in b_tec.set_output_carriers:
            self.results['time_dependent']['output_' + car] = [b_tec.var_output[t, car].value for t in self.set_t_full]
        self.results['time_dependent']['emissions_pos'] = [b_tec.var_tec_emissions_pos[t].value for t in self.set_t_full]
        self.results['time_dependent']['emissions_neg'] = [b_tec.var_tec_emissions_neg[t].value for t in self.set_t_full]
        if b_tec.find_component('var_x'):
            self.results['time_dependent']['var_x'] = [b_tec.var_x[t].value for t in self.set_t_full]
        if b_tec.find_component('var_y'):
            self.results['time_dependent']['var_y'] = [b_tec.var_y[t].value for t in self.set_t_full]
        if b_tec.find_component('var_z'):
            self.results['time_dependent']['var_z'] = [b_tec.var_z[t].value for t in self.set_t_full]

        return self.results

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
        if self.existing:
            size_max = self.size_initial
        else:
            size_max = self.size_max

        if self.size_is_int:
            size_domain = NonNegativeIntegers
        else:
            size_domain = NonNegativeReals

        b_tec.para_size_min = Param(domain=NonNegativeReals, initialize=self.size_min, mutable=True)
        b_tec.para_size_max = Param(domain=NonNegativeReals, initialize=size_max, mutable=True)

        if self.existing:
            b_tec.para_size_initial = Param(within=size_domain, initialize=self.size_initial)

        if self.existing and not self.decommission:
            # Decommissioning is not possible, size fixed
            b_tec.var_size = Param(within=size_domain, initialize=b_tec.para_size_initial)
        else:
            # Decommissioning is possible, size variable
            b_tec.var_size = Var(within=size_domain, bounds=(b_tec.para_size_min,
                                                                        b_tec.para_size_max))

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

        economics = self.economics
        discount_rate = set_discount_rate(configuration, economics)
        capex_model = set_capex_model(configuration, economics)

        # CAPEX auxiliary (used to calculate theoretical CAPEX)
        # For new technologies, this is equal to actual CAPEX
        # For existing technologies it is used to calculate fixed OPEX
        b_tec.var_capex_aux = Var()
        annualization_factor = annualize(discount_rate, economics.lifetime)
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
            self.big_m_transformation_required = 1
            b_tec.const_capex_aux = Piecewise(b_tec.var_capex_aux, b_tec.var_size,
                                              pw_pts=b_tec.para_bp_x,
                                              pw_constr_type='EQ',
                                              f_rule=b_tec.para_bp_y_annual,
                                              pw_repn='SOS2')
        elif capex_model == 3:
            b_tec.para_unit_capex = Param(domain=Reals, initialize=economics.capex_data['unit_capex'], mutable=True)
            b_tec.para_fix_capex = Param(domain=Reals, initialize=economics.capex_data['fix_capex'], mutable=True)
            b_tec.para_unit_capex_annual = Param(domain=Reals,
                                                 initialize=annualization_factor * economics.capex_data['unit_capex'],
                                                 mutable=True)
            b_tec.para_fix_capex_annual = Param(domain=Reals,
                                                 initialize=annualization_factor * economics.capex_data['fix_capex'],
                                                 mutable=True)
            b_tec.const_capex_aux = Constraint(
                expr=b_tec.var_size * b_tec.para_unit_capex_annual + b_tec.para_fix_capex_annual == b_tec.var_capex_aux)
        elif capex_model == 4:
            pass
            # Needs to be overwritten in specific technology

        # CAPEX
        if self.existing and not self.decommission:
            b_tec.var_capex = Param(domain=Reals, initialize=0)
        else:
            b_tec.var_capex = Var()
            if self.existing:
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
        if energyhub.model_information.clustered_data and not modelled_with_full_res:
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
                if energyhub.model_information.clustered_data and not modelled_with_full_res:
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
        if energyhub.model_information.clustered_data and not modelled_with_full_res:
            sequence = energyhub.data.k_means_specs.full_resolution['sequence']

        if existing:
            size_initial = self.size_initial
            size_max = size_initial
        else:
            size_max = self.size_max

        b_tec.set_output_carriers = Set(initialize=performance_data['output_carrier'])

        def init_output_bounds(bounds, t, car):
            if energyhub.model_information.clustered_data and not modelled_with_full_res:
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
        self.set_t = set_t_clustered
        self.sequence = energyhub.data.k_means_specs.full_resolution['sequence']

        if self.existing:
            size_initial = self.size_initial
            size_max = size_initial
        else:
            size_max = self.size_max

        rated_power = self.fitted_performance.rated_power

        sequence = energyhub.data.k_means_specs.full_resolution['sequence']

        if not (self.technology_model == 'RES') and not (self.technology_model == 'CONV4'):
            def init_input_bounds(bounds, t, car):
                return tuple(self.fitted_performance.bounds['input'][car][t - 1, :] * size_max * rated_power)

            b_tec.var_input_aux = Var(set_t_clustered, b_tec.set_input_carriers, within=NonNegativeReals,
                                      bounds=init_input_bounds)

            b_tec.const_link_full_resolution_input = link_full_resolution_to_clustered(b_tec.var_input_aux,
                                                                                          b_tec.var_input,
                                                                                          set_t_full,
                                                                                          sequence,
                                                                                          b_tec.set_input_carriers)
            self.input = b_tec.var_input_aux

        def init_output_bounds(bounds, t, car):
            return tuple(self.fitted_performance.bounds['output'][car][t - 1, :] * size_max * rated_power)

        b_tec.var_output_aux = Var(set_t_clustered, b_tec.set_output_carriers, within=NonNegativeReals,
                                   bounds=init_output_bounds)

        b_tec.const_link_full_resolution_output = link_full_resolution_to_clustered(b_tec.var_output_aux,
                                                                                       b_tec.var_output,
                                                                                       set_t_full,
                                                                                       sequence,
                                                                                       b_tec.set_output_carriers)

        self.output = b_tec.var_output_aux

        return b_tec

    def __define_dynamics(self, b_tec):
        """
        Selects the dynamic constraints that are required based on the technology dynamic performance parameters or the
        performance function type.
        :param b_tec:
        :return:
        """
        SU_load = self.performance_data['SU_load']
        SD_load = self.performance_data['SD_load']
        min_uptime = self.performance_data['min_uptime']
        min_downtime = self.performance_data['min_downtime']
        max_startups = self.performance_data['max_startups']
        performance_function_type4 = ("performance_function_type" in self.performance_data) and \
                                     (self.performance_data['performance_function_type'] == 4)
        if (min_uptime + min_downtime > -2) or (max_startups > -1) or (SU_load + SD_load > -2) or \
                performance_function_type4:
            b_tec = self.__dynamics_SUSD_logic(b_tec)
        if not performance_function_type4 and (SU_load + SD_load > -2):
            b_tec = self.__dynamics_fast_SUSD(b_tec)

        return b_tec

    def __dynamics_SUSD_logic(self, b_tec):
        """
        Adds the startup and shutdown logic to the technology model and constrains the maximum number of startups.

        Based on Equations 4-5 in Morales-España, G., Ramírez-Elizondo, L., & Hobbs, B. F. (2017). Hidden power system
        inflexibilities imposed by traditional unit commitment formulations. Applied Energy, 191, 223–238.
        https://doi.org/10.1016/J.APENERGY.2017.01.089

        """
        # New variables
        b_tec.var_x = Var(self.set_t_full, domain=NonNegativeReals, bounds=(0, 1))
        b_tec.var_y = Var(self.set_t_full, domain=NonNegativeReals, bounds=(0, 1))
        b_tec.var_z = Var(self.set_t_full, domain=NonNegativeReals, bounds=(0, 1))

        # Check for default values
        para_names = ['SU_time', 'SD_time']
        for para in para_names:
            if self.performance_data[para] < 0:
                self.performance_data[para] = 0
                warn("Using SU/SD logic constraints, parameter " + str(para) + " set to default value 0")
        para_names = ['min_uptime', 'min_downtime']
        for para in para_names:
            if self.performance_data[para] < 0:
                self.performance_data[para] = 1
                warn("Using SU/SD logic constraints, parameter " + str(para) + " set to default value 1")

        # Collect parameters
        SU_time = self.performance_data['SU_time']
        SD_time = self.performance_data['SD_time']
        min_uptime = self.performance_data['min_uptime']
        min_downtime = self.performance_data['min_downtime'] + SU_time + SD_time
        max_startups = self.performance_data['max_startups']

        # Enforce startup/shutdown logic
        def init_SUSD_logic1(const, t):
            if t == 1:
                return Constraint.Skip
            else:
                return b_tec.var_x[t] - b_tec.var_x[t - 1] == b_tec.var_y[t] - b_tec.var_z[t]

        b_tec.const_SUSD_logic1 = Constraint(self.set_t_full, rule=init_SUSD_logic1)

        def init_SUSD_logic2(const, t):
            if t >= min_uptime:
                return b_tec.var_y[t - min_uptime + 1] <= b_tec.var_x[t]
            else:
                return b_tec.var_y[len(self.set_t_full) + (t - min_uptime + 1)] <= b_tec.var_x[t]

        b_tec.const_SUSD_logic2 = Constraint(self.set_t_full, rule=init_SUSD_logic2)

        def init_SUSD_logic3(const, t):
            if t >= min_downtime:
                return b_tec.var_z[t - min_downtime + 1] <= 1 - b_tec.var_x[t]
            else:
                return b_tec.var_z[len(self.set_t_full) + (t - min_downtime + 1)] <= 1 - b_tec.var_x[t]

        b_tec.const_SUSD_logic3 = Constraint(self.set_t_full, rule=init_SUSD_logic3)

        # Constrain number of startups
        if not max_startups == -1:
            def init_max_startups(const):
                return sum(b_tec.var_y[t] for t in self.set_t_full) <= max_startups

            b_tec.const_max_startups = Constraint(rule=init_max_startups)

        return b_tec

    def __dynamics_fast_SUSD(self, b_tec):
        """
        Adds startup and shutdown load constraints to the model.

        Based on Equations 9-11 and 13 in Morales-España, G., Ramírez-Elizondo, L., & Hobbs, B. F. (2017). Hidden power
        system inflexibilities imposed by traditional unit commitment formulations. Applied Energy, 191, 223–238.
        https://doi.org/10.1016/J.APENERGY.2017.01.089

        """

        # Check for default values
        para_names = ['SU_load', 'SD_load']
        for para in para_names:
            if self.performance_data[para] < 0:
                self.performance_data[para] = 1
                warn("Using SU/SD load constraints, parameter" + str(para) + "set to default value 1")


        # Collect parameters
        SU_load = self.performance_data['SU_load']
        SD_load = self.performance_data['SD_load']
        main_car = self.performance_data['main_input_carrier']
        rated_power = self.fitted_performance.rated_power

        # SU load limit
        s_indicators = range(0, 2)

        def init_SU_load(dis, t, ind):
            if ind == 0:  # no startup (y=0)
                dis.const_y_off = Constraint(expr=b_tec.var_y[t] == 0)

            else:  # tech in startup
                dis.const_y_on = Constraint(expr=b_tec.var_y[t] == 1)

                def init_SU_load_limit(cons):
                    if self.technology_model == 'CONV3':
                        return self.input[t, main_car] <= b_tec.var_size * SU_load * rated_power
                    else:
                        return sum(self.input[t, car_input] for car_input in b_tec.set_input_carriers) \
                               <= b_tec.var_size * SU_load * rated_power
                dis.const_SU_load_limit = Constraint(rule=init_SU_load_limit)

        b_tec.dis_SU_load = Disjunct(self.set_t_full, s_indicators, rule=init_SU_load)

        def bind_disjunctions_SU_load(dis, t):
            return [b_tec.dis_SU_load[t, i] for i in s_indicators]

        b_tec.disjunction_SU_load = Disjunction(self.set_t_full, rule=bind_disjunctions_SU_load)

        # SD load limit
        s_indicators = range(0, 2)

        def init_SD_load(dis, t, ind):
            if ind == 0:  # no shutdown (z=0)
                dis.const_z_off = Constraint(expr=b_tec.var_z[t] == 0)

            else:  # tech in shutdown
                dis.const_z_on = Constraint(expr=b_tec.var_z[t] == 1)

                def init_SD_load_limit(cons):
                    if t == 1:
                        return Constraint.Skip
                    else:
                        if self.technology_model == 'CONV3':
                            return self.input[t - 1, main_car] <= b_tec.var_size * SD_load * rated_power
                        else:
                            return sum(self.input[t - 1, car_input] for car_input in b_tec.set_input_carriers) \
                                   <= b_tec.var_size * SD_load * rated_power

                dis.const_SD_load_limit = Constraint(rule=init_SD_load_limit)

        b_tec.dis_SD_load = Disjunct(self.set_t_full, s_indicators, rule=init_SD_load)

        def bind_disjunctions_SD_load(dis, t):
            return [b_tec.dis_SD_load[t, i] for i in s_indicators]

        b_tec.disjunction_SD_load = Disjunction(self.set_t_full, rule=bind_disjunctions_SD_load)

        return b_tec