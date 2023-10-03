import pvlib
from timezonefinder import TimezoneFinder
import pandas as pd
from pathlib import Path
from pyomo.environ import *
from scipy.interpolate import interp1d
import numpy as np

from ..technology import Technology
from ..utilities import FittedPerformance

class Res(Technology):
    def __init__(self,
                tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance(self.performance_data)

    def fit_technology_performance(self, node_data):

        location = node_data.location
        climate_data = node_data.data['climate_data']

        if self.name == 'Photovoltaic':
            if 'system_type' in self.performance_data:
                self.__perform_fitting_PV(climate_data, location, system_data=self.performance_data['system_type'])
            else:
                self.__perform_fitting_PV(climate_data, location)

        elif self.name == 'SolarThermal':
            self.__perform_fitting_ST(climate_data)

        elif 'WindTurbine' in self.name:
            if 'hubheight' in self.performance_data:
                hubheight = self.performance_data['hubheight']
            else:
                hubheight = 120
            self.__perform_fitting_WT(climate_data, hubheight)

    def __perform_fitting_PV(self, climate_data, location, **kwargs):
        """
        Calculates capacity factors and specific area requirements for a PV system
        :param climate_data: contains information on weather data
        :param location: contains lon, lat and altitude
        :param PV_type: (optional) can specify a certain type of module
        :return: returns capacity factors and specific area requirements
        """
        # Todo: get perfect tilting angle
        if not kwargs.__contains__('system_data'):
            system_data = dict()
            system_data['tilt'] = 18
            system_data['surface_azimuth'] = 180
            system_data['module_name'] = 'SunPower_SPR_X20_327'
            system_data['inverter_eff'] = 0.96
        else:
            system_data = kwargs['system_data']

        def define_pv_system(location, system_data):
            """
            defines the pv system
            :param location: location information (latitude, longitude, altitude, time zone)
            :param system_data: contains data on tilt, surface_azimuth, module_name, inverter efficiency
            :return: returns PV model chain, peak power, specific area requirements
            """
            module_database = pvlib.pvsystem.retrieve_sam('CECMod')
            module = module_database[system_data['module_name']]

            # Define temperature losses of module
            temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][
                'open_rack_glass_glass']

            # Create PV model chain
            inverter_parameters = {'pdc0': 5000, 'eta_inv_nom': system_data['inverter_eff']}
            system = pvlib.pvsystem.PVSystem(surface_tilt=system_data['tilt'],
                                             surface_azimuth=system_data['surface_azimuth'],
                                             module_parameters=module,
                                             inverter_parameters=inverter_parameters,
                                             temperature_model_parameters=temperature_model_parameters)

            pv_model = pvlib.modelchain.ModelChain(system, location, spectral_model="no_loss", aoi_model="physical")
            peakpower = module.STC
            specific_area = module.STC / module.A_c / 1000 / 1000

            return pv_model, peakpower, specific_area

        # Define parameters for convinience
        lon = location.lon
        lat = location.lat
        alt = location.altitude

        # Get location
        tf = TimezoneFinder()
        tz = tf.timezone_at(lng=lon, lat=lat)
        location = pvlib.location.Location(lat, lon, tz=tz, altitude=alt)

        # Initialize pv_system
        pv_model, peakpower, specific_area = define_pv_system(location, system_data)

        # Run system with climate data
        pv_model.run_model(climate_data)

        # Calculate cap factors
        power = pv_model.results.ac.p_mp
        capacity_factor = power / peakpower

        # Calculate output bounds
        lower_output_bound = np.zeros(shape=(len(climate_data)))
        upper_output_bound = capacity_factor.to_numpy()
        output_bounds = np.column_stack((lower_output_bound, upper_output_bound))

        # Output Bounds
        self.fitted_performance.bounds['output']['electricity'] = output_bounds
        # Coefficients
        self.fitted_performance.coefficients['capfactor'] = capacity_factor
        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 1
        # Other Data
        self.fitted_performance.other['specific_area'] = specific_area


    def __perform_fitting_ST(self, climate_data):
        # Todo: code this
        print('Not coded yet')

    def __perform_fitting_WT(self, climate_data, hubheight):
        # Load data for wind turbine type
        WT_data = pd.read_csv(Path('./data/technology_data/RES/WT_data/WT_data.csv'), delimiter=';')
        WT_data = WT_data[WT_data['TurbineName'] == self.name]

        # Load wind speed and correct for height
        ws = climate_data['ws10']

        # TODO: make power exponent choice possible
        # TODO: Make different heights possible
        alpha = 1 / 7
        # if data.node_data.windPowerExponent(node) >= 0
        #     alpha = data.node_data.windPowerExponent(node);
        # else:
        #     if data.node_data.offshore(node) == 1:
        #         alpha = 0.45;
        #     else:
        #         alpha = 1 / 7;

        if hubheight > 0:
            ws = ws * (hubheight / 10) ** alpha

        # Make power curve
        rated_power = WT_data.iloc[0]['RatedPowerkW']
        x = np.linspace(0, 35, 71)
        y = WT_data.iloc[:, 13:84]
        y = y.to_numpy()

        f = interp1d(x, y)
        ws[ws < 0] = 0
        capacity_factor = f(ws) / rated_power

        # Calculate output bounds
        lower_output_bound = np.zeros(shape=(len(climate_data)))
        upper_output_bound = capacity_factor[0]
        output_bounds = np.column_stack((lower_output_bound, upper_output_bound))

        # Output Bounds
        self.fitted_performance.bounds['output']['electricity'] = output_bounds
        # Coefficients
        self.fitted_performance.coefficients['capfactor'] = capacity_factor[0]
        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 1
        # Other Data
        self.fitted_performance.rated_power = rated_power / 1000


    def construct_tech_model(self, b_tec, energyhub):
        """
        Adds constraints to technology blocks for tec_type RES (renewable technology)

        **Parameter declarations:**

        - Capacity Factor of technology for each time step.

        **Constraint declarations:**

        - Output of technology. The output can be curtailed in three different ways. For ``curtailment == 0``, there is
          no curtailment possible. For ``curtailment == 1``, the curtailment is continuous. For ``curtailment == 2``,
          the size needs to be an integer, and the technology can only be curtailed discretely, i.e. by turning full
          modules off. For ``curtailment == 0`` (default), it holds:

        .. math::
            Output_{t, car} = CapFactor_t * Size

        :param obj b_tec: technology block
        :param Energyhub energyhub: energyhub instance
        :return: technology block
        """
        super(Res, self).construct_tech_model(b_tec, energyhub)

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        coeff = self.fitted_performance.coefficients
        rated_power = self.fitted_performance.rated_power

        if 'curtailment' in performance_data:
            curtailment = performance_data['curtailment']
        else:
            curtailment = 0

        # PARAMETERS
        # Set capacity factors
        capfactor = coeff['capfactor']

        # CONSTRAINTS
        if curtailment == 0:  # no curtailment allowed (default)
            def init_input_output(const, t, c_output):
                return self.output[t, c_output] == \
                       capfactor[t - 1] * b_tec.var_size * rated_power

            b_tec.const_input_output = Constraint(self.set_t, b_tec.set_output_carriers, rule=init_input_output)

        elif curtailment == 1:  # continuous curtailment
            def init_input_output(const, t, c_output):
                return self.output[t, c_output] <= \
                       capfactor[t - 1] * b_tec.var_size * rated_power

            b_tec.const_input_output = Constraint(self.set_t, b_tec.set_output_carriers,
                                                  rule=init_input_output)

        elif curtailment == 2:  # discrete curtailment
            b_tec.var_size_on = Var(self.set_t, within=NonNegativeIntegers,
                                    bounds=(b_tec.para_size_min, b_tec.para_size_max))

            def init_curtailed_units(const, t):
                return b_tec.var_size_on[t] <= b_tec.var_size

            b_tec.const_curtailed_units = Constraint(self.set_t, rule=init_curtailed_units)

            def init_input_output(const, t, c_output):
                return self.output[t, c_output] == \
                       capfactor[t - 1] * b_tec.var_size_on[t] * rated_power

            b_tec.const_input_output = Constraint(self.set_t, b_tec.set_output_carriers,
                                                  rule=init_input_output)

        return b_tec

    def report_results(self, b_tec):
        """
        Function to report results of technologies after optimization

        :param b_tec: technology model block
        :return: dict results: holds results
        """
        super(Res, self).report_results(b_tec)

        rated_power = self.fitted_performance.rated_power
        capfactor = self.fitted_performance.coefficients['capfactor']
        if self.performance_data['curtailment'] == 2:
            self.results['time_dependent']['units_on'] = [b_tec.var_size_on[t].value for t in self.set_t]
        self.results['time_dependent']['max_out'] = [capfactor[t - 1] * b_tec.var_size.value * rated_power for t in self.set_t]
        for car in b_tec.set_output_carriers:
            self.results['time_dependent']['curtailment_' + car] = \
                self.results['time_dependent']['max_out'] - self.results['time_dependent']['output_' + car]

        return self.results
