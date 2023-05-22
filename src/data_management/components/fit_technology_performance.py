import copy
import statsmodels.api as sm
import numpy as np
from scipy import optimize
from scipy.interpolate import griddata
import pvlib
from timezonefinder import TimezoneFinder
import pandas as pd
from scipy.interpolate import interp1d

from src.data_management.components.utilities import *


def perform_fitting_PV(climate_data, location, **kwargs):
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
        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

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

    # return fit
    fitting = FittedPerformance()
    # Output Bounds
    fitting.bounds['output']['electricity'] = output_bounds
    # Coefficients
    fitting.coefficients['capfactor'] = capacity_factor
    # Time dependent coefficents
    fitting.time_dependent_coefficients = 1
    # Other Data
    fitting.other['specific_area'] = specific_area
    return fitting


def perform_fitting_ST(climate_data):
    # Todo: code this
    print('Not coded yet')


def perform_fitting_WT(climate_data, turbine_model, hubheight):
    # Load data for wind turbine type
    WT_data = pd.read_csv(r'.\data\technology_data\RES\WT_data\WT_data.csv', delimiter=';')
    WT_data = WT_data[WT_data['TurbineName'] == turbine_model]

    # Load wind speed and correct for height
    ws = climate_data['ws10']

    #TODO: make power exponent choice possible
    #TODO: Make different heights possible
    alpha = 1/7
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
    rated_power =  WT_data.iloc[0]['RatedPowerkW']
    x = np.linspace(0, 35, 71)
    y = WT_data.iloc[:,13:84]
    y = y.to_numpy()

    f = interp1d(x, y)
    ws[ws < 0] = 0
    capacity_factor = f(ws) / rated_power

    # Calculate output bounds
    lower_output_bound = np.zeros(shape=(len(climate_data)))
    upper_output_bound = capacity_factor[0]
    output_bounds = np.column_stack((lower_output_bound, upper_output_bound))

    # return fit
    fitting = FittedPerformance()
    # Output Bounds
    fitting.bounds['output']['electricity'] = output_bounds
    # Coefficients
    fitting.coefficients['capfactor'] = capacity_factor[0]
    # Time dependent coefficents
    fitting.time_dependent_coefficients = 1
    # Other Data
    fitting.rated_power = rated_power / 1000
    return fitting


def perform_fitting_tec_CONV1(tec_data, climate_data):
    """
    Fits conversion technology type 1 and returns fitted parameters as a dict
    :param performance_data: contains X and y data of technology performance
    :param performance_function_type: options for type of performance function (linear, piecewise,...)
    :param nr_seg: number of segments on piecewise defined function
    """
    # reshape performance_data for CONV1
    temp = copy.deepcopy(tec_data['performance']['out'])
    tec_data['performance']['out'] = {}
    tec_data['performance']['out']['out'] = temp

    time_steps = len(climate_data)
    fitting = fit_performance_generic_tecs(tec_data, time_steps)

    return fitting

def perform_fitting_tec_CONV2(tec_data, climate_data):
    """
    Fits conversion technology type 2 and returns fitted parameters as a dict
    :param performance_data: contains X and y data of technology performance
    :param performance_function_type: options for type of performance function (linear, piecewise,...)
    :param nr_seg: number of segments on piecewise defined function
    """
    time_steps = len(climate_data)
    if tec_data['size_based_on'] == 'output':
        raise Exception('size_based_on == output for CONV2 not possible.')
    fitting = fit_performance_generic_tecs(tec_data, time_steps)

    return fitting

def perform_fitting_tec_CONV3(tec_data, climate_data):
    """
    Fits conversion technology type 3 and returns fitted parameters as a dict
    :param performance_data: contains X and y data of technology performance
    :param performance_function_type: options for type of performance function (linear, piecewise,...)
    :param nr_seg: number of segments on piecewise defined function
    """
    time_steps = len(climate_data)
    if tec_data['size_based_on'] == 'output':
        raise Exception('size_based_on == output for CONV3 not possible.')
    fitting = fit_performance_generic_tecs(tec_data, time_steps)

    # Input bounds recalculation
    for car in fitting.input_carrier:
        if not car == tec_data['main_input_carrier']:
            fitting.bounds['input'][car] = fitting.bounds['input'][tec_data['main_input_carrier']]\
                                           * tec_data['input_ratios'][car]

    return fitting

def perform_fitting_tec_CONV4(tec_data, climate_data):
    """
    Fits conversion technology type 4 and returns fitted parameters as a dict
    :param performance_data: contains X and y data of technology performance
    :param performance_function_type: options for type of performance function (linear, piecewise,...)
    :param nr_seg: number of segments on piecewise defined function
    """
    time_steps = len(climate_data)

    # return fit
    fitting = FittedPerformance(tec_data)
    # Output Bounds
    fitting.bounds['output'][tec_data['main_output_carrier']] = np.column_stack((np.zeros(shape=(time_steps)),
                                                     np.ones(shape=(time_steps))))
    for car in fitting.output_carrier:
        if not car == tec_data['main_output_carrier']:
            fitting.bounds['output'][car] = fitting.bounds['output'][tec_data['main_output_carrier']]\
                                           * tec_data['output_ratios'][car]

    # Time dependent coefficents
    fitting.time_dependent_coefficients = 0

    # Other Data
    if 'rated_power' in tec_data:
        fitting.rated_power = tec_data['rated_power']

    return fitting

def perform_fitting_tec_STOR(tec_data, climate_data):

    time_steps = len(climate_data)
    # Calculate ambient loss factor
    theta = tec_data['performance']['theta']
    ambient_loss_factor = (65 - climate_data['temp_air']) / (90 - 65) * theta

    # return fit
    fitting = FittedPerformance()
    # Output Bounds
    for car in tec_data['output_carrier']:
        fitting.bounds['output'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                               np.ones(shape=(time_steps))*tec_data['performance']['discharge_max']))
    # Input Bounds
    for car in tec_data['input_carrier']:
        fitting.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                               np.ones(shape=(time_steps))*tec_data['performance']['charge_max']))
    # Coefficients
    fitting.coefficients['ambient_loss_factor'] = ambient_loss_factor.to_numpy()
    for par in tec_data['performance']:
        if not par == 'theta':
            fitting.coefficients[par] = tec_data['performance'][par]
    # Time dependent coefficents
    fitting.time_dependent_coefficients = 1

    return fitting

def perform_fitting_tec_DAC_adsorption(tec_data, climate_data):
    """
    Performs fitting for technology type DAC_adsorption
    :param tec_data: technology data
    :param climate_data: climate data
    :return:
    """

    # Number of timesteps
    time_steps = len(climate_data)

    # Number of segments
    nr_segments = tec_data['nr_segments']

    # Read performance data from file
    performance_data = pd.read_csv('./data/technology_data/CO2Capture/DAC_adsorption_data/dac_adsorption_performance.txt', sep=",")
    performance_data = performance_data.rename(columns={"T": "temp_air", "RH": "humidity"})

    # Unit Conversion of input data
    performance_data.E_tot = performance_data.E_tot.multiply(performance_data.CO2_Out / 3600) # in MWh / h
    performance_data.E_el = performance_data.E_el.multiply(performance_data.CO2_Out / 3600) # in kwh / h
    performance_data.E_th = performance_data.E_th.multiply(performance_data.CO2_Out / 3600) # in kwh / h
    performance_data.CO2_Out = performance_data.CO2_Out / 1000 # in t / h

    # Get humidity and temperature
    RH = copy.deepcopy(climate_data['rh'])
    T = copy.deepcopy(climate_data['temp_air'])

    # Set minimum temperature
    T.loc[T < min(performance_data.temp_air)] = min(performance_data.temp_air)

    # Derive performance points for each timestep
    def interpolate_performance_point(t, rh, point_data, var):
        zi = griddata((point_data.temp_air, point_data.humidity), point_data[var], (T, RH), method='linear')
        return zi

    CO2_Out = np.empty(shape=(len(T), len(performance_data.Point.unique())))
    E_tot = np.empty(shape=(len(T), len(performance_data.Point.unique())))
    E_el = np.empty(shape=(len(T), len(performance_data.Point.unique())))
    for point in performance_data.Point.unique():
        CO2_Out[:, point-1] = interpolate_performance_point(T, RH,
                                                   performance_data.loc[performance_data.Point == point],
                                                   'CO2_Out')
        E_tot[:, point-1] = interpolate_performance_point(T, RH,
                                                   performance_data.loc[performance_data.Point == point],
                                                   'E_tot')
        E_el[:, point-1] = interpolate_performance_point(T, RH,
                                                   performance_data.loc[performance_data.Point == point],
                                                   'E_el')

    # Derive piecewise definition
    alpha = np.empty(shape=(len(T), nr_segments))
    beta = np.empty(shape=(len(T), nr_segments))
    b = np.empty(shape=(len(T), nr_segments+1))
    gamma = np.empty(shape=(len(T), nr_segments))
    delta = np.empty(shape=(len(T), nr_segments))
    a = np.empty(shape=(len(T), nr_segments+1))
    el_in_max = np.empty(shape=(len(T)))
    th_in_max = np.empty(shape=(len(T)))
    out_max = np.empty(shape=(len(T)))
    total_in_max = np.empty(shape=(len(T)))

    print('Deriving performance data for DAC...')

    for timestep in range(len(T)):
        if timestep % 100 == 1:
            print("\rComplete: ", round(timestep/len(T),2)*100, "%", end="")
        # Input-Output relation
        y = {}
        y['CO2_Out'] = CO2_Out[timestep, :]
        time_step_fit = fit_piecewise_function(E_tot[timestep, :], y, int(nr_segments))
        alpha[timestep, :] = time_step_fit['CO2_Out']['alpha1']
        beta[timestep, :] = time_step_fit['CO2_Out']['alpha2']
        b[timestep, :] = time_step_fit['CO2_Out']['bp_x']
        out_max[timestep] = max(time_step_fit['CO2_Out']['bp_y'])
        total_in_max[timestep] = max(time_step_fit['CO2_Out']['bp_x'])

        # Input-Input relation
        y = {}
        y['E_el'] = E_el[timestep, :]
        time_step_fit = fit_piecewise_function(E_tot[timestep, :], y, int(nr_segments))
        gamma[timestep, :] = time_step_fit['E_el']['alpha1']
        delta[timestep, :] = time_step_fit['E_el']['alpha2']
        a[timestep, :] = time_step_fit['E_el']['bp_x']
        el_in_max[timestep] = max(time_step_fit['E_el']['bp_y'])
        th_in_max[timestep] = max(time_step_fit['E_el']['bp_x'])

    print("Complete: ", 100, "%")

    # return fit
    fitting = FittedPerformance()
    # Output Bounds
    fitting.bounds['output']['CO2'] = np.column_stack((np.zeros(shape=(time_steps)),
                                                   out_max))
    # Input Bounds
    fitting.bounds['input']['electricity'] = np.column_stack((np.zeros(shape=(time_steps)),
                                                   el_in_max + th_in_max / tec_data['performance']['eta_elth']))
    fitting.bounds['input']['heat'] = np.column_stack((np.zeros(shape=(time_steps)),
                                                   th_in_max))
    fitting.bounds['input']['total'] = [sum(x) for x in zip(fitting.bounds['input']['heat'],
                                                            fitting.bounds['input']['electricity'])]
    # Coefficients
    fitting.coefficients['alpha'] = alpha
    fitting.coefficients['beta'] = beta
    fitting.coefficients['b'] = b
    fitting.coefficients['gamma'] = gamma
    fitting.coefficients['delta'] = delta
    fitting.coefficients['a'] = a
    # Time dependent coefficents
    fitting.time_dependent_coefficients = 1

    return fitting


def perform_fitting_tec_HP(tec_data, climate_data, HP_type):
    """
    Performs fitting for technology type HeatPump

    The equations are based on Ruhnau, O., Hirth, L., & Praktiknjo, A. (2019). Time series of heat demand and
    heat pump efficiency for energy system modeling. Scientific Data, 6(1).
    https://doi.org/10.1038/s41597-019-0199-y

    :param tec_data: technology data
    :param climate_data: climate data
    :return:
    """
    # Min part-load
    min_part_load = tec_data['min_part_load']

    # Performance function type
    performance_function_type = tec_data['performance_function_type']

    # Ambient air temperature
    T = copy.deepcopy(climate_data['temp_air'])

    # Number of timesteps:
    time_steps = len(climate_data)

    # Determine T_out
    if tec_data['application'] == 'radiator_heating':
        t_out = 40 - T
    elif tec_data['application'] == 'floor_heating':
        t_out = 30 - 0.5 * T
    else:
        t_out = tec_data['T_out']

    # Determine delta T
    delta_T = t_out - T

    # Determine COP
    if HP_type == 'HeatPump_AirSourced':
        cop = 6.08 - 0.09 * delta_T + 0.0005 * delta_T ** 2
    elif HP_type == 'HeatPump_GroundSourced':
        cop = 10.29 - 0.21 * delta_T + 0.0012 * delta_T ** 2
    elif HP_type == 'HeatPump_WaterSourced':
        cop = 9.97 - 0.20 * delta_T + 0.0012 * delta_T ** 2

    print('Deriving performance data for Heat Pump...')

    if performance_function_type == 1 or performance_function_type == 2:  # Linear performance function
        size_alpha = 1
    elif performance_function_type == 3:
        size_alpha = 2
    else:
        raise Exception("performance_function_type must be an integer between 1 and 3")

    fit = {}
    fit['out'] = {}
    alpha1 = np.empty(shape=(time_steps, size_alpha))
    alpha2 = np.empty(shape=(time_steps, size_alpha))
    bp_x = np.empty(shape=(time_steps, size_alpha + 1))
    for idx, cop_t in enumerate(cop):
        if idx % 100 == 1:
            print("\rComplete: ", round(idx/time_steps,2)*100, "%", end="")

        if performance_function_type == 1:
            x = np.linspace(min_part_load, 1, 9)
            y = (x / (1 - 0.9 * (1 - x))) * cop_t * x
            coeff = fit_linear_function(x, y)
            alpha1[idx, :] = coeff[0]
        elif performance_function_type == 2:
            x = np.linspace(min_part_load, 1, 9)
            y = (x / (1 - 0.9 * (1 - x))) * cop_t * x
            x = sm.add_constant(x)
            coeff = fit_linear_function(x, y)
            alpha1[idx, :] = coeff[1]
            alpha2[idx, :] = coeff[0]
        elif performance_function_type == 3:  # piecewise performance function
            y = {}
            x = np.linspace(min_part_load, 1, 9)
            y['out'] = (x / (1 - 0.9 * (1 - x))) * cop_t * x
            time_step_fit = fit_piecewise_function(x, y, 2)
            alpha1[idx, :] = time_step_fit['out']['alpha1']
            alpha2[idx, :] = time_step_fit['out']['alpha2']
            bp_x[idx, :] = time_step_fit['out']['bp_x']
    print("Complete: ", 100, "%")

    # Calculate input bounds
    fit['output_bounds'] = {}
    fit['coeff'] = {}
    if performance_function_type == 1:
        fit['coeff']['alpha1'] = alpha1.round(5)
        for c in tec_data['output_carrier']:
            fit['output_bounds'][c] = np.column_stack((np.zeros(shape=(time_steps)),
                                                       np.ones(shape=(time_steps)) * fit['coeff']['alpha1']))
    elif performance_function_type == 2:  # Linear performance function
        fit['coeff']['alpha1'] = alpha1.round(5)
        fit['coeff']['alpha2'] = alpha2.round(5)
        for c in tec_data['output_carrier']:
            fit['output_bounds'][c] = np.column_stack((np.zeros(shape=(time_steps)),
                                                       np.ones(shape=(time_steps))*fit['coeff']['alpha1'] + \
                                                       fit['coeff']['alpha2']))
    elif performance_function_type == 3:  # Piecewise performance function
        fit['coeff']['alpha1'] = alpha1.round(5)
        fit['coeff']['alpha2'] = alpha2.round(5)
        fit['coeff']['bp_x'] = bp_x.round(5)
        for c in tec_data['output_carrier']:
            fit['output_bounds'][c] = np.column_stack((np.zeros(shape=(time_steps)),
                                                       fit['coeff']['alpha1'][:,-1] + \
                                                       fit['coeff']['alpha2'][:,-1]))

    # return fit
    fitting = FittedPerformance()
    # Output Bounds
    fitting.bounds['output'] = fit['output_bounds']
    # Input Bounds
    for car in tec_data['input_carrier']:
        fitting.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                   np.ones(shape=(time_steps))))
    # Coefficients
    fitting.coefficients = fit['coeff']
    # Time dependent coefficents
    fitting.time_dependent_coefficients = 1

    return fitting

def perform_fitting_tec_GT(tec_data, climate_data):
    """
    Performs fitting for technology type GasTurbine

    The equations and data are based on Weimann, L., Ellerker, M., Kramer, G. J., & Gazzani, M. (2019). Modeling gas
    turbines in multi-energy systems: A linear model accounting for part-load operation, fuel, temperature,
    and sizing effects. International Conference on Applied Energy. https://doi.org/10.46855/energy-proceedings-5280

    :param tec_data: technology data
    :param climate_data: climate data
    :return:
    """
    # Ambient air temperature
    T = copy.deepcopy(climate_data['temp_air'])

    # Number of timesteps:
    time_steps = len(climate_data)

    # Temperature correction factors
    f = np.empty(shape=(time_steps))
    f[T <= 6] =  tec_data['gamma'][0] * (T[T <= 6] / tec_data['T_iso']) + tec_data['delta'][0]
    f[T > 6] =  tec_data['gamma'][1] * (T[T > 6] / tec_data['T_iso']) + tec_data['delta'][1]

    # Derive return
    fit = {}
    fit['coeff'] = {}
    fit['coeff']['f'] = f.round(5)
    fit['coeff']['alpha'] = round(tec_data['alpha'], 5)
    fit['coeff']['beta'] = round(tec_data['beta'], 5)
    fit['coeff']['epsilon'] = round(tec_data['epsilon'], 5)
    fit['coeff']['in_min'] = round(tec_data['in_min'], 5)
    fit['coeff']['in_max'] = round(tec_data['in_max'], 5)
    if len(tec_data['input_carrier']) == 2:
        fit['coeff']['max_H2_admixture'] = tec_data['max_H2_admixture']
    else:
        fit['coeff']['max_H2_admixture'] = 1

    # Input bounds
    fit['input_bounds'] = {}
    for c in tec_data['input_carrier']:
        if c == 'hydrogen':
            fit['input_bounds'][c] = np.column_stack((np.zeros(shape=(time_steps)),
                                                           np.ones(shape=(time_steps)) * tec_data['in_max'] *
                                                      fit['coeff']['max_H2_admixture']))
        else:
            fit['input_bounds'][c] = np.column_stack((np.zeros(shape=(time_steps)),
                                                           np.ones(shape=(time_steps)) * tec_data['in_max']))

    # Output bounds
    fit['output_bounds'] = {}
    fit['output_bounds']['electricity'] = np.column_stack((np.zeros(shape=(time_steps)),
                                                           f * (tec_data['in_max'] * fit['coeff']['alpha'] + fit['coeff']['beta'])))
    fit['output_bounds']['heat'] = np.column_stack((np.zeros(shape=(time_steps)),
                                                           fit['coeff']['epsilon'] * fit['coeff']['in_max'] -
                                                        f * (tec_data['in_max'] * fit['coeff']['alpha'] + fit['coeff']['beta'])))


    # return fit
    fitting = FittedPerformance()
    fitting.rated_power = tec_data['rated_power']
    # Output Bounds
    fitting.bounds['output'] = fit['output_bounds']
    # Input Bounds
    for car in tec_data['input_carrier']:
        fitting.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                   np.ones(shape=(time_steps))))
    # Coefficients
    fitting.coefficients = fit['coeff']
    # Time dependent coefficents
    fitting.time_dependent_coefficients = 1

    return fitting

def perform_fitting_tec_hydro_open(tec_data, climate_data):
    """
    Performs fitting for technology type Hydro_Open

    :param tec_data: technology data
    :param climate_data: climate data
    :return:
    """
    time_steps = len(climate_data)

    # return fit
    fitting = FittedPerformance()
    # Output Bounds
    for car in tec_data['output_carrier']:
        fitting.bounds['output'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                               np.ones(shape=(time_steps))*tec_data['performance']['discharge_max']))
    # Input Bounds
    for car in tec_data['input_carrier']:
        fitting.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                               np.ones(shape=(time_steps))*tec_data['performance']['charge_max']))
    # Coefficients
    for par in tec_data['performance']:
        fitting.coefficients[par] = tec_data['performance'][par]

    # Natural inflow
    if 'hydro_natural_inflow' in climate_data:
        fitting.coefficients['hydro_natural_inflow'] = climate_data['hydro_natural_inflow']
    else:
        raise Exception('Using Technology Type Hydro_Open requires a hydro_natural_inflow in climate data'
                        ' to be defined for this node. You can do this by using DataHandle.read_hydro_natural_inflow')

    # Time dependent coefficents
    fitting.time_dependent_coefficients = 1

    return fitting

