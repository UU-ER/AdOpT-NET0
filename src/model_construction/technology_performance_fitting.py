import statsmodels.api as sm
import numpy as np
from scipy import optimize
import pvlib
import datetime
import pytz
from timezonefinder import TimezoneFinder
import pandas as pd
from scipy.interpolate import interp1d


def fit_performance(technology, tec=None, climate_data=None):
    """
    Fits the performance parameters for a technology.
    :param technology: Dict read from json files with performance data and options for performance fits
    :return: dict of performance coefficients used in the model
    """
    # Initialize parameters dict
    parameters = dict()

    # Get options form file
    tec_type = technology['TechnologyPerf']['tec_type']
    if not (tec_type == 1):
        tec_data = technology['TechnologyPerf']

    # Derive performance parameters for respective performance function type
    if tec_type == 1:  # Renewable technologies
        if tec == 'PV':
            if 'system_type' in technology:
                parameters['fit'] = perform_fitting_PV(climate_data, system_data=technology['system_type'])
            else:
                parameters['fit'] = perform_fitting_PV(climate_data)
        elif tec=='ST':
            parameters['fit'] = perform_fitting_ST(climate_data)
        elif 'WT' in tec:
            if 'hubheight' in technology:
                hubheight = technology['hubheight']
            else:
                hubheight = 120
            parameters['fit'] = perform_fitting_WT(climate_data, technology['Name'], hubheight)


    elif tec_type == 2: # n inputs -> n output, fuel and output substitution
        parameters['fit'] = perform_fitting_tectype2(tec_data)

    elif tec_type == 3: # n inputs -> n output, fuel and output substitution
        parameters['fit'] = perform_fitting_tectype3(tec_data)

    elif tec_type == 6:  # storage technologies
        parameters['fit'] = perform_fitting_tectype6(tec_data, climate_data)

    parameters['TechnologyPerf'] = technology['TechnologyPerf']
    parameters['Economics'] = technology['Economics']
    return parameters

def perform_fitting_PV(climate_data, **kwargs):
    """
    Calculates capacity factors and specific area requirements for a PV system
    :param climate_data: contains information on weather data, and location
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
    lon = climate_data['longitude']
    lat = climate_data['latitude']
    alt = climate_data['altitude']

    # Get location
    tf = TimezoneFinder()
    tz = tf.timezone_at(lng=lon, lat=lat)
    location = pvlib.location.Location(lat, lon, tz=tz, altitude=alt)

    # Initialize pv_system
    pv_model, peakpower, specific_area = define_pv_system(location, system_data)

    # Run system with climate data
    pv_model.run_model(climate_data['dataframe'])

    # Calculate cap factors
    power = pv_model.results.ac.p_mp
    capacity_factor = power / peakpower

    # return parameters
    parameters = dict()
    parameters['capacity_factor'] = capacity_factor
    parameters['specific_area'] = specific_area
    return parameters

def perform_fitting_ST(climate_data):
    # Todo: code this
    print('Not coded yet')

def perform_fitting_WT(climate_data, turbine_model, hubheight):
    # Load data for wind turbine type
    WT_data = pd.read_csv(r'.\data\technology_data\WT_data\WT_data.csv', delimiter=';')
    WT_data = WT_data[WT_data['TurbineName'] == turbine_model]

    # Load wind speed and correct for height
    ws = climate_data['dataframe']['ws10']

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
    name_plate =  WT_data.iloc[0]['RatedPowerkW']
    x = np.linspace(0, 35, 71)
    y = WT_data.iloc[:,13:84]
    y = y.to_numpy()

    f = interp1d(x, y)
    capacity_factor = f(ws) / name_plate

    # return parameters
    parameters = dict()
    parameters['capacity_factor'] = capacity_factor[0]
    return parameters


def perform_fitting_tectype2(tec_data):
    """
    Fits technology type 2 and returns parameters as a dict
    :param performance_data: contains X and y data of technology performance
    :param performance_function_type: options for type of performance function (linear, piecewise,...)
    :param nr_seg: number of segments on piecewise defined function
    """
    performance_data = tec_data['performance']
    performance_function_type = tec_data['performance_function_type']
    if performance_function_type == 3:
        nr_seg = tec_data['nr_segments_piecewise']
    else:
        nr_seg = 3

    parameters = dict()
    if performance_function_type == 1 or performance_function_type == 2:  # Linear performance function
        X = performance_data['in']
        if performance_function_type == 2:
            X = sm.add_constant(X)
        y = performance_data['out']
        linmodel = sm.OLS(y, X)
        linfit = linmodel.fit()
        coeff = linfit.params
        parameters['alpha1'] = round(coeff[1], 5)
        parameters['alpha2'] = round(coeff[0], 5)
    elif performance_function_type == 3:  # piecewise performance function
        X = performance_data['in']
        y = performance_data['out']
        parameters = fit_piecewise_function(X,y,nr_seg)
    return parameters

def perform_fitting_tectype3(tec_data):
    """
    Fits technology type 2 and returns parameters as a dict
    :param performance_data: contains X and y data of technology performance
    :param performance_function_type: options for type of performance function (linear, piecewise,...)
    :param nr_seg: number of segments on piecewise defined function
    """
    performance_data = tec_data['performance']
    performance_function_type = tec_data['performance_function_type']
    if performance_function_type == 3:
        nr_seg = tec_data['nr_segments_piecewise']
    else:
        nr_seg = 3

    parameters = dict()
    if performance_function_type == 1 or performance_function_type == 2:  # Linear performance function
        parameters['alpha1'] = dict()
        parameters['alpha2'] = dict()
        X = performance_data['in']
        if performance_function_type == 2:
            X = sm.add_constant(X)
        for c in performance_data['out']:
            y = performance_data['out'][c]
            linmodel = sm.OLS(y, X)
            linfit = linmodel.fit()
            coeff = linfit.params
            parameters['alpha1'][c] = round(coeff[0], 5)
        if performance_function_type == 2:
            parameters['alpha2'][c] = round(coeff[1], 5)
    elif performance_function_type == 3:  # piecewise performance function
        X = performance_data['in']
        y = performance_data['out']
        parameters = fit_piecewise_function(X,y,nr_seg)
        # TODO: This is currently only coded for a single output
    return parameters

def perform_fitting_tectype6(tec_data, climate_data):
    theta = tec_data['performance']['theta']

    parameters = {}
    parameters['ambient_loss_factor'] =  (65 - climate_data['dataframe']['temp_air']) / (90 - 65) * theta
    for par in tec_data['performance']:
        if not par == 'theta':
            parameters[par] = tec_data['performance'][par]

    return parameters


def fit_piecewise_function(X, Y, nr_seg):
    """
    Returns parameters of a piecewise defined function
    TODO: Code this for multidimensional X data
    :param X: x-values of data
    :param Y: y-values of data
    :param nr_seg: number of segments on piecewise defined function
    :return: x and y breakpoints, slope and intercept parameters of piecewise defined function
    """
    def segments_fit(X, Y, count):
        """
        Fits a piecewise defined function to x-y data
        Thanks to ruoyu0088, available on github
        :param X: x-values
        :param Y: y-values
        :param count: how many segments
        :return: x and y coordinates of piecewise defined function
        """
        xmin = X.min()
        xmax = X.max()
        seg = np.full(count - 1, (xmax - xmin) / count)
        px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

        def func(p):
            seg = p[:count - 1]
            py = p[count - 1:]
            px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
            return px, py

        def err(p):
            px, py = func(p)
            Y2 = np.interp(X, px, py)
            return np.mean((Y - Y2) ** 2)

        r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
        return func(r.x)

    parameters = dict()
    px, py = segments_fit(X, Y, nr_seg)

    alpha1 = []
    alpha2 = []
    for seg in range(0, nr_seg):
        al2 = (py[seg + 1] - py[seg]) / (px[seg + 1] - px[seg]) # Slope
        al1 = py[seg] - (py[seg + 1] - py[seg]) / (px[seg + 1] - px[seg]) * px[seg] # Intercept
        alpha2.append(al2)
        alpha1.append(al1)

    parameters['alpha1'] = alpha1
    parameters['alpha2'] = alpha2
    parameters['bp_x'] = px
    parameters['bp_y'] = py
    return parameters

