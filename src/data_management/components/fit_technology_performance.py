import statsmodels.api as sm
import numpy as np
from scipy import optimize
import pvlib
from timezonefinder import TimezoneFinder
import pandas as pd
from scipy.interpolate import interp1d

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

    # return fit
    fitting = dict()
    fitting['capacity_factor'] = capacity_factor
    fitting['specific_area'] = specific_area
    return fitting

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
    rated_power =  WT_data.iloc[0]['RatedPowerkW']
    x = np.linspace(0, 35, 71)
    y = WT_data.iloc[:,13:84]
    y = y.to_numpy()

    f = interp1d(x, y)
    capacity_factor = f(ws) / rated_power

    # return fit
    fitting = dict()
    fitting['capacity_factor'] = capacity_factor[0]
    fitting['rated_power'] = rated_power / 1000

    return fitting


def perform_fitting_tec_CONV1(tec_data):
    """
    Fits conversion technology type 1 and returns fitted parameters as a dict
    :param performance_data: contains X and y data of technology performance
    :param performance_function_type: options for type of performance function (linear, piecewise,...)
    :param nr_seg: number of segments on piecewise defined function
    """
    performance_data = tec_data['performance']
    performance_function_type = tec_data['performance_function_type']
    if 'nr_segments_piecewise' in performance_data:
        nr_seg = performance_data['nr_segments_piecewise']
    else:
        nr_seg = 2

    fitting = {}
    if performance_function_type == 1 or performance_function_type == 2:  # Linear performance function
        fitting['out'] = dict()
        x = performance_data['in']
        if performance_function_type == 2:
            x = sm.add_constant(x)
        y = performance_data['out']
        linmodel = sm.OLS(y, x)
        linfit = linmodel.fit()
        coeff = linfit.params
        if performance_function_type == 1:
            fitting['out']['alpha1'] = round(coeff[0], 5)
        if performance_function_type == 2:
            fitting['out']['alpha1'] = round(coeff[1], 5)
            fitting['out']['alpha2'] = round(coeff[0], 5)
    elif performance_function_type == 3:  # piecewise performance function
        y = {}
        x = performance_data['in']
        y['out'] = performance_data['out']
        fitting = fit_piecewise_function(x, y, nr_seg)
    return fitting

def perform_fitting_tec_CONV2(tec_data):
    """
    Fits conversion technology type 2 and returns fitted parameters as a dict
    :param performance_data: contains X and y data of technology performance
    :param performance_function_type: options for type of performance function (linear, piecewise,...)
    :param nr_seg: number of segments on piecewise defined function
    """
    performance_data = tec_data['performance']
    performance_function_type = tec_data['performance_function_type']
    if 'nr_segments_piecewise' in performance_data:
        nr_seg = performance_data['nr_segments_piecewise']
    else:
        nr_seg = 2

    fitting = {}
    if performance_function_type == 1 or performance_function_type == 2:  # Linear performance function
        x = performance_data['in']
        if performance_function_type == 2:
            x = sm.add_constant(x)
        for c in performance_data['out']:
            fitting[c] = dict()
            y = performance_data['out'][c]
            linmodel = sm.OLS(y, x)
            linfit = linmodel.fit()
            coeff = linfit.params
            if performance_function_type == 1:
                fitting[c]['alpha1'] = round(coeff[0], 5)
            if performance_function_type == 2:
                fitting[c]['alpha1'] = round(coeff[1], 5)
                fitting[c]['alpha2'] = round(coeff[0], 5)
    elif performance_function_type == 3:  # piecewise performance function
        x = performance_data['in']
        Y =  performance_data['out']
        fitting = fit_piecewise_function(x, Y, nr_seg)
    return fitting

def perform_fitting_tec_CONV3(tec_data):
    """
    Fits conversion technology type 3 and returns fitted parameters as a dict
    :param performance_data: contains X and y data of technology performance
    :param performance_function_type: options for type of performance function (linear, piecewise,...)
    :param nr_seg: number of segments on piecewise defined function
    """
    performance_data = tec_data['performance']
    performance_function_type = tec_data['performance_function_type']
    if 'nr_segments_piecewise' in performance_data:
        nr_seg = performance_data['nr_segments_piecewise']
    else:
        nr_seg = 2

    fitting = {}
    if performance_function_type == 1 or performance_function_type == 2:  # Linear performance function
        x = performance_data['in']
        if performance_function_type == 2:
            x = sm.add_constant(x)
        for c in performance_data['out']:
            fitting[c] = dict()
            y = performance_data['out'][c]
            linmodel = sm.OLS(y, x)
            linfit = linmodel.fit()
            coeff = linfit.params
            if performance_function_type == 1:
                fitting[c]['alpha1'] = round(coeff[0], 5)
            if performance_function_type == 2:
                fitting[c]['alpha1'] = round(coeff[1], 5)
                fitting[c]['alpha2'] = round(coeff[0], 5)
    elif performance_function_type == 3:  # piecewise performance function
        x = performance_data['in']
        Y = performance_data['out']
        fitting = fit_piecewise_function(x, Y, nr_seg)
    return fitting

def perform_fitting_tec_STOR(tec_data, climate_data):
    theta = tec_data['performance']['theta']

    fitting = {}
    ambient_loss_factor = (65 - climate_data['dataframe']['temp_air']) / (90 - 65) * theta
    fitting['ambient_loss_factor'] = ambient_loss_factor.to_numpy()
    for par in tec_data['performance']:
        if not par == 'theta':
            fitting[par] = tec_data['performance'][par]

    return fitting


def fit_piecewise_function(X, Y, nr_seg):
    """
    Returns fitted parameters of a piecewise defined function
    :param np.array X: x-values of data
    :param np.array Y: y-values of data
    :param nr_seg: number of segments on piecewise defined function
    :return: x and y breakpoints, slope and intercept parameters of piecewise defined function
    """
    def segments_fit(X, Y, count):
        """
        Fits a piecewise defined function to x-y data
        :param list X: x-values
        :param dict Y: y-values (can have multiple dimensions)
        :param count: how many segments
        :return: x and y coordinates of piecewise defined function
        """
        X = np.array(X)
        xmin = min(X)
        xmax = max(X)
        seg = np.full(count - 1, (xmax - xmin) / count)
        px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        py_init = np.array([])
        for car in Y:
            y = np.array(Y[car])
            py_init = np.append(py_init, np.interp(px_init, X, y))

        def err(p):
            """
            Calculates root mean square error of multiple curve fittings
            """
            # get variables
            free_x_bp = p[:count - 1]
            y_bps = p[count - 1:]
            # Calculate y residuals
            y_bp = np.empty((0, count+1))
            y_res = np.empty(0)
            for idx, car in enumerate(Y):
                y_bp = np.append(y_bp, np.reshape(y_bps[idx*(count+1):(idx+1)*(count+1)], (1,-1)), axis=0)
                y_res = np.append(y_res,
                      np.mean((Y[car] - np.interp(X, np.r_[np.r_[xmin, free_x_bp].cumsum(), xmax], y_bp[idx])) ** 2))
            return np.sum(y_res)

        options = {}
        options['disp'] = 1
        options['maxiter'] = 1500
        r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead', options=options, tol=10^-6)

        # Retrieve results
        px = np.r_[xmin, r.x[:count - 1].cumsum(), xmax].round(5)
        pys = r.x[count - 1:].ravel().round(5)
        py = {}
        for idx, car in enumerate(Y):
            py[car] = pys[idx * (count + 1):(idx + 1) * (count + 1)]
        return px, py

    fitting = {}
    px, py = segments_fit(X, Y, nr_seg)

    for idx, car in enumerate(Y):
        fitting[car] = {}
        alpha1 = []
        alpha2 = []
        for seg in range(0, nr_seg):
            al1 = (py[car][seg + 1] - py[car][seg]) / (px[seg + 1] - px[seg]) # Slope
            al2 = py[car][seg] - (py[car][seg + 1] - py[car][seg]) / (px[seg + 1] - px[seg]) * px[seg] # Intercept
            alpha1.append(al1)
            alpha2.append(al2)
        fitting[car]['alpha1'] = alpha1
        fitting[car]['alpha2'] = alpha2
        fitting[car]['bp_y'] = py[car]
        fitting['bp_x'] = px
    return fitting

