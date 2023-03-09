import copy
import statsmodels.api as sm
import numpy as np
from scipy import optimize
from scipy.interpolate import griddata
import pvlib
from timezonefinder import TimezoneFinder
import pandas as pd
from scipy.interpolate import interp1d
import pwlf


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
            fitting['out']['alpha1'] = coeff[0]
        if performance_function_type == 2:
            fitting['out']['alpha1'] = coeff[1]
            fitting['out']['alpha2'] = coeff[0]
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
                fitting[c]['alpha1'] = coeff[0]
            if performance_function_type == 2:
                fitting[c]['alpha1'] = coeff[1]
                fitting[c]['alpha2'] = coeff[0]
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
                fitting[c]['alpha1'] = coeff[0]
            if performance_function_type == 2:
                fitting[c]['alpha1'] = coeff[1]
                fitting[c]['alpha2'] = coeff[0]
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

def perform_fitting_tec_DAC_adsorption(tec_data, climate_data):
    nr_segments = tec_data['nr_segments']

    # Read performance data from file
    performance_data = pd.read_csv('./data/technology_data/DAC_adsorption_data/dac_adsorption_performance.txt', sep=",")
    performance_data = performance_data.rename(columns={"T": "temp_air", "RH": "humidity"})

    # Unit Conversion of input data
    performance_data.E_tot = performance_data.E_tot.multiply(performance_data.CO2_Out / 3600) # in MWh / h
    performance_data.E_el = performance_data.E_el.multiply(performance_data.CO2_Out / 3600) # in kwh / h
    performance_data.E_th = performance_data.E_th.multiply(performance_data.CO2_Out / 3600) # in kwh / h
    performance_data.CO2_Out = performance_data.CO2_Out / 1000 # in t / h

    # Get humidity and temperature
    RH = copy.deepcopy(climate_data['dataframe']['rh'])
    T = copy.deepcopy(climate_data['dataframe']['temp_air'])

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
        b[timestep, :] = time_step_fit['bp_x']
        out_max[timestep] = max(time_step_fit['CO2_Out']['bp_y'])
        total_in_max[timestep] = max(time_step_fit['bp_x'])

        # Input-Input relation
        y = {}
        y['E_el'] = E_el[timestep, :]
        time_step_fit = fit_piecewise_function(E_tot[timestep, :], y, int(nr_segments))
        gamma[timestep, :] = time_step_fit['E_el']['alpha1']
        delta[timestep, :] = time_step_fit['E_el']['alpha2']
        a[timestep, :] = time_step_fit['bp_x']
        el_in_max[timestep] = max(time_step_fit['E_el']['bp_y'])
        th_in_max[timestep] = max(time_step_fit['bp_x'])

    print("Complete: ", 100, "%")
    fitting = {}
    fitting['alpha'] = alpha
    fitting['beta'] = beta
    fitting['b'] = b
    fitting['gamma'] = gamma
    fitting['delta'] = delta
    fitting['a'] = a
    fitting['el_in_max'] = el_in_max
    fitting['th_in_max'] = th_in_max
    fitting['out_max'] = out_max
    fitting['rated_power'] = 1
    return fitting


def fit_piecewise_function(X, Y, nr_segments):
    """
    Returns fitted parameters of a piecewise defined function with multiple y-series
    :param np.array X: x-values of data
    :param np.array Y: y-values of data
    :param nr_seg: number of segments on piecewise defined function
    :return: x and y breakpoints, slope and intercept parameters of piecewise defined function
    """

    def regress_piecewise(x, y, nr_segments, x_bp=None):
        """
        Returns fitted parameters of a piecewise defined function
        :param np.array X: x-values of data
        :param np.array y: y-values of data
        :param nr_seg: number of segments on piecewise defined function
        :return: x and y breakpoints, slope and intercept parameters of piecewise defined function
        """
        # Perform fit
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        if not x_bp:
            my_pwlf.fit(nr_segments)
        else:
            my_pwlf.fit_with_breaks(x_bp)

        # retrieve data
        bp_x = my_pwlf.fit_breaks
        bp_y = my_pwlf.predict(bp_x)

        alpha1 = []
        alpha2 = []
        for seg in range(0, nr_segments):
            al1 = (bp_y[seg + 1] - bp_y[seg]) / (bp_x[seg + 1] - bp_x[seg])  # Slope
            al2 = bp_y[seg] - (bp_y[seg + 1] - bp_y[seg]) / (bp_x[seg + 1] - bp_x[seg]) * bp_x[seg]  # Intercept
            alpha1.append(al1)
            alpha2.append(al2)

        return bp_x, bp_y, alpha1, alpha2


    fitting = {}
    for idx, car in enumerate(Y):
        fitting[car] = {}
        y = np.array(Y[car])
        if idx == 0:
            bp_x, bp_y, alpha1, alpha2 = regress_piecewise(X, y, nr_segments)
            bp_x0 = bp_x
        else:
            bp_x, bp_y, alpha1, alpha2 = regress_piecewise(X, y, nr_segments, bp_x0)

        fitting[car]['alpha1'] = alpha1
        fitting[car]['alpha2'] = alpha2
        fitting[car]['bp_y'] = bp_y
        fitting['bp_x'] = bp_x

    return fitting

