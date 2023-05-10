import statsmodels.api as sm
import pwlf
import numpy as np
import json
import scandir
import os

def open_json(tec, rootpath):
    """
    Reads technology data from json file
    """
    # Read in JSON files
    root = rootpath
    file_list = []

    for path, subdirs, files in scandir.walk(root):
        for name in files:
            if tec in name:
                filepath = os.path.join(path, name)
                with open(filepath) as json_file:
                    technology_data = json.load(json_file)


    # Assign name
    technology_data['Name'] = tec
    return technology_data

class Economics:
    """
    Class to manage economic data of technologies and networks
    """
    def __init__(self, economics):
        self.capex_model = economics['CAPEX_model']
        self.capex_data = {}
        if 'unit_CAPEX' in economics:
            self.capex_data['unit_capex'] = economics['unit_CAPEX']
        if 'piecewise_CAPEX' in economics:
            self.capex_data['piecewise_capex'] = economics['piecewise_CAPEX']
        if 'gamma1' in economics:
            self.capex_data['gamma1'] = economics['gamma1']
            self.capex_data['gamma2'] = economics['gamma2']
        if 'gamma3' in economics:
            self.capex_data['gamma3'] = economics['gamma3']
        self.opex_variable = economics['OPEX_variable']
        self.opex_fixed = economics['OPEX_fixed']
        self.discount_rate = economics['discount_rate']
        self.lifetime = economics['lifetime']
        self.decommission_cost = economics['decommission_cost']

def fit_performance_function_type1(performance_data, time_steps):
    """
    Fits performance function for input-output data for type 1

    :param performance_data: performance data
    :return:
    """
    fit = {}
    fit['coeff'] = {}
    
    x = performance_data['in']
    # Fit performance
    for car in performance_data['out']:
        fit['coeff'][car] = {}
        y = performance_data['out'][car]
        coeff = fit_linear_function(x, y)
        fit['coeff'][car]['alpha1'] = round(coeff[0], 6)

    # Calculate input bounds
    fit['input_bounds'] = np.column_stack((np.zeros(shape=(time_steps)),
                                                   np.ones(shape=(time_steps))))

    fit['output_bounds'] = {}
    for car in performance_data['out']:
        # Calculate output bounds
        fit['output_bounds'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                       np.ones(shape=(time_steps))*fit['coeff'][car]['alpha1']))

    return fit

def fit_performance_function_type2(performance_data, time_steps):
    """
    Fits performance function for input-output data for type 2

    :param performance_data: performance data
    :return:
    """
    fit = {}
    fit['coeff'] = {}

    x = performance_data['in']
    x = sm.add_constant(x)
    fit['input_bounds'] = np.column_stack((np.zeros(shape=(time_steps)),
                                                  np.ones(shape=(time_steps))))
    for car in performance_data['out']:
        fit['coeff'][car] = {}
        y = performance_data['out'][car]
        coeff = fit_linear_function(x, y)
        fit['coeff'][car]['alpha1'] = round(coeff[1], 6)
        fit['coeff'][car]['alpha2'] = round(coeff[0], 6)

    # Calculate input bounds
    fit['input_bounds'] = np.column_stack((np.zeros(shape=(time_steps)),
                                               np.ones(shape=(time_steps))))
    fit['output_bounds'] = {}
    for car in performance_data['out']:
        # Calculate output bounds
        fit['output_bounds'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                       np.ones(shape=(time_steps))*fit['coeff'][car]['alpha1'] + \
                                                       fit['coeff'][car]['alpha2']))
    return fit


def fit_performance_function_type3(performance_data, nr_seg, time_steps):
    """
    Fits performance function for input-output data for type 1

    :param performance_data: performance data
    :return:
    """
    x = performance_data['in']
    Y = performance_data['out']
    fit = fit_piecewise_function(x, Y, nr_seg)

    # Calculate input bounds
    fit['input_bounds'] = np.column_stack((np.zeros(shape=(time_steps)),
                                               np.ones(shape=(time_steps))))
    fit['output_bounds'] = {}
    for car in performance_data['out']:
        # Calculate output bounds
        fit['output_bounds'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                       np.ones(shape=(time_steps))*fit['coeff'][car]['alpha1'][-1] + \
                                                       fit['coeff'][car]['alpha2'][-1]))

    return fit


def fit_linear_function(x, y):
    """
    Fits linear model to x and y data and returns coefficients
    """
    linmodel = sm.OLS(y, x)
    linfit = linmodel.fit()
    coeff = linfit.params
    return coeff

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
        if x_bp is None:
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


    fit = {}
    fit['coeff'] = {}

    for idx, car in enumerate(Y):
        fit['coeff'][car] = {}
        y = np.array(Y[car])
        if idx == 0:
            bp_x, bp_y, alpha1, alpha2 = regress_piecewise(X, y, nr_segments)
            bp_x0 = bp_x
        else:
            bp_x, bp_y, alpha1, alpha2 = regress_piecewise(X, y, nr_segments, bp_x0)

        fit['coeff'][car]['alpha1'] = alpha1
        fit['coeff'][car]['alpha2'] = alpha2
        fit['coeff'][car]['bp_y'] = bp_y
        fit['coeff']['bp_x'] = bp_x

    return fit