import statsmodels.api as sm
import pwlf
import numpy as np

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
    fitting = {}
    x = performance_data['in']
    # Fit performance
    for c in performance_data['out']:
        fitting[c] = dict()
        y = performance_data['out'][c]
        coeff = fit_linear_function(x, y)
        fitting[c]['alpha1'] = round(coeff[0], 6)

    # Calculate input bounds
    fitting['input_bounds'] = np.column_stack((np.zeros(shape=(time_steps)),
                                                   np.ones(shape=(time_steps))))

    fitting['output_bounds'] = {}
    for c in performance_data['out']:
        # Calculate output bounds
        fitting['output_bounds'][c] = np.column_stack((np.zeros(shape=(time_steps)),
                                                       np.ones(shape=(time_steps))*fitting[c]['alpha1']))

    return fitting

def fit_performance_function_type2(performance_data, time_steps):
    """
    Fits performance function for input-output data for type 2

    :param performance_data: performance data
    :return:
    """
    fitting = {}
    x = performance_data['in']
    x = sm.add_constant(x)
    fitting['input_bounds'] = np.column_stack((np.zeros(shape=(time_steps)),
                                                  np.ones(shape=(time_steps))))
    for c in performance_data['out']:
        fitting[c] = dict()
        y = performance_data['out'][c]
        coeff = fit_linear_function(x, y)
        fitting[c]['alpha1'] = round(coeff[1], 6)
        fitting[c]['alpha2'] = round(coeff[0], 6)

    # Calculate input bounds
    fitting['input_bounds'] = np.column_stack((np.zeros(shape=(time_steps)),
                                               np.ones(shape=(time_steps))))
    fitting['output_bounds'] = {}
    for c in performance_data['out']:
        # Calculate output bounds
        fitting['output_bounds'][c] = np.column_stack((np.zeros(shape=(time_steps)),
                                                       np.ones(shape=(time_steps))*fitting[c]['alpha1'] + \
                                                       fitting[c]['alpha2']))
    return fitting


def fit_performance_function_type3(performance_data, nr_seg, time_steps):
    """
    Fits performance function for input-output data for type 1

    :param performance_data: performance data
    :return:
    """
    x = performance_data['in']
    Y = performance_data['out']
    fitting = fit_piecewise_function(x, Y, nr_seg)

    # Calculate input bounds
    fitting['input_bounds'] = np.column_stack((np.zeros(shape=(time_steps)),
                                               np.ones(shape=(time_steps))))
    fitting['output_bounds'] = {}
    for c in performance_data['out']:
        # Calculate output bounds
        fitting['output_bounds'][c] = np.column_stack((np.zeros(shape=(time_steps)),
                                                       np.ones(shape=(time_steps))*fitting[c]['alpha1'][-1] + \
                                                       fitting[c]['alpha2'][-1]))

    return fitting


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