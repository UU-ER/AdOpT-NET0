import statsmodels.api as sm
import pwlf
import numpy as np
import json
import os
from pathlib import Path
from math import floor, log10

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


def fit_performance_generic_tecs(tec_data, time_steps):
    """
    Fits technology performance according to performance function type
    :param tec_data: technology data
    :param time_steps: number of timesteps
    :return: fitting
    """

    performance_data = tec_data['performance']
    performance_function_type = tec_data['performance_function_type']
    size_based_on = tec_data['size_based_on']

    # Calculate fit
    if performance_function_type == 1:
        fitting = FitGenericTecTypeType1(tec_data)
    elif performance_function_type == 2:
        fitting = FitGenericTecTypeType2(tec_data)
    elif performance_function_type == 3:
        fitting = FitGenericTecTypeType3(tec_data)
    else:
        raise Exception("performance_function_type must be an integer between 1 and 3")
    fitting.fit_performance_function(performance_data)
    fitting.calculate_input_bounds(size_based_on, time_steps)
    fitting.calculate_output_bounds(size_based_on, time_steps)

    # Write remaining information to object
    if 'rated_power' in tec_data:
        fitting.rated_power = tec_data['rated_power']
    fitting.time_dependent_coefficients = 0
    return fitting


class FittedPerformance:
    """
    Class to manage performance of technologies
    """

    def __init__(self, tec_data=None):
        self.rated_power = 1
        self.bounds = {'input': {}, 'output': {}}
        self.coefficients = {}
        self.time_dependent_coefficients = 0
        self.other = {}
        if tec_data:
            if 'input_carrier' in tec_data:
                self.input_carrier = tec_data['input_carrier']
            if 'output_carrier' in tec_data:
                self.output_carrier = tec_data['output_carrier']


class FitGenericTecTypeType1(FittedPerformance):
    """
    Subclass to fit performance of type1 performance functions (linear, through origin)
    out = alpha1 * in
    """

    def fit_performance_function(self, performance_data):
        """
        Fits performance function for input-output data for type 1 technologies

        :param performance_data: performance data
        :return: coefficients
        """
        x = performance_data['in']

        for car in performance_data['out']:
            self.coefficients[car] = {}
            y = performance_data['out'][car]
            fit = fit_linear_function(x, y)
            self.coefficients[car]['alpha1'] = sig_figs(fit[0], 6)

    def calculate_input_bounds(self, size_based_on, time_steps):
        """
        Calculates input bounds for type 1 generic technologies

        :param str size_based_on: 'input' or 'output'
        :return: input bounds
        """
        if size_based_on == 'input':
            for car in self.input_carrier:
                self.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                             np.ones(shape=(time_steps))))
        elif size_based_on == 'output':
            for car in self.input_carrier:
                if car in self.coefficients:
                    car_aux = car
                else:
                    car_aux = 'out'
                self.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                             np.ones(shape=(time_steps)) /
                                                             self.coefficients[car_aux]['alpha1']))
        else:
            raise Exception('size_based_on must be either input or output')

    def calculate_output_bounds(self, size_based_on, time_steps):
        """
        Calculates output bounds for type 1 generic technologies

        :param str size_based_on: 'input' or 'output'
        :return: output bounds
        """
        if size_based_on == 'input':
            for car in self.output_carrier:
                if car in self.coefficients:
                    car_aux = car
                else:
                    car_aux = 'out'
                self.bounds['output'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                              np.ones(shape=(time_steps)) *
                                                              self.coefficients[car_aux]['alpha1']))
        elif size_based_on == 'output':
            for car in self.output_carrier:
                self.bounds['output'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                              np.ones(shape=(time_steps))))
        else:
            raise Exception('size_based_on must be either input or output')


class FitGenericTecTypeType2(FittedPerformance):
    """
    Subclass to fit performance of type1 performance functions (linear, with min partload)
    out = alpha1 * in + alpha2
    (out - alpha2)/alpha1
    """

    def fit_performance_function(self, performance_data):
        """
        Fits performance function for input-output data for type 2 technologies

        :param performance_data: performance data
        :return: coefficients
        """
        x = performance_data['in']
        x = sm.add_constant(x)

        for car in performance_data['out']:
            self.coefficients[car] = {}
            y = performance_data['out'][car]
            fit = fit_linear_function(x, y)
            self.coefficients[car]['alpha1'] = sig_figs(fit[1], 6)
            self.coefficients[car]['alpha2'] = sig_figs(fit[0], 6)

    def calculate_input_bounds(self, size_based_on, time_steps):
        """
        Calculates input bounds for type 1 generic technologies

        :param str size_based_on: 'input' or 'output'
        :return: input bounds
        """
        if size_based_on == 'input':
            for car in self.input_carrier:
                self.bounds['input'][car] = np.column_stack((np.zeros(shape=time_steps),
                                                             np.ones(shape=time_steps)))
        elif size_based_on == 'output':
            for car in self.input_carrier:
                if car in self.coefficients:
                    car_aux = car
                else:
                    car_aux = 'out'
                self.bounds['input'][car] = (np.column_stack((np.zeros(shape=time_steps),
                                                              np.ones(shape=time_steps))) -
                                             self.coefficients[car_aux]['alpha2']) / self.coefficients[car_aux][
                                                'alpha1']
        else:
            raise Exception('size_based_on must be either input or output')

    def calculate_output_bounds(self, size_based_on, time_steps):
        """
        Calculates output bounds for type 1 generic technologies

        :param str size_based_on: 'input' or 'output'
        :return: output bounds
        """
        if size_based_on == 'input':
            for car in self.output_carrier:
                if car in self.coefficients:
                    car_aux = car
                else:
                    car_aux = 'out'
                self.bounds['output'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                              np.ones(shape=(time_steps)) * self.coefficients[car_aux][
                                                                  'alpha1'] + \
                                                              self.coefficients[car_aux]['alpha2']))
        elif size_based_on == 'output':
            for car in self.output_carrier:
                self.bounds['output'][car] = np.column_stack((np.zeros(shape=time_steps),
                                                              np.ones(shape=time_steps)))
        else:
            raise Exception('size_based_on must be either input or output')


class FitGenericTecTypeType3(FittedPerformance):
    """
    Subclass to fit performance of type3 performance functions (piecewise linear, with min partload)
    out = alpha1[i] * in + alpha2
    """

    def fit_performance_function(self, performance_data):
        """
        Fits performance function for input-output data for type 2 technologies

        :param performance_data: performance data
        :return: coefficients
        """
        if 'nr_segments_piecewise' in performance_data:
            nr_seg = performance_data['nr_segments_piecewise']
        else:
            nr_seg = 2
        x = performance_data['in']
        y = performance_data['out']
        self.coefficients = fit_piecewise_function(x, y, nr_seg)

    def calculate_input_bounds(self, size_based_on, time_steps):
        """
        Calculates input bounds for type 1 generic technologies

        :param str size_based_on: 'input' or 'output'
        :return: input bounds
        """
        if size_based_on == 'input':
            for car in self.input_carrier:
                self.bounds['input'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                             np.ones(shape=(time_steps))))
        elif size_based_on == 'output':
            for car in self.input_carrier:
                if car in self.coefficients:
                    car_aux = car
                else:
                    car_aux = 'out'
                self.bounds['input'][car] = (np.column_stack((np.zeros(shape=time_steps),
                                                              np.ones(shape=time_steps))) -
                                             self.coefficients[car_aux]['alpha2'][-1]) / \
                                            self.coefficients[car_aux]['alpha1'][-1]
        else:
            raise Exception('size_based_on must be either input or output')

    def calculate_output_bounds(self, size_based_on, time_steps):
        """
        Calculates output bounds for type 1 generic technologies

        :param str size_based_on: 'input' or 'output'
        :return: output bounds
        """
        if size_based_on == 'input':
            for car in self.output_carrier:
                if car in self.coefficients:
                    car_aux = car
                else:
                    car_aux = 'out'
                self.bounds['output'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                              np.ones(shape=(time_steps)) *
                                                              self.coefficients[car_aux]['alpha1'][-1] + \
                                                              self.coefficients[car_aux]['alpha2'][-1]))
        elif size_based_on == 'output':
            for car in self.output_carrier:
                self.bounds['output'][car] = np.column_stack((np.zeros(shape=(time_steps)),
                                                              np.ones(shape=(time_steps))))
        else:
            raise Exception('size_based_on must be either input or output')


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

    for idx, car in enumerate(Y):
        fit[car] = {}
        y = np.array(Y[car])
        if idx == 0:
            bp_x, bp_y, alpha1, alpha2 = regress_piecewise(X, y, nr_segments)
            bp_x0 = bp_x
        else:
            bp_x, bp_y, alpha1, alpha2 = regress_piecewise(X, y, nr_segments, bp_x0)

        fit[car]['alpha1'] = [sig_figs(float(num), 4) for num in alpha1]
        fit[car]['alpha2'] = [sig_figs(float(num), 4) for num in alpha2]
        fit[car]['bp_y'] = [sig_figs(float(num), 4) for num in bp_y]
        fit[car]['bp_x'] = [sig_figs(float(num), 4) for num in bp_x]

    return fit


def sig_figs(x: float, precision: int):
    """
    Rounds a number to number of significant figures
    Parameters:
    - x - the number to be sig_figsed
    - precision (integer) - the number of significant figures
    Returns:
    - float
    """

    x = float(x)
    precision = int(precision)

    if x == 0:
        rounded = 0
    else:
        rounded = round(x, -int(floor(log10(abs(x)))) + (precision - 1))

    return rounded