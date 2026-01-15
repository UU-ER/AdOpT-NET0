import json
import os
import pwlf
import numpy as np
from math import floor, log10
from statsmodels import api as sm
from pathlib import Path


def open_json(tec: str, load_path: str | Path) -> dict:
    """
    Reads technology data from json file

    :param str tec: Technology name
    :param str, Path load_path: directory to look for json in
    :return: dict with technology data from json
    :rtype: dict
    """
    # Read in JSON files
    for path, subdirs, files in os.walk(load_path):
        if "technology_data" in locals():
            break
        else:
            for name in files:
                if (tec + ".json") == name:
                    filepath = os.path.join(path, name)
                    with open(filepath) as json_file:
                        technology_data = json.load(json_file)
                    break

    # Assign name
    if "technology_data" in locals():
        technology_data["name"] = tec
    else:
        raise Exception("There is no json data file for technology " + tec)

    return technology_data


def fit_linear_function(x: np.array, y: np.array) -> np.array:
    """
    Fits linear model to x and y data and returns coefficients

    :param np.array x: x data
    :param np.array y: y data
    :return: coefficients of OLS regression
    :rtype: np.array
    """
    linmodel = sm.OLS(y, x)
    linfit = linmodel.fit()
    coeff = linfit.params
    return coeff


def fit_piecewise_function(X: np.array, Y: np.array, nr_segments: int) -> dict:
    """
    Returns fitted parameters of a piecewise defined function with multiple y-series

    :param np.array X: x-values of data
    :param np.array Y: y-values of data
    :param int nr_seg: number of segments on piecewise defined function
    :return: x and y breakpoints, slope and intercept parameters of piecewise defined function
    :rtype: dict
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
            al2 = (
                bp_y[seg]
                - (bp_y[seg + 1] - bp_y[seg]) / (bp_x[seg + 1] - bp_x[seg]) * bp_x[seg]
            )  # Intercept
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

        fit[car]["alpha1"] = [sig_figs(float(num), 4) for num in alpha1]
        fit[car]["alpha2"] = [sig_figs(float(num), 4) for num in alpha2]
        fit[car]["bp_y"] = [sig_figs(float(num), 4) for num in bp_y]
        fit[car]["bp_x"] = [sig_figs(float(num), 4) for num in bp_x]

    return fit


def sig_figs(x: float, precision: int):
    """
    Rounds a number to number of significant figures

    :param float x: number to round
    :param int precision: rounding precision
    :return: rounded number
    :rtype: float
    """

    x = float(x)
    precision = int(precision)

    if x == 0:
        rounded = 0
    else:
        rounded = round(x, -int(floor(log10(abs(x)))) + (precision - 1))

    return rounded


class FitGenericTecTypeType1:
    """
    Class to fit performance of type1 performance functions (linear, through origin)
    out = alpha1 * in
    """

    def __init__(self, input_carriers, output_carriers):
        self.input_carriers = input_carriers
        self.output_carriers = output_carriers
        self.coeff = {}
        self.bounds = {}

    def fit_performance_function(self, performance_data: dict):
        """
        Fits performance function for input-output data for type 1 technologies

        :param dict performance_data: performance data
        """
        x = performance_data["in"]

        for car in performance_data["out"]:
            self.coeff[car] = {}
            y = performance_data["out"][car]
            fit = fit_linear_function(x, y)
            self.coeff[car]["alpha1"] = sig_figs(fit[0], 6)

        return self.coeff

    def calculate_input_bounds(self, size_based_on: str, time_steps: int):
        """
        Calculates input bounds for type 1 generic technologies

        :param str size_based_on: 'input' or 'output'
        :param int time_steps: number of time steps
        """
        input_bounds = {}

        if size_based_on == "input":
            for car in self.input_carriers:
                input_bounds[car] = np.column_stack(
                    (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
                )
        elif size_based_on == "output":
            for car in self.input_carriers:
                if car in self.coeff:
                    car_aux = car
                else:
                    car_aux = "out"
                input_bounds[car] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps)) / self.coeff[car_aux]["alpha1"],
                    )
                )
        else:
            raise Exception("size_based_on must be either input or output")

        return input_bounds

    def calculate_output_bounds(self, size_based_on: str, time_steps: int):
        """
        Calculates output bounds for type 1 generic technologies

        :param str size_based_on: 'input' or 'output'
        :param int time_steps: number of time steps
        """
        output_bounds = {}

        if size_based_on == "input":
            for car in self.output_carriers:
                if car in self.coeff:
                    car_aux = car
                else:
                    car_aux = "out"
                output_bounds[car] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps)) * self.coeff[car_aux]["alpha1"],
                    )
                )
        elif size_based_on == "output":
            for car in self.output_carriers:
                output_bounds[car] = np.column_stack(
                    (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
                )
        else:
            raise Exception("size_based_on must be either input or output")

        return output_bounds


class FitGenericTecTypeType2:
    """
    Class to fit performance of type1 performance functions (linear, with min partload)
    out = alpha1 * in + alpha2
    (out - alpha2)/alpha1
    """

    def __init__(self, input_carriers, output_carriers):
        self.input_carriers = input_carriers
        self.output_carriers = output_carriers
        self.coeff = {}
        self.bounds = {}

    def fit_performance_function(self, performance_data: dict):
        """
        Fits performance function for input-output data for type 2 technologies

        :param dict performance_data: performance data
        """
        x = performance_data["in"]
        x = sm.add_constant(x)

        for car in performance_data["out"]:
            self.coeff[car] = {}
            y = performance_data["out"][car]
            fit = fit_linear_function(x, y)
            self.coeff[car]["alpha1"] = sig_figs(fit[1], 6)
            self.coeff[car]["alpha2"] = sig_figs(fit[0], 6)

        return self.coeff

    def calculate_input_bounds(self, size_based_on: str, time_steps: int):
        """
        Calculates input bounds for type 2 generic technologies

        :param str size_based_on: 'input' or 'output'
        :param int time_steps: number of time steps
        """
        input_bounds = {}

        if size_based_on == "input":
            for car in self.input_carriers:
                input_bounds[car] = np.column_stack(
                    (np.zeros(shape=time_steps), np.ones(shape=time_steps))
                )
        elif size_based_on == "output":
            for car in self.input_carriers:
                if car in self.coeff:
                    car_aux = car
                else:
                    car_aux = "out"
                input_bounds[car] = (
                    np.column_stack(
                        (np.zeros(shape=time_steps), np.ones(shape=time_steps))
                    )
                    - self.coeff[car_aux]["alpha2"]
                ) / self.coeff[car_aux]["alpha1"]
        else:
            raise Exception("size_based_on must be either input or output")

        return input_bounds

    def calculate_output_bounds(self, size_based_on: str, time_steps: int):
        """
        Calculates output bounds for type 2 generic technologies

        :param str size_based_on: 'input' or 'output'
        :param int time_steps: number of time steps
        """
        output_bounds = {}

        if size_based_on == "input":
            for car in self.output_carriers:
                if car in self.coeff:
                    car_aux = car
                else:
                    car_aux = "out"
                output_bounds[car] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps)) * self.coeff[car_aux]["alpha1"]
                        + self.coeff[car_aux]["alpha2"],
                    )
                )
        elif size_based_on == "output":
            for car in self.output_carriers:
                output_bounds[car] = np.column_stack(
                    (np.zeros(shape=time_steps), np.ones(shape=time_steps))
                )
        else:
            raise Exception("size_based_on must be either input or output")

        return output_bounds


class FitGenericTecTypeType34:
    """
    Class to fit performance of type3 performance functions (piecewise linear, with min partload)
    out = alpha1[i] * in + alpha2
    """

    def __init__(self, input_carriers, output_carriers):
        self.input_carriers = input_carriers
        self.output_carriers = output_carriers
        self.coeff = {}
        self.bounds = {}

    def fit_performance_function(self, performance_data: dict):
        """
        Fits performance function for input-output data for type 3/4 technologies

        :param dict performance_data: performance data
        """
        if "nr_segments_piecewise" in performance_data:
            nr_seg = performance_data["nr_segments_piecewise"]
        else:
            nr_seg = 2
        x = performance_data["in"]
        y = performance_data["out"]
        self.coeff = fit_piecewise_function(x, y, nr_seg)

        return self.coeff

    def calculate_input_bounds(self, size_based_on: str, time_steps: int):
        """
        Calculates input bounds for type 3/4 generic technologies

        :param str size_based_on: 'input' or 'output'
        :param int time_steps: number of time steps
        """
        input_bounds = {}

        if size_based_on == "input":
            for car in self.input_carriers:
                input_bounds[car] = np.column_stack(
                    (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
                )
        elif size_based_on == "output":
            for car in self.input_carriers:
                if car in self.coeff:
                    car_aux = car
                else:
                    car_aux = "out"
                input_bounds[car] = (
                    np.column_stack(
                        (np.zeros(shape=time_steps), np.ones(shape=time_steps))
                    )
                    - self.coeff[car_aux]["alpha2"][-1]
                ) / self.coeff[car_aux]["alpha1"][-1]
        else:
            raise Exception("size_based_on must be either input or output")

        return input_bounds

    def calculate_output_bounds(self, size_based_on: str, time_steps: int):
        """
        Calculates output bounds for type 3/4 generic technologies

        :param str size_based_on: 'input' or 'output'
        :param int time_steps: number of time steps
        """
        output_bounds = {}

        if size_based_on == "input":
            for car in self.output_carriers:
                if car in self.coeff:
                    car_aux = car
                else:
                    car_aux = "out"
                output_bounds[car] = np.column_stack(
                    (
                        np.zeros(shape=(time_steps)),
                        np.ones(shape=(time_steps)) * self.coeff[car_aux]["alpha1"][-1]
                        + self.coeff[car_aux]["alpha2"][-1],
                    )
                )
        elif size_based_on == "output":
            for car in self.output_carriers:
                output_bounds[car] = np.column_stack(
                    (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
                )
        else:
            raise Exception("size_based_on must be either input or output")

        return output_bounds
