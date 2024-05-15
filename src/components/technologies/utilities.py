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
        technology_data["Name"] = tec
    else:
        raise Exception("There is no json data file for technology " + tec)

    return technology_data


def set_capex_model(config: dict, economics) -> int:
    """
    Sets the capex model of a technology

    Takes either the global capex model or the model defined in respective technology
    :param dict config: dict containing model information
    :param economics: Economics class
    :return: CAPEX model
    :rtype: int
    """
    if config["economic"]["global_simple_capex_model"]["value"]:
        capex_model = 1
    else:
        capex_model = economics.capex_model
    return capex_model


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
