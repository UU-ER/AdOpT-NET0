import statsmodels.api as sm
import numpy as np

from ..utilities import (
    fit_linear_function,
    fit_piecewise_function,
    sig_figs,
)


class FitGenericTecTypeType1:
    """
    Class to fit performance of type1 performance functions (linear, through origin)
    out = alpha1 * in
    """

    def __init__(self, params):
        self.input_parameters = params
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
            for car in self.input_parameters.input_carrier:
                input_bounds[car] = np.column_stack(
                    (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
                )
        elif size_based_on == "output":
            for car in self.input_parameters.input_carrier:
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
            for car in self.input_parameters.output_carrier:
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
            for car in self.input_parameters.output_carrier:
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

    def __init__(self, params):
        self.input_parameters = params
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
        Calculates input bounds for type 1 generic technologies

        :param str size_based_on: 'input' or 'output'
        :param int time_steps: number of time steps
        """
        input_bounds = {}

        if size_based_on == "input":
            for car in self.input_parameters.input_carrier:
                input_bounds[car] = np.column_stack(
                    (np.zeros(shape=time_steps), np.ones(shape=time_steps))
                )
        elif size_based_on == "output":
            for car in self.input_parameters.input_carrier:
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
            for car in self.input_parameters.output_carrier:
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
            for car in self.input_parameters.output_carrier:
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

    def __init__(self, params):
        self.input_parameters = params
        self.coeff = {}
        self.bounds = {}

    def fit_performance_function(self, performance_data: dict):
        """
        Fits performance function for input-output data for type 2 technologies

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
            for car in self.input_parameters.input_carrier:
                input_bounds[car] = np.column_stack(
                    (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
                )
        elif size_based_on == "output":
            for car in self.input_parameters.input_carrier:
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
            for car in self.input_parameters.output_carrier:
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
            for car in self.input_parameters.output_carrier:
                output_bounds[car] = np.column_stack(
                    (np.zeros(shape=(time_steps)), np.ones(shape=(time_steps)))
                )
        else:
            raise Exception("size_based_on must be either input or output")

        return output_bounds
