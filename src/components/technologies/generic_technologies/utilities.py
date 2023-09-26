import statsmodels.api as sm
import numpy as np

from src.components.technologies.utilities import FittedPerformance, fit_linear_function, fit_piecewise_function, sig_figs


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
    else:
        fitting.rated_power = 1
    fitting.time_dependent_coefficients = 0
    return fitting


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


