import numpy as np
from pyomo.gdp import *
import time
import src.config_model as m_config
from pyomo.environ import *
import numpy as np


def perform_disjunct_relaxation(component):
    print('Big-M Transformation...')
    start = time.time()
    xfrm = TransformationFactory('gdp.bigm')
    xfrm.apply_to(component)
    m_config.presolve.big_m_transformation_required = 0
    print('Big-M Transformation completed in ' + str(time.time() - start) + ' s')
    return component

def annualize(r, t):
    if r==0:
        annualization_factor = 1/t
    else:
        annualization_factor = r / (1 - (1 / (1 + r) ** t))
    return annualization_factor


def calculate_output_bounds(tec_data):
    """
    Calculates bounds for technology outputs for each input carrier
    """
    technology_model = tec_data.technology_model
    size_is_int = tec_data.size_is_int
    size_max = tec_data.size_max
    performance_data = tec_data.performance_data
    fitted_performance = tec_data.fitted_performance
    if size_is_int:
        rated_power = fitted_performance['rated_power']
    else:
        rated_power = 1

    bounds = {}

    if technology_model == 'RES':  # Renewable technology with cap_factor as input
        cap_factor = fitted_performance['capacity_factor']
        for c in performance_data['output_carrier']:
            max_bound = float(size_max * max(cap_factor) * rated_power)
            bounds[c] = (0, max_bound)

    elif technology_model == 'CONV1' or technology_model.startswith('HeatPump_'):  # n inputs -> n output, fuel and output substitution
        performance_function_type = performance_data['performance_function_type']
        alpha1 = fitted_performance['out']['alpha1']
        if technology_model.startswith('HeatPump_'):
            alpha1 = np.amax(alpha1, axis=0)[-1]
        for c in performance_data['output_carrier']:
            if performance_function_type == 1:
                max_bound = size_max * alpha1 * rated_power
            if performance_function_type == 2:
                alpha2 = fitted_performance['out']['alpha2']
                if technology_model.startswith('HeatPump_'):
                    alpha2 = np.amax(alpha2, axis=0)[-1]
                max_bound = size_max * (alpha1 + alpha2) * rated_power
            if performance_function_type == 3:
                alpha2 = fitted_performance['out']['alpha2']
                if technology_model.startswith('HeatPump_'):
                    alpha2 = np.amax(alpha2, axis=0)
                max_bound = size_max * (alpha1[-1] + alpha2[-1]) * rated_power
            bounds[c] = (0, max_bound)

    elif technology_model == 'CONV2':  # n inputs -> n output, fuel and output substitution
        alpha1 = {}
        alpha2 = {}
        performance_function_type = performance_data['performance_function_type']
        for c in performance_data['performance']['out']:
            alpha1[c] = fitted_performance[c]['alpha1']
            if performance_function_type == 1:
                max_bound = alpha1[c] * size_max * rated_power
            if performance_function_type == 2:
                alpha2[c] = fitted_performance[c]['alpha2']
                max_bound = size_max * (alpha1[c] + alpha2[c]) * rated_power
            if performance_function_type == 3:
                alpha2[c] = fitted_performance[c]['alpha2']
                max_bound = size_max * (alpha1[c][-1] + alpha2[c][-1]) * rated_power
            bounds[c] = (0, max_bound)

    elif technology_model == 'CONV3':  # 1 input -> n outputs, output flexible, linear performance
        alpha1 = {}
        alpha2 = {}
        performance_function_type = performance_data['performance_function_type']
        # Get performance parameters
        for c in performance_data['performance']['out']:
            alpha1[c] = fitted_performance[c]['alpha1']
            if performance_function_type == 1:
                max_bound = alpha1[c] * size_max * rated_power
            if performance_function_type == 2:
                alpha2[c] = fitted_performance[c]['alpha2']
                max_bound = size_max * (alpha1[c] + alpha2[c]) * rated_power
            if performance_function_type == 3:
                alpha2[c] = fitted_performance[c]['alpha2']
                max_bound = size_max * (alpha1[c][-1] + alpha2[c][-1]) * rated_power
            bounds[c] = (0, max_bound)

    elif technology_model == 'STOR':  # Storage technology (1 input -> 1 output)
        for c in performance_data['output_carrier']:
            bounds[c] = (0, size_max)

    elif technology_model == 'DAC_adsorption':
        bounds['CO2'] = (0, size_max * max(fitted_performance['out_max']))

    return bounds


def calculate_input_bounds(tec_data):
    """
    Calculates bounds for technology inputs for each input carrier
    """
    technology_model = tec_data.technology_model
    size_max = tec_data.size_max
    performance_data = tec_data.performance_data
    fitted_performance = tec_data.fitted_performance

    bounds = {}
    if technology_model == 'CONV3':
        main_car = performance_data['main_input_carrier']
        for c in performance_data['input_carrier']:
            if c == main_car:
                bounds[c] = (0, size_max)
            else:
                bounds[c] = (0, size_max * performance_data['input_ratios'][c])
    elif technology_model == 'DAC_adsorption':
        bounds['electricity'] = (0, size_max * (max(fitted_performance['el_in_max']) +
                                                max(fitted_performance['th_in_max'])) /
                                 performance_data['performance']['eta_elth'])
        bounds['heat'] = (0, size_max * max(fitted_performance['th_in_max']))
    else:
        for c in performance_data['input_carrier']:
            bounds[c] = (0, size_max)
    return bounds
