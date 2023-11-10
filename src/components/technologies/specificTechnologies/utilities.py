from scipy import interpolate
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from src.components.technologies.utilities import fit_piecewise_function

def fit_turbomachinery(machinery_data):

    performance_data = {}

    # Parameters needed for pump performance calculations
    nominal_head = machinery_data['nominal_head']
    frequency = machinery_data['frequency']
    pole_pairs = machinery_data['pole_pairs']
    N = (120 * frequency) / (pole_pairs * 2)
    omega = 2 * np.pi * N / 60
    nr_segments_design = machinery_data['nr_segments_design']
    nr_segments_performance = machinery_data['nr_segments_performance']

    # TODO formulate constraints on diameter
    # obtain performance curves (omega s, Ds) and (omega s, efficiency)
    design = pd.read_csv(machinery_data['balje_data_path'], delimiter=';')
    design_efficiency = pd.read_csv(machinery_data['design_data_path'], delimiter=';')

    # TODO: decide which pump curve is used - Francis reversible = centrifugal?
    # remove values that are outside of the chosen pumps operating range
    design = design[(design['Specific_rotational_speed'] >= machinery_data['omega_s_min']) &
                    (design['Specific_rotational_speed'] <= machinery_data['omega_s_max'])]
    # calculate design flow from optimum Ds, omega-s curve
    design['Q_design'] = design.apply(lambda row: ((row['Specific_rotational_speed']
                                                    * ((9.81 * nominal_head) ** 0.75))
                                                   / omega) ** 2, axis=1)
    # calculate diameter at this design flow
    design['D'] = design.apply(lambda row: ((row['D_s'] * (row['Q_design'] ** 0.5))
                                            / ((9.81 * nominal_head) ** 0.25)), axis=1)

    # obtain efficiency at these design specific rotational speed
    design_efficiency_interpl = interpolate.interp1d(design_efficiency['Specific_rotational_speed'],
                                                     design_efficiency['Eta_design'], kind='linear',
                                                     fill_value='extrapolate')
    design['Eta_design'] = design_efficiency_interpl(design['Specific_rotational_speed'])

    # calculate the design power output that is obtained with the design flow at design efficiency
    if machinery_data['type'] == 'pump':
        design['P_design'] = design['Q_design'] * 1000 * 9.81 * nominal_head * (10 ** -6) * design['Eta_design']
    elif machinery_data['type'] == 'turbine':
        design['P_design'] = design['Q_design'] * 1000 * 9.81 * nominal_head * (10 ** -6) * design['Eta_design']

    # Interpolate values for the equally spaced points using linear interpolation
    x_values = np.linspace(design['Q_design'].min(), design['Q_design'].max(), 20)
    interpolated_y = interp1d(design['Q_design'], design['P_design'], kind='linear')(x_values)

    # get performance data for that pump
    y = {}
    y['design'] = interpolated_y
    # fitting data
    fit_design = fit_piecewise_function(x_values, y, nr_segments_design)
    performance_data['design'] = fit_design['design']

    efficiency_partload = pd.read_csv(machinery_data['partload_data_path'])
    efficiency_partload['Eta'] = efficiency_partload['Eta'] / 100
    efficiency_partload['Q'] = efficiency_partload['Q'] / 100

    # scale down efficiency to match values from design
    delta_max_eff = max(efficiency_partload['Eta']) - max(design['Eta_design'])
    efficiency_partload['Eta'] = efficiency_partload['Eta'] - delta_max_eff

    if machinery_data['type'] == 'pump':
        efficiency_partload['P'] = efficiency_partload['Q'] * 1000 * 9.81 * nominal_head * (10 ** -6) * \
                                   efficiency_partload['Eta']
    elif machinery_data['type'] == 'turbine':
        efficiency_partload['P'] = efficiency_partload['Q'] * 1000 * 9.81 * nominal_head * (10 ** -6) * \
                                   efficiency_partload['Eta']

    x = efficiency_partload['Q'].values
    y = {}
    y['performance'] = efficiency_partload['P'].values
    fit_performance = fit_piecewise_function(x, y, nr_segments_performance)

    performance_data['performance'] = fit_performance['performance']

    # Bounds
    performance_data['bounds'] = {}
    performance_data['bounds']['Q_ub'] = max(fit_performance['performance']['bp_x']) * max(fit_design['design']['bp_x'])
    performance_data['bounds']['P_ub'] = fit_performance['performance']['alpha2'][-1] * max(fit_design['design']['bp_y']) + \
                                   fit_performance['performance']['alpha1'][-1] * performance_data['bounds']['Q_ub']

    return performance_data