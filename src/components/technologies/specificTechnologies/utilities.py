from scipy import interpolate
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from pathlib import Path

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

    # Path read data from
    data_path = 'data/ob_input_data/' + machinery_data['type'] + '_' + machinery_data['subtype'] + '/'

    # TODO formulate constraints on diameter
    # obtain performance curves (omega s, Ds) and (omega s, efficiency)
    design = pd.read_csv(Path(data_path + 'balje_diagram.csv'))
    design_efficiency = pd.read_csv(Path(data_path + 'efficiency.csv'))

    # Extrapolate design_efficiency
    coefficients = np.polyfit(design_efficiency['Specific_rotational_speed'], design_efficiency['Eta_design'], 2)
    poly_fit = np.poly1d(coefficients)

    avg_diff = design_efficiency['Specific_rotational_speed'].diff().dropna().mean()
    new_omega_s = np.linspace(machinery_data['omega_s_min'], machinery_data['omega_s_max'], 20)
    new_eta_design = poly_fit(new_omega_s)
    design_efficiency = pd.DataFrame({'Specific_rotational_speed':  list(new_omega_s),
                'Eta_design': list(new_eta_design)})

    # remove values that are outside of the chosen pumps operating range
    design = design[(design['Specific_rotational_speed'] >= machinery_data['omega_s_min']) &
                    (design['Specific_rotational_speed'] <= machinery_data['omega_s_max'])]
    # calculate design flow from optimum Ds, omega-s curve
    design['Q_design'] = design.apply(lambda row: (((row['Specific_rotational_speed']
                                                    * ((9.81 * nominal_head) ** 0.75))
                                                   / omega) ** 2), axis=1)
    # calculate diameter at this design flow
    design['D'] = design.apply(lambda row: ((row['D_s'] * ((row['Q_design']) ** 0.5))
                                            / ((9.81 * nominal_head) ** 0.25)), axis=1)

    design_efficiency_interpl = interpolate.interp1d(design_efficiency['Specific_rotational_speed'],
                                                     design_efficiency['Eta_design'], kind='linear')
    design['Eta_design'] = design_efficiency_interpl(design['Specific_rotational_speed'])

    # calculate the design power output that is obtained with the design flow at design efficiency
    if machinery_data['type'] == 'pump':
        design['P_design'] = design['Q_design'] * 1000 * 9.81 * nominal_head * (10 ** -6) / design['Eta_design']
    elif machinery_data['type'] == 'turbine':
        design['P_design'] = design['Q_design'] * 1000 * 9.81 * nominal_head * (10 ** -6) * design['Eta_design']

    design = design[(design['P_design'] >= machinery_data['min_power'])]

    conversion_factor_flow_rate = 3600 # from m³/s to m³/h
    design['Q_design'] = design['Q_design'] * conversion_factor_flow_rate

    # Interpolate values for the equally spaced points using linear interpolation
    x_values = np.linspace(design['Q_design'].min(), design['Q_design'].max(), 5)
    interpolated_y = interp1d(design['Q_design'], design['P_design'], kind='linear')(x_values)

    # get performance data for that pump
    y = {}
    y['design'] = interpolated_y
    # fitting data
    fit_design = fit_piecewise_function(x_values, y, nr_segments_design)
    performance_data['design'] = fit_design['design']

    efficiency_partload = pd.read_csv(Path(data_path + 'part_load.csv'))
    efficiency_partload['Eta'] = efficiency_partload['Eta'] / 100
    efficiency_partload['Q'] = efficiency_partload['Q'] / 100

    # scale down efficiency to match values from design
    delta_max_eff = max(efficiency_partload['Eta']) - max(design['Eta_design'])
    efficiency_partload['Eta'] = efficiency_partload['Eta'] - delta_max_eff

    if machinery_data['type'] == 'pump':
        efficiency_partload['P'] = efficiency_partload['Q'] * 1000 * 9.81 * nominal_head * (10 ** -6) / \
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

def fit_turbomachinery_capex(machinery_data):

    # capex constants & calculation from AlZohbi (2018)
    capex_data = {}
    nominal_head = machinery_data['nominal_head']
    nr_segments_capex = machinery_data['nr_segments_capex']
    capex_constant_a = machinery_data['capex_constant_a']
    capex_constant_b = machinery_data['capex_constant_b']
    capex_constant_c = machinery_data['capex_constant_c']
    inflation_correction = 1.2692 # from 2018-2023 EUR

    # for pump P in kW, for turbine P in MW: basevalue is for 1 MW.
    capex_basevalue = (capex_constant_a * (1 ** capex_constant_b) * (nominal_head ** capex_constant_c) *
                       inflation_correction)

    # scaling factor: using capex calculation by Aggidis et al. (2010) - equation 13
    capex_constant_a_scaling = 12000
    capex_constant_b_scaling = 0.2
    capex_constant_c_scaling = 0.56

    size_values = np.arange(0.1, 10.1, 0.1) # is P in MW
    df_size_scaling = pd.DataFrame({'size_P': size_values})

    # P in kW: basevalue is for 1 MW
    capex_basevalue_scaling = (capex_constant_a_scaling * ((1000/(nominal_head ** capex_constant_b_scaling)) ** capex_constant_c_scaling))
    df_size_scaling['scaling_factor'] = ((capex_constant_a_scaling * (((df_size_scaling['size_P'].values * 1000)/
                                                                       (nominal_head ** capex_constant_b_scaling))
                                                         ** capex_constant_c_scaling))) / capex_basevalue_scaling

    df_size_scaling['capex'] = df_size_scaling['scaling_factor'].values * capex_basevalue

    x = df_size_scaling['size_P'].values
    y = {}
    y['capex'] = df_size_scaling['capex'].values
    fit_capex = fit_piecewise_function(x, y, nr_segments_capex)

    capex_data['capex'] = fit_capex['capex']

    return capex_data
