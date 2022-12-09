import statsmodels.api as sm
import numpy as np
from scipy import optimize
import pvlib
import datetime
import pytz
from timezonefinder import TimezoneFinder
import pandas as pd
from scipy.interpolate import interp1d


def fit_netw_performance(network, climate_data=None):
    """
    Fits the performance parameters for a network, i.e. the consumption at each node.
    :param obj network: Dict read from json files with performance data and options for performance fits
    :param obj climate_data: Climate data
    :return: dict of performance coefficients used in the model
    """
    # Initialize parameters dict
    parameters = dict()

    # Get energy consumption at nodes form file
    energycons = network['NetworkPerf']['energyconsumption']
    network['NetworkPerf'].pop('energyconsumption')

    if energycons:
        EnergyConsumptionFit = {}
        for car in energycons:
            ec = energycons[car]
            EnergyConsumptionFit[car] = {}
            if energycons[car]['cons_model'] == 1:
                EnergyConsumptionFit[car]['send'] = {}
                EnergyConsumptionFit[car]['send'] = energycons[car]
                EnergyConsumptionFit[car]['send'].pop('cons_model')
                # TODO: OptionaL implmementation: receiving energy consumption
                EnergyConsumptionFit[car]['receive'] = {}
                EnergyConsumptionFit[car]['receive']['k_flow'] = 0
                EnergyConsumptionFit[car]['receive']['k_flowDistance'] = 0
            elif energycons[car]['cons_model'] == 2:
                temp = energycons[car]
                EnergyConsumptionFit[car]['send'] = {}
                EnergyConsumptionFit[car]['send']['k_flow'] = round(temp['c'] * temp['T'] / temp['eta'] / \
                                                                    temp['LHV'] * ((temp['p'] / 30) ** \
                                                                   ((temp['gam'] - 1) / temp['gam']) - 1),4)
                EnergyConsumptionFit[car]['send']['k_flowDistance'] = 0
                # TODO: OptionaL implmementation: receiving energy consumption
                EnergyConsumptionFit[car]['receive'] = {}
                EnergyConsumptionFit[car]['receive']['k_flow'] = 0
                EnergyConsumptionFit[car]['receive']['k_flowDistance'] = 0

        parameters['EnergyConsumption'] = EnergyConsumptionFit
    else:
        parameters['EnergyConsumption'] = {}

    parameters['NetworkPerf'] = network['NetworkPerf']
    parameters['Economics'] = network['Economics']
    parameters['connection'] = network['connection']
    parameters['distance'] = network['distance']
    return parameters
