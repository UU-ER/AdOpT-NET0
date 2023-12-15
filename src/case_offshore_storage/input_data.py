import pandas as pd
from pathlib import Path
import numpy as np

class InputDataConfig:
    def __init__(self, test=0):
        # Prices
        self.price_ng = 43.92
        self.price_co2 = 110

        if test == 1:
            # timescale
            self.start_date = '01-01 00:00'
            self.end_date = '02-01 00:00'
            # Self sufficiency range
            self.f_self_sufficiency = [2]
            self.f_offshore_share = [0.5]
            # Emission reduction goals
            self.f_emission_reduction = [0.95]
        else:
            # timescale
            self.start_date = '01-01 00:00'
            self.end_date = '12-31 23:00'
            # Self sufficiency range
            self.f_self_sufficiency = np.arange(0.5, 2.5, 0.25).tolist()
            self.f_offshore_share = np.arange(0.2, 1.2, 0.2).tolist()
            # Emission reduction goals
            self.f_emission_reduction = np.arange(0, 1.1, 0.1).tolist()

        # Scaling of demand
        self.f_demand_scaling = 0.05

        # Reporting
        if test == 1:
            self.save_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/StorageOffshoreTest/'
        else:
            self.save_path = '//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/StorageOffshore/'

        # Solver settings
        self.mipgap = 0


def determine_time_series(f_demand, f_offshore, f_self_sufficiency):
    time_series = pd.read_csv(Path('./cases/storage/clean_data/time_series.csv'))
    demand = time_series['demand'] * f_demand
    annual_demand = sum(demand)

    s_pv = 194522 / (100661 + 194522)
    s_wind = 100661 / (100661 + 194522)

    e_offshore = sum(time_series['wind_offshore'])
    e_onshore = sum(time_series['wind_onshore']) * s_wind + sum(time_series['PV']) * s_pv

    # capacity required for 1MWh annual generation onshore/offshore
    c_offshore = 1 / e_offshore * annual_demand * f_offshore * f_self_sufficiency
    c_onshore = 1 / e_onshore * annual_demand * (1 - f_offshore) * f_self_sufficiency

    # generation profiles
    p_offshore = c_offshore * time_series['wind_offshore']
    p_onshore = c_onshore * (time_series['wind_onshore'] * s_wind + time_series['PV'] * s_pv)

    return demand, p_onshore, p_offshore
