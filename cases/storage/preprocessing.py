import pandas as pd
import numpy as np
from pathlib import Path


def read_eraa_data(read_path, climate_year, bidding_zone):
    """
    Reads capacity factors for offshore wind for a bidding_zone from ERAA data file
    :param read_path:
    :param climate_year:
    :param country:
    :return:
    """
    capacity_factor = pd.read_excel(read_path, sheet_name=bidding_zone, skiprows=range(1, 11))
    capacity_factor = capacity_factor.iloc[:, 1:]
    capacity_factor.set_index(capacity_factor.columns[0], inplace=True)
    capacity_factor.columns = range(1982, 1982 + len(capacity_factor.columns))
    capacity_factor = capacity_factor[climate_year]
    return capacity_factor

climate_year = 2009
bidding_zone = 'UK00'
save_path = Path('./cases/storage/clean_data/time_series.csv')

data = pd.DataFrame()

# Offshore wind capacity factors
read_path = Path('E:/00_Data/00_RenewableGeneration/ENTSOE_ERAA/Wind offshore/PECD_Wind_Offshore_2030_edition 2022.1.xlsx')
data['wind'] = read_eraa_data(read_path, climate_year, bidding_zone)

# PV capacity factors
read_path = Path('E:/00_Data/00_RenewableGeneration/ENTSOE_ERAA/Solar/PECD_LFSolarPV_2030_edition 2022.1.xlsx')
data['PV'] = read_eraa_data(read_path, climate_year, bidding_zone)

# demand
read_path = Path('E:/00_Data/00_EnergyDemandEurope/Electricity/ENTSOE_ERAA/2022/Demand data/Demand Time Series/Demand_TimeSeries_2030_NationalTrends_without_bat.xlsx')
data['demand'] = read_eraa_data(read_path, climate_year, bidding_zone)

data.to_csv(save_path)