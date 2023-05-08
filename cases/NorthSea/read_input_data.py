import pandas as pd

def read_demand_data_eraa(climate_year, region):
    """
    reads demand data for respective climate year and region from ERAA dataset (year 2030)

    :param int climate_year: climate year to read
    :param str region: region to read
    :return pandas.series demand_profile: demand profile as series
    """
    data_path =  r'./cases/NorthSea/Demand_Electricity/ERAA_Demand_' + region + '_2030.xlsx'
    demand_profiles = pd.read_excel(data_path)
    demand_profile = demand_profiles[climate_year]

    return demand_profile

def read_capacity_factors_eraa(climate_year, region):
    """
    reads production profiles for respective climate year and region from ERAA dataset (year 2030).

    It adds solar, wind onshore and wind offshore respectively for the whole region.

    :param int climate_year: climate year to read
    :param str region: region to read
    :return dict capacity_factors: production profile as series
    """
    data_path = {}
    data_path['PV'] =  r'./cases/NorthSea/ProductionProfiles/ERAA_CapFactor_' + region + '_Solar_2030.xlsx'
    data_path['Wind_On'] =  r'./cases/NorthSea/ProductionProfiles/ERAA_CapFactor_' + region + '_WindOn_2030.xlsx'
    data_path['Wind_Of'] =  r'./cases/NorthSea/ProductionProfiles/ERAA_CapFactor_' + region + '_WindOff_2030.xlsx'

    capacity_factors = {}
    for type in data_path:
        cf = pd.read_excel(data_path[type])
        capacity_factors[type] = cf[str(climate_year)]

    return capacity_factors

def read_installed_capacity_eraa(region):
    """
    reads production profiles for respective climate year and region from ERAA dataset (year 2030).

    It adds solar, wind onshore and wind offshore respectively for the whole region.

    :param int climate_year: climate year to read
    :param str region: region to read
    :return dict capacity_factors: production profile as series
    """
    data_path = r'./cases/NorthSea/InstalledCapacity/ERAA_InstalledCapacity.xlsx'

    instcap = pd.read_excel(data_path, index_col= 1)

    Conv = {}
    RE = {}
    Conv['PowerPlant_Nuclear'] = instcap[region]['Nuclear']
    Conv['PowerPlant_Gas'] = instcap[region]['Gas']
    # Conv['PowerPlant_Lignite'] = instcap[region]['Lignite']
    Conv['PowerPlant_Coal'] = instcap[region]['Hard Coal']
    Conv['Storage_Battery'] = instcap[region]['Batteries']
    Conv['Storage_PumpedHydro_Closed'] = instcap[region]['Hydro - Pump Storage Closed Loop']
    RE['Wind_On'] = instcap[region]['Wind Onshore']
    RE['Wind_Of'] = instcap[region]['Wind Offshore']
    RE['PV'] = instcap[region]['Solar (Photovoltaic)']

    installed_capacities = {}
    installed_capacities['RE'] = RE
    installed_capacities['Conventional'] = Conv

    return installed_capacities



