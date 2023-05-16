import pandas as pd
import pickle
import numpy as np


# Prepare demand data
def preprocess_demand_data_eraa(scenario, year, climate_year, region):
    """
    reads demand data for respective climate year and region from ERAA dataset (year 2030)

    :param int climate_year: climate year to read
    :param str region: region to read
    :return pandas.series demand_profile: demand profile as series
    """
    data_path = r'E:/00_Data/00_ElectricityDemandEurope/ENTSOE_TYNDP/'

    # Determine file name
    if scenario == 'NT':
        filename = 'Demand_TimeSeries_'+ str(year)+'_NationalTrends.xlsx'
        skip_rows = 10

    elif scenario == 'GA':
        filename = 'Demand_TimeSeries_' + str(year) + '_GA_release.xlsb'
        skip_rows = 6
    elif scenario == 'DE':
        filename = 'Demand_TimeSeries_' + str(year) + '_DE_release.xlsb'
        skip_rows = 6

    demand_profile = pd.read_excel(data_path + filename, sheet_name=region, skiprows=skip_rows)
    demand_profile = demand_profile[climate_year]

    return demand_profile


def scale_demand_data(national_profile, scaling_factors):


    key_path = r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/Demand_Electricity/Scaling.xlsx'

    keys = pd.read_excel(key_path, index_col= 0, sheet_name='ToPython')

    # read in population keys
    population_keys = {}
    for prov in keys['Population_key'].keys():
        population_keys[prov] = keys['Population_key'][prov]

    # read in industrial demand keys
    industry_keys = {}
    for prov in keys['Industrial_key'].keys():
        industry_keys[prov] = keys['Industrial_key'][prov]

    regions = industry_keys.keys()

    # This is from IEA statistics
    share_demand = {}
    share_demand['Res'] = 0.21
    share_demand['Ind'] = 1-share_demand['Res']

    # residential/industrial demand not scaled
    demand_not_scaled = {}
    demand_not_scaled['Total'] = national_profile

    demand_not_scaled['Res'] = {}
    demand_not_scaled['Res']['Total'] = national_profile * share_demand['Res']
    for prov in regions:
        demand_not_scaled['Res'][prov] = population_keys[prov] * demand_not_scaled['Res']['Total']

    demand_not_scaled['Ind'] = {}
    demand_not_scaled['Ind']['Total'] = national_profile * share_demand['Ind']
    for prov in regions:
        demand_not_scaled['Ind'][prov] = industry_keys[prov] * demand_not_scaled['Ind']['Total']

    # Scaled demand for res, ind
    demand_scaled = {}
    for type in ['Res', 'Ind']:
        demand_scaled[type] = {}
        for prov in regions:
            demand_scaled[type][prov] = (demand_not_scaled[type][prov] - demand_not_scaled[type][prov].mean()) * \
                                        scaling_factors[type] + demand_not_scaled[type][prov].mean()

    # Total demand
    demand_scaled['Total'] = {}
    demand_scaled['Total']['Total'] = None
    for prov in regions:
        demand_scaled['Total'][prov] = demand_scaled['Res'][prov] + demand_scaled['Ind'][prov]
        if isinstance(demand_scaled['Total']['Total'], pd.Series):
            demand_scaled['Total']['Total'] = demand_scaled['Total']['Total'] + demand_scaled['Total'][prov]
        else:
            demand_scaled['Total']['Total'] = demand_scaled['Total'][prov]

    # for prov in regions:
    #     print(demand_scaled['Total'][prov].std()/demand_scaled['Total'][prov].mean())

    # Correct for delta
    correction_delta = demand_scaled['Total']['Total'] - demand_not_scaled['Total']
    demand_corrected = {}
    share_demand_hour = {}
    for prov in regions:
        share_demand_hour[prov] = demand_scaled['Total'][prov] / demand_scaled['Total']['Total']
        demand_corrected[prov] = demand_scaled['Total'][prov] - correction_delta * share_demand_hour[prov]

    # demand_corrected['Total'] = sum(demand_corrected.values())

    demand_corrected = pd.DataFrame.from_dict(demand_corrected)

    return demand_corrected


def aggregate_provinces_to_node(demand_corrected):

    # Read keys
    key_path = r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/Demand_Electricity/Scaling.xlsx'
    keys = pd.read_excel(key_path, index_col= 0, sheet_name='ToPython')

    demand_profiles = {}
    nodes = keys['Node'].values
    nodes = np.unique(nodes)
    for node in nodes:
        provinces_at_node = keys[keys['Node'] == node].index
        demand_profiles[node] = demand_corrected[provinces_at_node].sum(axis = 1)

    demand_profiles = pd.DataFrame.from_dict(demand_profiles)

    return demand_profiles

save_path = r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/Demand_Electricity/Demand_'

scenarios = ['GA', 'DE', 'NT']
regions = ['DE00', 'BE00', 'DKW1', 'UK00', 'NL00', 'NOS0']
years = [2030, 2040, 2050]
climate_years = [1995, 2008, 2009]
scaling_factors = {'Res': 2, 'Ind': 0.3}


for scenario in scenarios:
    for year in years:
        for climate_year in climate_years:
            if not (scenario == 'NT' and year > 2040):
                demand = pd.DataFrame(index=range(0,8760))
                file_name = scenario + '_' + str(year) + '_ClimateYear' + str(climate_year) + '.csv'
                for region in regions:
                    demand[region] = preprocess_demand_data_eraa(scenario, year, climate_year, region)
                    if region == 'NL00':
                        NL_national_profile = demand[region]
                        demandNL_corrected = scale_demand_data(NL_national_profile, scaling_factors)
                        demandNL_at_nodes = aggregate_provinces_to_node(demandNL_corrected)
                        demand = pd.concat([demand, demandNL_at_nodes], axis = 1)
                demand.to_csv(save_path + file_name)