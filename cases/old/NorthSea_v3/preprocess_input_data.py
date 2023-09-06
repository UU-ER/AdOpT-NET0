import pandas as pd
import pickle
import numpy as np
from cases.NorthSea.read_input_data import *
from src.data_management import average_series

# Prepare demand data
def preprocess_demand_data_eraa(scenario, year, climate_year, region):
    """
    reads demand data for respective climate year and region from ERAA dataset (year 2030)

    :param int climate_year: climate year to read
    :param str region: region to read
    :return pandas.series demand_profile: demand profile as series
    """
    data_path = r'E:/00_Data/00_EnergyDemandEurope/Electricity/ENTSOE_TYNDP/'

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
    # Clean data


    demand_profile = demand_profile[climate_year]
    mean = demand_profile.mean()
    demand_profile.loc[demand_profile < mean * 0.1] = np.nan
    demand_profile = demand_profile.interpolate()

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


def read_capacity_factors_eraa(climate_year, region):
    """
    reads production profiles for respective climate year and region from ERAA dataset (year 2030).

    It adds solar, wind onshore and wind offshore respectively for the whole region.

    :param int climate_year: climate year to read
    :param str region: region to read
    :return dict capacity_factors: production profile as series
    """
    data_path = {}
    data_path['PV'] =  r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/ProductionProfiles/ERAA_CapFactor_' + region + '_Solar_2030.xlsx'
    data_path['Wind_On'] =  r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/ProductionProfiles/ERAA_CapFactor_' + region + '_WindOn_2030.xlsx'
    data_path['Wind_Of'] =  r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/ProductionProfiles/ERAA_CapFactor_' + region + '_WindOff_2030.xlsx'

    capacity_factors = {}
    for type in data_path:
        cf = pd.read_excel(data_path[type])
        try:
            capacity_factors[type] = cf[str(climate_year)]
        except:
            capacity_factors[type] = cf[climate_year]

    return capacity_factors


def divide_dataframe(df, n):
    # Divide each entry by 24
    divided_df = df / n

    # Concatenate the dataframe
    concatenated_df = pd.DataFrame(divided_df.values.repeat(n, axis=0))

    return concatenated_df


def scale_capacity_factors(profile, climate_year, sd = 0.05):

    key_path = r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/Demand_Electricity/Scaling.xlsx'
    keys = pd.read_excel(key_path, index_col= 0, sheet_name='ToPython')

    cap_factors = read_capacity_factors_eraa(climate_year, 'NL00')

    data_path = r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/InstalledCapacity/ERAA_InstalledCapacity.xlsx'
    installed_capacity_NL = pd.read_excel(data_path, index_col= 0, sheet_name='InstalledCapacitiesNL')

    installed_capacity_at_node = {}
    installed_capacity_at_node['PV'] = {}
    installed_capacity_at_node['Wind'] = {}

    nodes = keys['Node'].values
    nodes = np.unique(nodes)

    for tec in installed_capacity_at_node:
        installed_capacity_at_node[tec] = {}
        for node in nodes:
            provinces_at_node = keys[keys['Node'] == node].index
            installed_capacity_at_node[tec][node] = installed_capacity_NL[tec + '_2030'][provinces_at_node].sum()

    for region in nodes:
        profile[region + '_tot'] = 0
        for series in installed_capacity_at_node:
            if series == 'Wind':
                series_aux = 'Wind_On'
            else:
                series_aux = series

            cf = cap_factors[series_aux] * np.random.normal(1, sd, size=len(cap_factors[series_aux]))


            profile[region + '_tot'] = profile[region + '_tot'] + cf * \
                                      installed_capacity_at_node[series][region]
            profile[region + '_' + series] = cf * installed_capacity_at_node[series][region]

    return profile


def save_demand_data_to_csv():
    scenarios = ['GA', 'DE', 'NT']
    regions = {'onDE': 'DE00', 'onBE': 'BE00', 'onDKW': 'DKW1', 'onUK': 'UK00', 'onNL': 'NL00', 'onNOS': 'NOS0'}
    years = [2030, 2040, 2050]
    climate_years = [1995, 2008, 2009]
    scaling_factors = {'Res': 2, 'Ind': 0.3}


    save_path = r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/Demand_Electricity/Demand_'
    for scenario in scenarios:
        for year in years:
            for climate_year in climate_years:
                if not (scenario == 'NT' and year > 2040):
                    demand = pd.DataFrame(index=range(0,8760))
                    file_name = scenario + '_' + str(year) + '_ClimateYear' + str(climate_year) + '.csv'
                    for region in regions:
                        demand[region] = preprocess_demand_data_eraa(scenario, year, climate_year, regions[region])
                        if regions[region] == 'NL00':
                            NL_national_profile = demand[region]
                            demandNL_corrected = scale_demand_data(NL_national_profile, scaling_factors)
                            demandNL_at_nodes = aggregate_provinces_to_node(demandNL_corrected)
                            demand = pd.concat([demand, demandNL_at_nodes], axis = 1)
                    demand.to_csv(save_path + file_name)


def save_generic_production_profiles_to_csv():
    scenarios = ['GA', 'DE', 'NT']
    regions = {'onDE': 'DE00', 'onBE': 'BE00', 'onDKW': 'DKW1', 'onUK': 'UK00', 'onNL': 'NL00', 'onNOS': 'NOS0'}
    years = [2030, 2040, 2050]
    climate_years = [1995, 2008, 2009]
    scaling_factors = {'Res': 2, 'Ind': 0.3}

    for climate_year in climate_years:
        save_path = r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/ProductionProfiles/Production_Profiles' + str(
            climate_year) + '.csv'

        # Other regions
        cap_factors = {}
        installed_capacities = {}
        run_of_river_output = {}

        # Capacity Factors
        for region in regions:
            cap_factors[region] = read_capacity_factors_eraa(climate_year, regions[region])
            installed_capacities[region] = read_installed_capacity_eraa(region)

        # Run of River
        columns = [i for i in range(16, 16 + 37)]
        column_names = ['Day', *range(1982, 2018)]
        for region in regions:
            if not region == 'onDKW' and not region == 'onNOS':
                output = pd.read_excel(
                    'E:/00_Data/00_RenewableGeneration/ENTSOE_ERAA/Hydro Inflows/PEMMDB_' + regions[region] + '_Hydro Inflow_2030.xlsx',
                    sheet_name='Run of River', skiprows=12, usecols=columns, names=column_names)
                run_of_river_output[region] = divide_dataframe(output[climate_year], 24) * 1000

        region = 'onDE'
        profile = pd.DataFrame(index=cap_factors[region]['PV'].index)
        for region in regions:
            profile[region + '_tot'] = 0
            for series in cap_factors[region]:
                profile[region + '_tot'] = profile[region + '_tot'] + cap_factors[region][series] * \
                                           installed_capacities[region]['RE'][series]
                profile[region + '_' + series] = cap_factors[region][series] * installed_capacities[region]['RE'][
                    series]
            if not region == 'onDKW' and not region == 'onNOS':
                profile[region + '_run_of_river'] = run_of_river_output[region]
                profile[region + '_tot'] = profile[region + '_tot'] + run_of_river_output[region][0][0:8760]

        # Generic Production NL
        profile = scale_capacity_factors(profile, climate_year, sd=0.05)

        profile.to_csv(save_path)


def save_hydro_inflows_to_csv():
    scenarios = ['GA', 'DE', 'NT']
    regions = {'onDE': 'DE00', 'onBE': 'BE00', 'onDKW': 'DKW1', 'onUK': 'UK00', 'onNL': 'NL00', 'onNOS': 'NOS0'}
    years = [2030, 2040, 2050]
    climate_years = [1995, 2008, 2009]
    scaling_factors = {'Res': 2, 'Ind': 0.3}
    year = 2030
    hydro_types = ['Reservoir', 'Pump storage - Open Loop']
    columns = [i for i in range(16,16+37)]
    column_names = ['Week', *range(1982, 2018)]
    inflow = pd.DataFrame()

    for hydro_type in hydro_types:
            for region in regions:
                if not region == 'onDKW':
                    data_path = r'E:/00_Data/00_RenewableGeneration/ENTSOE_ERAA/Hydro Inflows/PEMMDB_' + regions[region] + '_Hydro Inflow_' + str(
                        year) + '.xlsx'
                    for climate_year in climate_years:
                        temp = pd.read_excel(data_path, sheet_name=hydro_type, skiprows=12, usecols= columns, names=column_names)
                        temp = temp
                        temp = divide_dataframe(temp[climate_year], 7*24) * 1000
                        inflow[region] = temp[0:8760]
                        save_path = r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/Hydro_Inflows/HydroInflow' + hydro_type + str(
                            climate_year) + '.csv'
                        inflow.to_csv(save_path)


def calculate_h2_demand():
    scenarios = {'GA': 'Global Ambition', 'DE': 'Distributed Energy'}
    regions = {'onDE': 'DE', 'onBE': 'BE', 'onDKW': 'DK', 'onUK': 'UK', 'NL00': 'NL', 'onNOS': 'NO'}
    years = [2030, 2040, 2050]
    climate_years = [1995, 2008, 2009]


    climate_data_path = 'C:/Users/6574114/Documents/Research/era5download/Borssele_Kavel_I.csv'
    climate_data = pd.read_csv(climate_data_path)
    T = climate_data['temp_air']
    # T_avg = average_series(T,24)
    base_temp = 15.5
    HDD = base_temp - T
    HDD[T >= base_temp] = 0

    total_HDD = HDD.sum()
    h2_allocation_for_heating = HDD / total_HDD


    h2_demand_path = 'E:/00_Data/00_EnergyDemandEurope/Hydrogen/220404_Updated_Gas_Data.xlsx'
    h2_demand_data = pd.read_excel(h2_demand_path, sheet_name='Total_demand')
    h2_demand_data = h2_demand_data.loc[(h2_demand_data['Parameter'] == 'Final demand') | (h2_demand_data['Parameter'] == 'Final Demand')]
    h2_demand_data = h2_demand_data.loc[(h2_demand_data['Fuel type'] == 'Hydrogen')]

    save_path = r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/Demand_Hydrogen/Demand_'
    for scenario in scenarios:
        for year in years:
            for climate_year in climate_years:
                demand = pd.DataFrame(index=range(0, 8760))
                file_name = scenario + '_' + str(year) + '_ClimateYear' + str(climate_year) + '.csv'
                for region in regions:
                    profile = np.ones(8760)

                    # Demand per year in GWh
                    total_demand = h2_demand_data.loc[(h2_demand_data['Country'] == regions[region]) &
                                                      (h2_demand_data['Scenario'] == scenarios[scenario]) &
                                                      (h2_demand_data['Year'] == year) &
                                                      (h2_demand_data['Case'] == 'Average (GWh/d)')
                    ]['Value'].values * 365
                    if len(total_demand) == 0:
                        total_demand = 0

                    heating_demand = h2_demand_data.loc[(h2_demand_data['Country'] == regions[region]) &
                                                      (h2_demand_data['Scenario'] == scenarios[scenario]) &
                                                      (h2_demand_data['Year'] == year) &
                                                      (h2_demand_data['Category'] == 'Heating_cooling')
                    ]['Value'].values *1000
                    if len(heating_demand) == 0:
                        heating_demand = 0
                    if regions[region] == 'NL':
                        NL_regions = [
                        'onNL_NE',
                        'onNL_SE',
                        'onNL_NW',
                        'onNL_SW',
                        'onNL_CE',
                        'onNL_W',
                        'onNL_E']
                        for NL_region in NL_regions:
                            demand[NL_region] = (profile * (total_demand - heating_demand) / 8760 +
                                              h2_allocation_for_heating * heating_demand) * 1000 / len(NL_regions)
                    else:
                        demand[region] = (profile * (total_demand - heating_demand) / 8760 +
                                          h2_allocation_for_heating * heating_demand) * 1000
                demand.to_csv(save_path + file_name)




save_demand_data_to_csv()
save_generic_production_profiles_to_csv()
save_hydro_inflows_to_csv()
calculate_h2_demand()








