import pandas as pd
import numpy as np
from mes_north_sea.preprocessing.utilities import Configuration, to_latex

climate_years = [1995, 2008, 2009]
c = Configuration()

demand2021 = pd.read_excel(c.loadpath_demand2021, sheet_name='Aggregate')
demand2021.set_index('Country Code', inplace=True)
demand2021 = demand2021['Demand (2021)']
demand2021.columns = ['\cite{Eurostat2023c}']


def preprocess_demand_data_eraa(config, region):
    """
    reads demand data for respective climate year and region from ERAA dataset (year 2030)

    :param Configuration config: configuration class
    :param str region: region to read
    :return pandas.series demand_profile: demand profile as series
    """
    demand_profile = pd.read_excel(config.loadpath_demand, sheet_name=region, skiprows=10)

    # Clean data
    demand_profile = demand_profile[config.climate_year]
    mean = demand_profile.mean()
    demand_profile.loc[demand_profile < mean * 0.1] = np.nan
    demand_profile = demand_profile.interpolate()

    return demand_profile



for climate_year in climate_years:
    c.climate_year = climate_year

    # get demand per node using PyPSA demand data
    local_demand = pd.read_csv(c.loadpath_demand_pypsa)
    local_demand['Country'] = local_demand['bus'].str[0:2]
    local_demand = pd.merge(local_demand, c.nodekeys_pypsa, left_on='bus', right_on='Node_PyPSA')
    total_demand_per_country_PyPSA = local_demand.groupby('Country').sum()

    local_demand.insert(7, 'NationalDemand', 99)
    for country in local_demand['Country'].unique():
        local_demand.loc[local_demand['Country'] == country, 'NationalDemand'] = \
        total_demand_per_country_PyPSA['demand'][
            country]
    local_demand['share_of_national_demand'] = local_demand['demand'] / local_demand['NationalDemand']
    local_demand = local_demand.groupby('Node').sum()
    total_demand_nodal_PyPSA = pd.DataFrame(local_demand['demand'] * 10**6)
    local_demand = local_demand[['share_of_national_demand']]

    # get national industrial demand
    industry_share = pd.read_excel(c.loadpath_industrialdemand_eurostat, sheet_name='Aggregate')

    # get employees in manufactoring
    employees = pd.read_excel(
        'C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/Demand/sbs_sectors_data.xlsx',
        sheet_name='Extract')
    employees = pd.merge(employees, c.nodekeys_nuts, right_on='NUTS_ID', left_on='Region')
    employees = employees[['CNTR_CODE', 'NUTS_ID', 'Share of national', 'Node']]
    share_industry_per_node = employees.groupby('Node').sum()
    share_industry_per_node.loc['NO1', 'Share of national'] = 1
    share_industry_per_node.loc['DK1', 'Share of national'] = 1
    local_demand = local_demand.join(share_industry_per_node)

    local_demand['industrial_demand'] = 0
    local_demand['total_demand'] = 0


    # Aggrate bidding zones and save national demand
    national_demand = pd.DataFrame(index=np.arange(8760))
    for country in c.countries:
        national_demand[country] = 0
        for bidding_zone in c.countries[country]:
            pass
            national_demand[country] = national_demand[country] + preprocess_demand_data_eraa(c, bidding_zone)

        national_demand.to_csv(c.savepath_demand_national + '_' + str(c.year) + '_cl' + str(climate_year) + '.csv')
        total_demand_per_country_ERAA = national_demand.sum()

        total_demand_per_country = pd.merge(total_demand_per_country_ERAA.to_frame(), industry_share, left_index=True, right_on='Country_Code')
        total_demand_per_country = total_demand_per_country[['Country_Code' ,0, 'share']]
        total_demand_per_country = total_demand_per_country.rename(columns={0: 'total_demand', 'share': 'industrial_share'})
        total_demand_per_country['industrial_demand'] = total_demand_per_country['total_demand'] * total_demand_per_country['industrial_share']
        total_demand_per_country['other_demand'] = total_demand_per_country['total_demand'] - total_demand_per_country['industrial_demand']

    for node in local_demand.index:
        print(node)
        country = c.nodekeys_nuts['CNTR_CODE'][c.nodekeys_nuts['Node'] == node].unique()
        country = country[0]
        print(country)
        total_industrial_demand = total_demand_per_country['industrial_demand'][total_demand_per_country['Country_Code'] == country].iloc[0]
        total_demand = total_demand_per_country['total_demand'][total_demand_per_country['Country_Code'] == country].iloc[0]
        local_demand.loc[node, 'industrial_demand'] = total_industrial_demand * local_demand['Share of national'][node]
        local_demand.loc[node, 'total_demand'] = total_demand * local_demand['share_of_national_demand'][node]
    local_demand['other_demand'] =  local_demand['total_demand'] - local_demand['industrial_demand']
    local_demand = local_demand[['industrial_demand', 'total_demand','other_demand']]
    local_demand = local_demand.merge(c.nodekeys_nuts[['Node', 'CNTR_CODE']].drop_duplicates(), left_index=True, right_on='Node', how='left')

    # Generate profiles
    nodal_profile_other = pd.DataFrame()
    nodal_profile_ind = pd.DataFrame()
    nodal_profile = pd.DataFrame()
    for node in local_demand['Node']:
        country = local_demand['CNTR_CODE'][local_demand['Node'] == node].iloc[0]
        national_profile = national_demand[country]
        national_profile_industrial = np.ones(8760) * (sum(national_profile) * industry_share['share'][industry_share['Country_Code'] == country].iloc[0]) / 8760
        national_profile_other = national_profile - national_profile_industrial

        regional_share = local_demand['other_demand'][local_demand['Node'] == node].iloc[0] / total_demand_per_country['other_demand'][total_demand_per_country['Country_Code'] == country].iloc[0]

        nodal_profile_ind[node] = np.ones(8760) * local_demand['industrial_demand'][local_demand['Node'] == node].iloc[0] /8760
        nodal_profile_other[node] = national_profile_other * regional_share
        nodal_profile[node] = nodal_profile_ind[node] + nodal_profile_other[node]
    nodal_profile_ind.to_csv(c.savepath_demand_node_disaggregated + 'IndustrialDemand' + '_' + str(climate_year) + '.csv')
    nodal_profile_other.to_csv(c.savepath_demand_node_disaggregated + 'OtherDemand' + '_' + str(climate_year) + '.csv')
    nodal_profile.to_csv(c.savepath_demand_node_aggregated + 'TotalDemand' + '_NT_' + str(climate_year) + '.csv')
    # local_demand.to_csv(c.savepath_demand_summary + '_' + str(climate_year) + '.csv')

    total_demand_national_ERAA = pd.DataFrame(total_demand_per_country_ERAA)
    total_demand_national_ERAA.columns = ['ERAA']
    total_demand_national_PyPSA = pd.DataFrame(total_demand_per_country_PyPSA['demand'] * 10**6)
    total_demand_national_PyPSA.columns = ['\cite{Neumann2023}']

    total_demand_national = pd.merge(total_demand_national_ERAA, total_demand_national_PyPSA, left_index=True, right_index=True)
    total_demand_national = pd.merge(total_demand_national, demand2021, left_index=True, right_index=True)
    total_demand_national.to_csv(c.savepath_demand_summary + 'NationalDemand_' + str(climate_year) + '.csv')
    to_latex(total_demand_national * 10**-6, 'National annual projected demand for 2030  GWh for the climate year ' + str(climate_year), c.savepath_demand_summary + 'LatexNationalDemand_' + str(climate_year) + '.tex')

    local_demand.set_index('Node', inplace=True)
    total_demand_nodal_PyPSA.columns = ['PyPsa_demand']
    total_demand_nodal = pd.DataFrame(local_demand[['total_demand', 'industrial_demand','other_demand']])
    total_demand_nodal = pd.merge(total_demand_nodal, total_demand_nodal_PyPSA, left_index=True, right_index=True)
    total_demand_nodal.to_csv(c.savepath_demand_summary + 'NodalDemand_' + str(climate_year) + '.csv')
    to_latex(total_demand_nodal * 10**-6, 'Nodal annual projected demand for 2030 in GWh for the climate year ' + str(climate_year), c.savepath_demand_summary + 'LatexNodalDemand_' + str(climate_year) + '.tex')
