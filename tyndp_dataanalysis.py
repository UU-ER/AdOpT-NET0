import pandas as pd

scenarios = ['DE', 'GA', 'NT']
climate_years = [1995, 2008, 2009]
years = [2030]
countries = ['onBE', 'onDE', 'onNL', 'onNOS', 'onUK']

summary = pd.DataFrame(columns=['Scenario',
                                'ClimateYear',
                                'Year',
                                'Country',
                                'Demand_avg',
                                'Demand_max',
                                'Demand_min'])

for scenario in scenarios:
    for climate_year in climate_years:
        for year in years:
            read_path = './cases/NorthSea_v3/Demand_Electricity/Demand_' + \
                        scenario + '_' + \
                        str(year) + '_ClimateYear' + \
                        str(climate_year) + '.csv'
            demand_data = pd.read_csv(read_path)
            for country in countries:
                summary.loc[len(summary.index)] = [scenario,
                                                   climate_year,
                                                   year,
                                                   country[2:],
                                                   demand_data[country].mean(),
                                                   demand_data[country].max(),
                                                   demand_data[country].min()]

summary.to_excel('./cases/NorthSea_v3/Demand_Electricity/DemandSummary_v2.xlsx')



