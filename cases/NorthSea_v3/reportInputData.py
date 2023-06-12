import pandas as pd
import pickle
import numpy as np
from src.data_management.components.fit_technology_performance import perform_fitting_WT
import src.data_management as dm
import json
import os


def report_demand_data():
    scenarios = ['GA', 'DE', 'NT']
    years = [2030, 2040, 2050]
    climate_years = [1995, 2008, 2009]


    load_path = r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/Demand_Electricity/Demand_'
    save_path = r'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/cases/NorthSea/Demand_Electricity/DemandSummary.xlsx'

    with pd.ExcelWriter(save_path) as writer:
        for scenario in scenarios:
            for climate_year in climate_years:
                Node = []
                Yr = []
                Average_GW = []
                Min_GW = []
                Max_GW = []
                Total_TWh = []
                SD_GW = []
                for year in years:
                    if not (scenario == 'NT' and year > 2040):
                        file_name = scenario + '_' + str(year) + '_ClimateYear' + str(climate_year) + '.csv'
                        demand = pd.read_csv(load_path + file_name, index_col=0)
                        for region in demand:
                            Node.append(region)
                            Yr.append(year)
                            Average_GW.append(demand[region].mean()/1000)
                            Min_GW.append(demand[region].min()/1000)
                            Max_GW.append(demand[region].max()/1000)
                            Total_TWh.append(demand[region].sum()/1000000)
                            SD_GW.append(demand[region].std()/1000)
                demand_summary = pd.DataFrame(list(zip(Node, Yr, Average_GW, Min_GW, Max_GW, Total_TWh, SD_GW)),
                                              columns=['Node','Year','Average_GW','Min_GW','Max_GW', 'Total_TWh', 'SD_GW'])
                demand_summary.to_excel(writer, sheet_name=scenario + str(climate_year))


report_demand_data()