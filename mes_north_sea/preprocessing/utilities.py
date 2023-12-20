import numpy as np
import pandas as pd
from src.components.technologies.genericTechnologies.res import Res
from src.components.technologies.utilities import FittedPerformance

class Configuration:
    def __init__(self):
        self.year = 2030
        self.climate_year = 2009
        self.countries = {'DE': ['DE00'], 'BE': ['BE00'], 'DK': ['DKW1', 'DKE1'], 'UK': ['UK00'], 'NL': ['NL00'],
                     'NO': ['NOM1', 'NON1', 'NOS0']}

        self.clean_data_path = 'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/'

        # DEMAND
        self.raw_data_path_demand = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/00_raw_data/demand/'

        self.loadpath_demand = self.raw_data_path_demand + 'Demand_TimeSeries_' + str(self.year) + '_NationalTrends.xlsx'
        self.savepath_demand_national = self.clean_data_path + 'reporting/demand/national/demand_electricity' #appended by year and climate year
        self.savepath_demand_node_disaggregated = self.clean_data_path + 'reporting/demand/nodal/demand_electricity' #appended by year and climate year
        self.savepath_demand_node_aggregated = self.clean_data_path + 'clean_data/demand/'
        self.savepath_demand_summary = self.clean_data_path + 'reporting/demand/' #appended by year and climate year

        # NUTS to nodes
        self.nodekeys_nuts = pd.read_csv(self.clean_data_path + 'nodekeys/nuts2nodes.csv')

        # PyPSA data
        self.nodekeys_pypsa = pd.read_csv(self.clean_data_path + 'nodekeys/pypsa2nodes.csv')
        self.loadpath_demand_pypsa = self.raw_data_path_demand + 'electricity_pypsa.csv'

        # Eurostat data
        self.loadpath_industrialdemand_eurostat = self.raw_data_path_demand + 'ten00129_page_spreadsheet.xlsx'

        # SBS data
        self.loadpath_emplopyment_sbs = self.raw_data_path_demand + 'sbs_sectors_data.xlsx'

        # Electricity demand 2021
        self.loadpath_demand2019 = self.raw_data_path_demand + 'ten00123_page_spreadsheet.xlsx'


        # CAPACITIES
        self.raw_data_path_cap = 'C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/00_raw_data/installed_capacities/'
        self.savepath_cap_per_node_summary = self.clean_data_path + 'reporting/installed_capacities/non_re_installed_capacities_node.tex'
        self.savepath_cap_per_node_summary_hydro = self.clean_data_path + 'reporting/installed_capacities/hydro_installed_capacities_country.tex'
        self.savepath_cap_per_country_summary = self.clean_data_path + 'reporting/installed_capacities/non_re_installed_capacities_country.tex'
        self.savepath_cap_per_node = self.clean_data_path + 'clean_data/installed_capacities/non_re_installed_capacities.csv'
        self.savepath_cap_per_country = self.clean_data_path + 'reporting/installed_capacities/non_re_installed_capacities.csv'

        # tyndp data
        self.load_path_tyndp_cap = self.raw_data_path_cap + '220310_Updated_Electricity_Modelling_Results.xlsx'
        self.load_path_tyndp_cap_hydro = self.raw_data_path_cap + 'ERAA 2022 PEMMDB National Estimates.xlsx'

        # PyPSA installed capacities
        self.load_path_pypsa_cap_all = self.raw_data_path_cap + 'powerplantspypsa.csv'

        # RE 2023
        self.load_path_re_cap_2023 = self.raw_data_path_cap + 'REinstalled2023NUTS.csv'
        self.load_path_re_potential = self.raw_data_path_cap + 'ENSPRESO_Integrated_NUTS2_Data.csv'

        # Save RE caps
        self.savepath_cap_re_per_country_summary = self.clean_data_path + 'reporting/installed_capacities/re_installed_capacities_country.tex'
        self.savepath_cap_re_per_node_summary = self.clean_data_path + 'reporting/installed_capacities/re_installed_capacities_node.tex'
        self.savepath_cap_re_per_nutsregion = self.clean_data_path + 'clean_data/installed_capacities/re_installed_capacities_nuts.csv'


def to_latex(df, caption, path, rounding=0, columns=None):
    """Writes a latex table"""
    round_format = '{:.' + str(rounding) + 'f}'
    latex_table = df.to_latex(
        index=True,
        na_rep=0,
        formatters={'name': str.upper},
        float_format=round_format.format,
        caption=caption,
        columns=columns
    )
    with open(path, 'w') as f:
        for item in latex_table:
            f.write(item)

class CalculateReGeneration(Res):
    def __init__(self):
        self.fitted_performance = FittedPerformance()
        self.name = 'WindTurbine_Onshore_1500'

    def fit_technology_performance(self, climate_data, technology, location):
        if technology == 'PV':
            self._perform_fitting_PV(climate_data, location)
        elif technology == 'Wind':
            hubheight = 110
            self._perform_fitting_WT(climate_data, hubheight)



