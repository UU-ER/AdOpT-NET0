import numpy as np
import pandas as pd

class Configuration:
    def __init__(self):
        self.year = 2030
        self.climate_year = 2009
        self.countries = {'DE': ['DE00'], 'BE': ['BE00'], 'DK': ['DKW1', 'DKE1'], 'UK': ['UK00'], 'NL': ['NL00'],
                     'NO': ['NOM1', 'NON1', 'NOS0']}

        self.rootpath = 'C:/Users/6574114/Documents/Research/EHUB-Py_Productive/'

        self.loadpath_demand = self.rootpath + 'mes_north_sea/raw_data/demand/Demand_TimeSeries_'+ str(self.year)+'_NationalTrends_without_bat.xlsx'
        self.savepath_demand_national = self.rootpath + 'mes_north_sea/reporting/demand/national/demand_electricity' #appended by year and climate year
        self.savepath_demand_node_disaggregated = self.rootpath + 'mes_north_sea/reporting/demand/nodal/demand_electricity' #appended by year and climate year
        self.savepath_demand_node_aggregated = self.rootpath + 'mes_north_sea/clean_data/demand/'
        self.savepath_demand_summary = self.rootpath + 'mes_north_sea/reporting/demand/' #appended by year and climate year

        # NUTS to nodes
        self.nodekeys_nuts = pd.read_csv(self.rootpath + 'mes_north_sea/nodekeys/nuts2nodes.csv')

        # PyPSA data
        self.nodekeys_pypsa = pd.read_csv(self.rootpath + 'mes_north_sea/nodekeys/pypsa2nodes.csv')
        self.loadpath_demand_pypsa = self.rootpath + 'mes_north_sea/raw_data/demand/electricity_pypsa.csv'

        # Eurostat data
        self.loadpath_industrialdemand_eurostat = self.rootpath + 'mes_north_sea/raw_data/demand/ten00129_page_spreadsheet.xlsx'

        # SBS data
        self.loadpath_emplopyment_sbs = self.rootpath + 'mes_north_sea/raw_data/demand/sbs_sectors_data.xlsx'

        # Electricity demand 2021
        self.loadpath_demand2021 = self.rootpath + 'mes_north_sea/raw_data/demand/ten00123_page_spreadsheet.xlsx'



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
