import pandas as pd
from mes_north_sea.preprocessing.utilities import Configuration, to_latex

import numpy as np


def divide_dataframe(df, n):
    divided_df = df / n
    concatenated_df = pd.DataFrame(divided_df.values.repeat(n, axis=0))
    return concatenated_df

c = Configuration()

hydro_inflows_nodes = pd.DataFrame()

cap_nodes = pd.read_csv(c.clean_data_path + 'clean_data/installed_capacities/capacities_node.csv')
cap_nodes = cap_nodes[['Country', 'Node', 'Technology', 'Capacity our work']]

hydro_tecs = {'Hydro - Pump Storage Open Loop (Energy)': 'Pump storage - Open Loop', 'Hydro - Reservoir (Energy)': 'Reservoir'}

for hydro_tec in hydro_tecs.keys():
    cap_hydro = cap_nodes[cap_nodes['Technology'] == hydro_tec]

    cap_hydro_national = cap_hydro[['Country', 'Capacity our work']].groupby('Country').sum().rename(columns={'Capacity our work': 'National Capacity'})
    cap_hydro = cap_hydro.merge(cap_hydro_national, right_on='Country', left_on='Country')
    cap_hydro['Share'] =  cap_hydro['Capacity our work'] / cap_hydro['National Capacity']

    regions = {'DE': ['DE00'], 'BE': ['BE00'], 'UK': ['UK00'], 'NL': ['NL00'], 'NO': ['NOS0', 'NOM1', 'NON1']}
    for idx, row in cap_hydro.iterrows():
        total_inflow = np.zeros(8760)
        for bidding_zone in regions[row['Country']]:
            data_path = c.load_path_hydro_inflow + bidding_zone + '_Hydro Inflow_' + str(c.year) + '.xlsx'
            temp = pd.read_excel(data_path, sheet_name=hydro_tecs[hydro_tec], skiprows=12,
                                 usecols=[i for i in range(16, 16 + 37)], names=['Week', *range(1982, 2018)])
            flow = divide_dataframe(temp[c.climate_year], 24*7) * 1000
            flow = np.array(flow.fillna(0)[0:8760][0])
            total_inflow = flow + total_inflow
        hydro_inflows_nodes[(row['Node'], hydro_tec)] = total_inflow * cap_hydro.set_index('Node').fillna(0)['Share'][row['Node']]

hydro_inflows_nodes.columns = pd.MultiIndex.from_tuples(hydro_inflows_nodes.columns)

hydro_inflows_nodes.to_csv(c.clean_data_path + 'clean_data/hydro_inflows/hydro_inflows.csv')

to_latex(hydro_inflows_nodes.sum()/1000000, 'Hydro Inflows in TWh (aggregated per Country and per technology)', c.clean_data_path + 'reporting/hydro_inflows/hydro_inflows.tex', rounding=2, columns=None)
