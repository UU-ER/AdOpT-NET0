import numpy as np
import pandas as pd
from types import SimpleNamespace

from mes_north_sea.preprocessing.utilities import Configuration, to_latex, CalculateReGeneration

c = Configuration()

pypsa_network_all = pd.read_csv(c.load_path_networks + 'electricity_pypsa.csv')
pypsa_nodekeys = c.nodekeys_pypsa[['Node_PyPSA', 'Node']]
countries_used = ['BE', 'NL', 'UK', 'NO', 'DK', 'DE']

def calculate_max_line_cap(df):
    df[['Node1', 'Node2']] = df.apply(lambda row: sorted([row['Node1'], row['Node2']]), axis=1, result_type='expand')
    df = df.set_index(['Node1', 'Node2'])
    df = df.groupby(['Node1', 'Node2']).max()
    return df

networks = {}
# Onshore
for type in ['AC', 'DC']:
    # TYNDP
    tyndp = pd.read_excel(c.load_path_networks + 'Transfer Capacities_ERAA2022_TY2030.xlsx', sheet_name='HV' + type, skiprows=9, header=None)
    tyndp_ntc = tyndp.drop(tyndp.columns[0:2], axis=1)
    tyndp_ntc = tyndp_ntc.T
    tyndp_ntc = tyndp_ntc.drop(tyndp_ntc.columns[0], axis=1)
    tyndp_ntc = tyndp_ntc.drop(tyndp_ntc.columns[2:6], axis=1)
    tyndp_ntc = tyndp_ntc.rename(columns = {1: 'Node1', 2: 'Node2'})
    tyndp_ntc = tyndp_ntc.set_index(['Node1', 'Node2']).T
    tyndp_ntc = tyndp_ntc.max().to_frame()
    tyndp_ntc = tyndp_ntc.rename(columns = {0: 'Cap'})

    tyndp_ntc = tyndp_ntc.reset_index()
    tyndp_ntc = calculate_max_line_cap(tyndp_ntc)
    tyndp_ntc = tyndp_ntc.reset_index()
    tyndp_ntc['Country1'] = tyndp_ntc['Node1'].str[0:2]
    tyndp_ntc['Country2'] = tyndp_ntc['Node2'].str[0:2]
    tyndp_ntc = tyndp_ntc[tyndp_ntc['Country1'].isin(countries_used) | tyndp_ntc['Country2'].isin(countries_used)]
    tyndp_ntc.loc[~tyndp_ntc['Country1'].isin(countries_used), 'Country1'] = 'export'
    tyndp_ntc.loc[~tyndp_ntc['Country2'].isin(countries_used), 'Country2'] = 'export'
    tyndp_ntc[['Country1', 'Country2']] = tyndp_ntc.apply(lambda row: sorted([row['Country1'], row['Country2']]), axis=1, result_type='expand')
    tyndp_ntc = tyndp_ntc.set_index(['Country1', 'Country2'])
    tyndp_ntc = tyndp_ntc.groupby(['Country1', 'Country2']).sum().reset_index()
    tyndp_ntc = tyndp_ntc[tyndp_ntc['Country1'] != tyndp_ntc['Country2']]
    tyndp_ntc.to_csv(c.savepath_network_summary + 'ERAAnetwork' + type + '.csv')

    networks[type] = tyndp_ntc

    # PyPSA
    pypsa_network = pypsa_network_all[pypsa_network_all['carrier'] == type]
    netw = pypsa_network[['bus0', 'bus1', 's_nom', 's_nom_max']]
    netw = netw.rename(columns = {'bus0': 'Node1', 'bus1': 'Node2'})
    netw = calculate_max_line_cap(netw) * 1000

    netw = netw.reset_index()

    netw = pd.merge(netw, pypsa_nodekeys, left_on='Node1', right_on='Node_PyPSA')
    netw = netw.rename(columns={'Node': 'Node1_ours'})
    netw = netw.drop(columns = 'Node_PyPSA')
    netw = pd.merge(netw, pypsa_nodekeys, left_on='Node2', right_on='Node_PyPSA')
    netw = netw.rename(columns={'Node': 'Node2_ours'})

    netw = netw.drop(columns = ['Node_PyPSA', 'Node1', 'Node2'])

    netw = netw[netw['Node1_ours'] != netw['Node2_ours']]
    netw = netw[netw['Node1_ours'].notna() | netw['Node2_ours'].notna()]
    netw = netw.fillna(value = 'export')
    netw = netw.groupby(['Node1_ours', 'Node2_ours']).sum().reset_index()
    netw['Country1'] = netw['Node1_ours'].str[0:2]
    netw['Country2'] = netw['Node2_ours'].str[0:2]
    netw.to_csv(c.savepath_network_summary + 'PyPSAnetwork' + type + '.csv')

    netw[['Country1', 'Country2']] = netw.apply(lambda row: sorted([row['Country1'], row['Country2']]),
                                                          axis=1, result_type='expand')
    netw = netw.set_index(['Country1', 'Country2'])
    netw = netw.groupby(['Country1', 'Country2']).sum().reset_index()
    netw = netw[netw['Country1'] != netw['Country2']]
    netw.to_csv(c.savepath_network_summary + 'PyPSAnetwork_national' + type + '.csv')






