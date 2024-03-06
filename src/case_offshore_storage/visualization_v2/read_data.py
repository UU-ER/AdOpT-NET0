import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import h5py
import numpy as np
import datetime

from utilities import *

@st.cache_data
def read_technology_operation(path_h5, path_re_gen):

    # Energybalance
    with h5py.File(path_h5, 'r') as hdf_file:
        df_bal = extract_datasets_from_h5_group(hdf_file["operation/energy_balance"])

    df_bal = df_bal.rename_axis(columns=['Node', 'Carrier', 'Variable']).T.reset_index()
    df_bal['Country'] = df_bal['Node'].str[0:2]
    df_bal['Technology'] = 'Energy Balance'
    df_bal = df_bal.set_index(['Country', 'Node', 'Technology', 'Carrier', 'Variable']).T

    # Curtailment
    max_re = pd.read_csv(path_re_gen,
        index_col=0, header=[0, 1])
    max_re = max_re.loc[:, (slice(None), 'total')].T.reset_index()
    max_re = max_re.drop(['Profile'], axis=1)
    max_re['Country'] = max_re['Node'].str[0:2]
    max_re['Carrier'] = 'electricity'
    max_re['Variable'] = 'max re'
    max_re['Technology'] = 'Energy Balance'
    max_re = max_re.set_index(['Country', 'Node', 'Technology', 'Carrier', 'Variable']).T
    max_re = max_re.reset_index(drop=True)

    df_bal = pd.concat([df_bal, max_re], axis=1)

    curtailment = df_bal.xs('max re', level='Variable', axis=1) - df_bal.xs('generic_production', level='Variable', axis=1)
    curtailment = curtailment.dropna(axis=1, how='any')
    curtailment = curtailment.T.reset_index()
    curtailment['Variable'] = 'Curtailment'
    curtailment = curtailment.set_index(['Country', 'Node', 'Technology', 'Carrier', 'Variable']).T

    df_bal = pd.concat([df_bal, curtailment], axis=1)

    # Technology Outputs
    with h5py.File(path_h5, 'r') as hdf_file:
        df_ope = extract_datasets_from_h5_group(hdf_file["operation/technology_operation"])

    df_ope = df_ope.rename_axis(columns=['Node', 'Technology', 'Variable']).T.reset_index()
    df_ope['Country'] = df_ope['Node'].str[0:2]
    df_ope = df_ope.set_index(['Country', 'Node', 'Technology', 'Variable']).T

    df_ope_el_out = df_ope.xs('electricity_output', level='Variable', axis=1).T.reset_index()
    df_ope_el_in = df_ope.xs('electricity_input', level='Variable', axis=1).T.reset_index()

    df_ope_el_out['Carrier'] = 'electricity'
    df_ope_el_in['Carrier'] = 'electricity'
    df_ope_el_out['Variable'] = 'output'
    df_ope_el_in['Variable'] = 'input'
    df_ope_el_out = df_ope_el_out.set_index(['Country', 'Node', 'Technology', 'Carrier', 'Variable']).T
    df_ope_el_in = df_ope_el_in.set_index(['Country', 'Node', 'Technology', 'Carrier', 'Variable']).T

    # Emissions
    df_emissions_tec = df_ope.xs('emissions_pos', level='Variable', axis=1).T.reset_index()
    df_emissions_tec = df_emissions_tec.set_index(['Country', 'Node', 'Technology']).T
    df_emissions_tec = df_emissions_tec.T.reset_index().groupby(['Country', 'Node']).sum()
    df_emissions_tec = df_emissions_tec.reset_index()
    df_emissions_tec['Variable'] = 'Technology Emissions'

    df_emissions_import = df_bal.xs('import', level='Variable', axis=1).T.reset_index()
    df_emissions_import = df_emissions_import[df_emissions_import['Carrier'] == 'electricity']
    df_emissions_import.drop(columns=['Technology', 'Carrier'], inplace=True)
    df_emissions_import = df_emissions_import.set_index(['Country', 'Node'])
    df_emissions_import = df_emissions_import * 0.8
    df_emissions_import = df_emissions_import.reset_index()
    df_emissions_import['Variable'] = 'Import Emissions'

    df_emissions_export = df_bal.xs('export', level='Variable', axis=1).T.reset_index()
    df_emissions_export = df_emissions_export[df_emissions_export['Carrier'] == 'hydrogen']
    df_emissions_export.drop(columns=['Technology', 'Carrier'], inplace=True)
    df_emissions_export = df_emissions_export.set_index(['Country', 'Node'])
    df_emissions_export = df_emissions_export * (-0.18)
    df_emissions_export = df_emissions_export.reset_index()
    df_emissions_export['Variable'] = 'Export Emissions'

    df_emissions = pd.concat([df_emissions_tec, df_emissions_import, df_emissions_export])
    df_emissions = df_emissions.set_index(['Country', 'Node', 'Variable'])
    df_emissions = df_emissions.groupby(['Country', 'Node']).sum().reset_index()
    df_emissions['Technology'] = 'Emission Balance'
    df_emissions['Carrier'] = 'Emissions'
    df_emissions['Variable'] = 'Emissions'

    df_emissions = df_emissions.set_index(['Country', 'Node', 'Technology', 'Carrier', 'Variable']).T

    df_all = pd.concat([df_bal, df_ope_el_out, df_ope_el_in, df_emissions], axis=1)

    hour = df_all.index.to_list()

    num_rows = len(hour)
    day = np.repeat(np.arange(1, num_rows + 1), 24)[0:num_rows]
    week = np.repeat(np.arange(1, num_rows + 1), 24*7)[0:num_rows]
    month = pd.date_range(start='2008-01-01 00:00', end='2008-12-31 00:00', freq='1h').month[0:num_rows].to_list()
    year = np.ones(num_rows)

    df_all.index = pd.MultiIndex.from_arrays([hour, day, week, month, year],
                                            names=['Hour', 'Day', 'Week', 'Month', 'Year'])

    # Emissions


    return df_all

@st.cache_data
def read_network(path_h5):
    with h5py.File(path_h5, 'r') as hdf_file:
        network_design = extract_datasets_from_h5_group(hdf_file["design/networks"])

    network_design = network_design.melt()
    network_design.columns = ['Network', 'Arc_ID', 'Variable', 'Value']
    network_design = network_design.pivot(columns='Variable', index=['Arc_ID', 'Network'], values='Value')
    network_design['FromNode'] = network_design['fromNode'].str.decode('utf-8')
    network_design['ToNode'] = network_design['toNode'].str.decode('utf-8')
    network_design.drop(columns=['fromNode', 'toNode', 'network'], inplace=True)
    network_design = network_design.reset_index()
    arc_ids = network_design[['Arc_ID', 'FromNode', 'ToNode']]

    with h5py.File(path_h5, 'r') as hdf_file:
        network_operation = extract_datasets_from_h5_group(hdf_file["operation/networks"])

    network_operation.columns.names = ['Network', 'Arc_ID', 'Variable']

    network_operation = network_operation.T.reset_index()
    network_operation = pd.merge(network_operation, arc_ids.drop_duplicates(subset=['Arc_ID']), how='inner', left_on='Arc_ID', right_on='Arc_ID')
    network_operation['FromCountry'] = network_operation['FromNode'].str[0:2]
    network_operation['ToCountry'] = network_operation['ToNode'].str[0:2]
    network_operation['Country_ID'] = network_operation['FromNode'].str[0:2] + network_operation['ToNode'].str[0:2]
    network_operation = network_operation.set_index(['Network', 'Arc_ID', 'Country_ID', 'Variable', 'FromNode', 'ToNode', 'FromCountry', 'ToCountry']).T

    network_operation = network_operation.drop(columns=[col for col in network_operation.columns if 'losses' in col])

    hour = network_operation.index.to_list()
    num_rows = len(hour)
    day = np.repeat(np.arange(1, num_rows + 1), 24)[0:num_rows]
    week = np.repeat(np.arange(1, num_rows + 1), 24*7)[0:num_rows]
    month = pd.date_range(start='2008-01-01 00:00', end='2008-12-31 00:00', freq='1h').month[0:num_rows].to_list()
    year = np.ones(num_rows)
    network_operation.index = pd.MultiIndex.from_arrays([hour, day, week, month, year],
                                            names=['Hour', 'Day', 'Week', 'Month', 'Year'])

    return network_operation

# def create_folium_marker(fig, location):
#     buffer = BytesIO()
#     plt.savefig(buffer, format='png', transparent=True)
#     plt.close(fig)
#     image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
#     html = '<img src="data:image/png;base64,{}">'.format(image_data)
#     marker_feature_group = folium.FeatureGroup()
#
#     folium.Marker(
#         location=location,
#         icon=folium.DivIcon(html=html),
#         draggable=False
#     ).add_to(marker_feature_group)
#     return marker_feature_group

