import streamlit as st
import altair as alt
from pathlib import Path
import pandas as pd

def compare_national_capacities():
    
    root_load_path = Path('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/reporting/installed_capacities')
    loadpaths = {}
    loadpaths['entsoe'] = Path('entsoe_national.csv')
    loadpaths['ours'] = Path('ours_national.csv')
    loadpaths['pypsa'] = Path('pypsa_national.csv')

    cap_entsoe = pd.read_csv(Path.joinpath(root_load_path, loadpaths['entsoe']), index_col=0)
    cap_ours = pd.read_csv(Path.joinpath(root_load_path, loadpaths['ours']), index_col=0)
    cap_pypsa = pd.read_csv(Path.joinpath(root_load_path, loadpaths['pypsa']), index_col=0)

    cap_national = pd.merge(cap_ours, cap_entsoe, how='outer',
                            right_on=['Country', 'Technology'],
                            left_on=['Country', 'Technology'])
    cap_national = pd.merge(cap_national, cap_pypsa, how='outer',
                            right_on=['Country', 'Technology'],
                            left_on=['Country', 'Technology'])

    cap_national = cap_national.rename(columns={'Capacity TYNDP': 'TYNDP',
                                                'Capacity ERAA': 'ERAA',
                                                'Capacity PyPsa': 'PyPSA',
                                                'Capacity our work': 'This work'})

    cap_entsoe_melted = pd.melt(cap_national, id_vars=['Country', 'Technology'], var_name='Dataset', value_name='Capacity')
    countries = ['UK', 'NO', 'BE', 'DK', 'DE', 'NL', 'BE']
    cap_entsoe_filtered = cap_entsoe_melted[cap_entsoe_melted['Country'].isin(countries)]

    technologies = st.multiselect('Select a technology to filter', cap_entsoe_filtered['Technology'].unique())

    cap_entsoe_filtered = cap_entsoe_filtered[cap_entsoe_filtered['Technology'].isin(technologies)]

    chart = alt.Chart(cap_entsoe_filtered).mark_bar().encode(
        x=alt.Y('Capacity:Q', title='Capacity (MW)'),
        y=alt.X('Dataset:N', title=None),
        color='Technology:N',
        row='Country:N'
    ).interactive()
    st.altair_chart(chart, theme="streamlit")

def show_nodal_capacities():
    root_load_path = Path('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/clean_data/installed_capacities')

    loadpaths = {}
    loadpaths['ours'] = Path('capacities_node.csv')
    cap_ours = pd.read_csv(Path.joinpath(root_load_path, loadpaths['ours']), index_col=0)
    cap_ours = cap_ours.drop(columns=['index'])
    cap_ours = cap_ours.groupby(['Country', 'Node', 'Technology']).sum()
    cap_ours_melted = pd.melt(cap_ours.reset_index(), id_vars=['Country', 'Technology', 'Node'], value_name='Capacity')

    technologies = st.multiselect('Select a technology to filter', cap_ours_melted['Technology'].unique())
    cap_ours_melted = cap_ours_melted[cap_ours_melted['Technology'].isin(technologies)]

    chart = alt.Chart(cap_ours_melted).mark_bar().encode(
        x=alt.Y('Capacity:Q', title='Capacity (MW)'),
        y=alt.X('Node:N', title=None),
        color='Technology:N',
    ).interactive()
    st.altair_chart(chart, theme="streamlit", use_container_width=True)