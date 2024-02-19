import pandas as pd
import streamlit as st
import altair as alt
from .utilities import *
import h5py

def show_main_results():
    # Session states
    if 'nodes' not in st.session_state:
        st.session_state['nodes'] = set()
    if 'carriers' not in st.session_state:
        st.session_state['carriers'] = set()
    if 'topology_loaded' not in st.session_state:
        st.session_state['topology_loaded'] = 0
    if 'summary_results' not in st.session_state:
        st.session_state['summary_results'] = None

    if st.session_state['summary_results'] is None:
        # summary_results = pd.read_excel('//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/baseline_demand/Summary_Plotting_appended.xlsx', index_col=0)
        summary_results = pd.read_excel('C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/Summary.xlsx')

        # Normalization
        baseline_costs = summary_results.loc[summary_results['Case'] == 'Baseline', 'total_costs'].values[0]
        baseline_emissions = summary_results.loc[summary_results['Case'] == 'Baseline', 'net_emissions'].values[0]
        summary_results['normalized_costs'] = summary_results['total_costs'] / baseline_costs
        summary_results['normalized_emissions'] = summary_results['net_emissions'] / baseline_emissions
        st.session_state['summary_results'] = summary_results

    summary_results = st.session_state['summary_results']

    st.header('Pareto Chart (full)')

    chart = alt.Chart(summary_results).mark_line(point=True).encode(
        x=alt.X('normalized_costs').scale(zero=False) ,
        y=alt.Y('normalized_emissions').scale(zero=False) ,
        color='Case'
    ).properties(
        width=600,
        height=400
    ).interactive()
    st.altair_chart(chart, theme="streamlit")


    st.header('Pareto Chart (zoomed to min cost point)')
    chart = alt.Chart(summary_results[summary_results['pareto_point'] == 0]).mark_circle().encode(
        x=alt.X('normalized_costs').scale(zero=False) ,
        y=alt.Y('normalized_emissions').scale(zero=False) ,
        color='Case'
    ).properties(
        width=600,
        height=400
    ).interactive()
    st.altair_chart(chart, theme="streamlit")

    # Compare cases
    case_sel1, case_sel2 = st.columns(2)
    with case_sel1:
        case1 = st.selectbox('Select Case 1: ', summary_results['Case'].unique())
    with case_sel2:
        case2 = st.selectbox('Select Case 2: ', summary_results['Case'].unique())

    # Curtailment
    summary_results_filtered = summary_results[summary_results['Case'].isin([case1, case2])]

    st.header('Electricity Imports')
    chart = alt.Chart(summary_results_filtered).mark_line(point=True).encode(
        x='normalized_costs',
        y='import_total',
        color='Case'
    ).properties(
        width=600,
        height=400
    ).interactive()
    st.altair_chart(chart, theme="streamlit")

    st.header('Electricity Curtailed')
    chart = alt.Chart(summary_results_filtered).mark_line(point=True).encode(
        x='normalized_costs',
        y='curtailment_total',
        color='Case'
    ).properties(
        width=600,
        height=400
    ).interactive()
    st.altair_chart(chart, theme="streamlit")

    st.header('Electricity Generation by source')
    for case in [case1, case2]:
        st.subheader(case)
        summary_results_case = summary_results[summary_results['Case'].isin([case])]

        # generation_by_source = summary_results_case[['normalized_costs',
        #                                                  'import_total',
        #                                                  'PowerPlant_Coal_existing',
        #                                                  'PowerPlant_Gas_existing',
        #                                                  'PowerPlant_Nuclear_existing',
        #                                                  'PowerPlant_Oil_existing']]
        generation_by_source = summary_results_case[['normalized_costs',
                                                         'import_total',
                                                         'generic_production_total',
                                                         'PowerPlant_Coal_existing',
                                                         'PowerPlant_Gas_existing',
                                                         'PowerPlant_Nuclear_existing',
                                                         'PowerPlant_Oil_existing']]
        generation_by_source_melted = generation_by_source.melt(id_vars = 'normalized_costs')

        if case == 'Baseline':
            chart = alt.Chart(generation_by_source_melted).mark_bar().encode(
                x='normalized_costs',
                y='sum(value)',
                color='variable'
            ).properties(
                width=600,
                height=400
            ).interactive()
            st.altair_chart(chart, theme="streamlit")
        else:
            chart = alt.Chart(generation_by_source_melted).mark_area().encode(
                x='normalized_costs',
                y='sum(value)',
                color='variable'
            ).properties(
                width=600,
                height=400
            ).interactive()
            st.altair_chart(chart, theme="streamlit")



