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
        summary_results = {}
        summary_results['baseline_demand'] = pd.read_excel('//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/baseline_demand/Summary_Plotting_appended.xlsx', index_col=0)
        summary_results['low_demand'] = pd.read_excel('//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/low_demand/Summary_Plotting_appended.xlsx', index_col=0)
        # summary_results = pd.read_excel('C:/Users/6574114/OneDrive - Universiteit Utrecht/PhD Jan/Papers/DOSTA - HydrogenOffshore/Summary.xlsx')

        # Normalization
        baseline_costs = summary_results['baseline_demand'].loc[summary_results['baseline_demand']['Case'] == 'Baseline', 'total_costs'].values[0]
        baseline_emissions = summary_results['baseline_demand'].loc[summary_results['baseline_demand']['Case'] == 'Baseline', 'net_emissions'].values[0]

        for key, result in summary_results.items():
            result['normalized_costs'] = result['total_costs'] / baseline_costs
            result['normalized_emissions'] = result['net_emissions'] / baseline_emissions

        st.session_state['summary_results'] = summary_results

    summary_results = st.session_state['summary_results']

    show_demand = st.selectbox('Select results to show: ', ['Normal Demand', 'Low Demand'])
    if show_demand == 'Normal Demand':
        plot_main = summary_results['baseline_demand']
        plot_side = summary_results['low_demand']
    elif show_demand == 'Low Demand':
        plot_main = summary_results['low_demand']
        plot_side = summary_results['baseline_demand']

    st.header('Pareto Chart (full)')

    col1, col2, col3 = st.columns(3)

    with col1:
        min_x = st.number_input("Min X", value=0.0, step= 0.01)
        min_y = st.number_input("Min Y", value=0.0, step= 0.01)

    with col2:
        max_x = st.number_input("Max X", value=max(plot_main['normalized_costs']), step= 0.01)
        max_y = st.number_input("Max Y", value=max(plot_main['normalized_emissions']), step= 0.01)

    chart_main = alt.Chart(plot_main).mark_line(point=True).encode(
        x=alt.X('normalized_costs', scale=alt.Scale(domain=[min_x, max_x])),
        y=alt.Y('normalized_emissions').scale(domain=[min_y, max_y]),
        color='Case'
    ).properties(
        width=600,
        height=400
    ).interactive()
    chart_side = alt.Chart(plot_side).mark_line(point=True).encode(
        x=alt.X('normalized_costs', scale=alt.Scale(domain=[min_x, max_x])),
        y=alt.Y('normalized_emissions').scale(domain=[min_y, max_y]),
        color=alt.value('gray')  # Set color to grayscale
    # = alt.Color('Case').scale(scheme="greys")
    ).properties(
        width=600,
        height=400
    ).interactive()
    st.altair_chart(chart_main + chart_side, theme="streamlit")


    st.header('Pareto Chart (zoomed to min cost point)')


    chart_main = alt.Chart(plot_main[plot_main['pareto_point'] == 0]).mark_circle(size=100).encode(
        x=alt.X('normalized_costs', scale=alt.Scale(domain=[min_x, max_x])),
        y=alt.Y('normalized_emissions').scale(domain=[min_y, max_y]),
        color='Case'
    ).properties(
        width=600,
        height=400
    ).interactive()
    chart_side = alt.Chart(plot_side[plot_side['pareto_point'] == 0]).mark_circle(size=100).encode(
        x=alt.X('normalized_costs', scale=alt.Scale(domain=[min_x, max_x])),
        y=alt.Y('normalized_emissions').scale(domain=[min_y, max_y]),
        color=alt.value('gray')  # Set color to grayscale
    ).properties(
        width=600,
        height=400
    ).interactive()
    st.altair_chart(chart_main + chart_side, theme="streamlit")

    # Compare cases
    case_sel1, case_sel2 = st.columns(2)
    with case_sel1:
        case1 = st.selectbox('Select Case 1: ', plot_main['Case'].unique())
    with case_sel2:
        case2 = st.selectbox('Select Case 2: ', plot_main['Case'].unique())

    # Curtailment
    summary_results_filtered = plot_main[plot_main['Case'].isin([case1, case2])]

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
        summary_results_case = plot_main[plot_main['Case'].isin([case])].drop_duplicates(subset=['normalized_costs'], keep='last')

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



