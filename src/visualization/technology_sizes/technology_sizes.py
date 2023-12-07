import altair as alt
import pandas as pd
import streamlit as st

def tec_sizes(tec_size_data):
    # Multi-select box for filtering technologies
    tec_size_data['total_cost'] = tec_size_data['capex'] + tec_size_data['opex_variable'] + tec_size_data['opex_fixed']
    selected_technologies = st.multiselect('Select Technologies to Filter', tec_size_data['technology'].unique(),
                                           default=tec_size_data['technology'].unique())

    # Filter the DataFrame based on selected technologies
    filtered_df = tec_size_data[tec_size_data['technology'].isin(selected_technologies)]

    st.header("Technology Size")
    chart = alt.Chart(filtered_df).mark_bar().encode(
        x='node:N',
        y='sum(size):Q',
        color='technology:N',
        tooltip=['node', 'technology', 'size']
    ).interactive()

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

    st.header("Technology Cost")
    chart = alt.Chart(filtered_df).mark_bar().encode(
        x='node:N',
        y='sum(total_cost):Q',
        color='technology:N',
        tooltip=['node', 'technology', 'total_cost']
    ).interactive()

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)