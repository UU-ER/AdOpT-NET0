import altair as alt
import pandas as pd
import streamlit as st


def tec_operation(tec_data, x_min, x_max):
    st.title("Technology Operation")
    tec = st.selectbox('Select a technology:', tec_data.keys())
    tec_data = tec_data[tec]

    # Filter the DataFrame based on selected x-limits
    tec_data = tec_data[(tec_data['Timestep'] >= x_min) & (tec_data['Timestep'] <= x_max)]

    st.header("Input")
    variables = [col for col in tec_data.columns if col.startswith('input')]
    variables.append('Timestep')
    values = tec_data[variables]
    values = pd.melt(values, value_vars=values, id_vars=['Timestep'])
    chart = alt.Chart(values).mark_area().encode(
        x='Timestep:T',
        y='value:Q',
        color="variable:N")
    st.altair_chart(chart, theme="streamlit", use_container_width=True)

    st.header("Output")
    variables = [col for col in tec_data.columns if col.startswith('output')]
    variables.append('Timestep')
    values = tec_data[variables]
    values = pd.melt(values, value_vars=values, id_vars=['Timestep'])
    chart = alt.Chart(values).mark_area().encode(
        x='Timestep:T',
        y='value:Q',
        color="variable:N")
    st.altair_chart(chart, theme="streamlit", use_container_width=True)

    st.header("Storage Level")
    variables = [col for col in tec_data.columns if col.startswith('storagelevel')]
    if len(variables) >= 1:
        variables.append('Timestep')
        values = tec_data[variables]
        values = pd.melt(values, value_vars=values, id_vars=['Timestep'])
        chart = alt.Chart(values).mark_area().encode(
            x='Timestep:T',
            y='value:Q',
            color="variable:N")
        st.altair_chart(chart, theme="streamlit", use_container_width=True)
