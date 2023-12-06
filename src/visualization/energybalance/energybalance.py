import streamlit as st
import altair as alt
import pandas as pd


def energybalance(energybalance, x_min, x_max):
    st.title("Energy Balance per Node")

    # Select carrier
    selected_carrier = st.selectbox('Select a carrier:', energybalance.keys())
    carrier = energybalance[selected_carrier]

    # Filter the DataFrame based on selected x-limits
    carrier = carrier[(carrier['Timestep'] >= x_min) & (carrier['Timestep'] <= x_max)]

    # Plot positive/negative values
    positive_variables = ['Timestep', 'Generic_production', 'Technology_outputs', 'Network_inflow', 'Import']
    positive_values = carrier[positive_variables]
    positive_values = pd.melt(positive_values, value_vars=positive_variables, id_vars=['Timestep'])

    negative_variables = ['Timestep', 'Demand', 'Technology_inputs', 'Network_outflow', 'Export']
    negative_values = carrier[negative_variables]
    negative_values = pd.melt(negative_values, value_vars=negative_variables, id_vars=['Timestep'])

    st.header("Supply")
    chart = alt.Chart(positive_values).mark_area().encode(
        x='Timestep:T',
        y='value:Q',
        color="variable:N")
    st.altair_chart(chart, theme="streamlit", use_container_width=True)

    st.header("Demand")
    chart = alt.Chart(negative_values).mark_area().encode(
        x='Timestep:T',
        y='value:Q',
        color="variable:N")
    st.altair_chart(chart, theme="streamlit", use_container_width=True)
