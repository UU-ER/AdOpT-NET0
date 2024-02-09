import streamlit as st
import altair as alt
import pandas as pd

from utilities import *

def plot_energybalance():
    # Select a carrier
    selected_carrier = st.sidebar.selectbox('Select a carrier:', st.session_state['carriers'])

    # Read data
    energybalance = {}
    for i in st.session_state['path'].keys():
        with h5py.File(st.session_state['path'][i], 'r') as hdf_file:
            df = extract_datasets_from_h5_group(hdf_file["operation/energy_balance"])
            df = df.loc[:, (df.columns.get_level_values(0) == selected_node[i]) & (
                        df.columns.get_level_values(1) == selected_carrier)]
        energybalance[i] = df

    # Export to csv
    if selected_option == 'Show single result':
        export_csv(energybalance[1], 'Download Energybalance as CSV', 'energybalance.csv')

    # Determine graph boundaries
    x_min, x_max = determine_graph_boundaries(energybalance[1].index)

    st.title("Energy Balance per Node")

    st.header("Supply")
    # Multi-select box for filtering series
    series_supply = ['generic_production',
                     'technology_outputs',
                     'network_inflow',
                     'import']
    selected_supply_series = st.multiselect('Select Series to Filter', series_supply,
                                            default=series_supply)
    for i in st.session_state['path'].keys():
        plot_data = energybalance[i].loc[:, energybalance[i].columns.get_level_values(2).isin(selected_supply_series)]
        plot_data.columns = plot_data.columns.get_level_values(2)
        chart = plot_area_chart(plot_data, x_min, x_max)
        st.altair_chart(chart, theme="streamlit", use_container_width=True)

    st.header("Demand")
    # Multi-select box for filtering series
    series_demand = ['demand',
                     'technology_inputs',
                     'network_outflow',
                     'export']
    selected_demand_series = st.multiselect('Select Series to Filter', series_demand,
                                            default=series_demand)
    for i in st.session_state['path'].keys():
        plot_data = energybalance[i].loc[:, energybalance[i].columns.get_level_values(2).isin(selected_demand_series)]
        plot_data.columns = plot_data.columns.get_level_values(2)
        chart = plot_area_chart(plot_data, x_min, x_max)
        st.altair_chart(chart, theme="streamlit", use_container_width=True)


def energybalance_supply(energybalance, x_min, x_max, selected_carrier):
    # Select carrier
    # selected_carrier = st.selectbox('Select a carrier:', energybalance.keys())
    carrier = energybalance[selected_carrier]

    # Filter the DataFrame based on selected x-limits
    carrier = carrier[(carrier['Timestep'] >= x_min) & (carrier['Timestep'] <= x_max)]

    # Plot positive/negative values
    positive_variables = ['Timestep', 'Generic_production', 'Technology_outputs', 'Network_inflow', 'Import']
    positive_values = carrier[positive_variables]
    positive_values = pd.melt(positive_values, value_vars=positive_variables, id_vars=['Timestep'])

    chart = alt.Chart(positive_values).mark_area().encode(
        x='Timestep:T',
        y='value:Q',
        color="variable:N")

    return chart

def energybalance_demand(energybalance, x_min, x_max, selected_carrier):

    # Select carrier
    # selected_carrier = st.selectbox('Select a carrier:', energybalance.keys())
    carrier = energybalance[selected_carrier]

    # Filter the DataFrame based on selected x-limits
    carrier = carrier[(carrier['Timestep'] >= x_min) & (carrier['Timestep'] <= x_max)]

    negative_variables = ['Timestep', 'Demand', 'Technology_inputs', 'Network_outflow', 'Export']
    negative_values = carrier[negative_variables]
    negative_values = pd.melt(negative_values, value_vars=negative_variables, id_vars=['Timestep'])

    chart = alt.Chart(negative_values).mark_area().encode(
        x='Timestep:T',
        y='value:Q',
        color="variable:N")

    return chart