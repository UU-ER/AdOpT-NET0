import streamlit as st
import altair as alt
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta

def get_boundaries_date(energybalance):
    min_value = None
    max_value = None

    for key, df in energybalance.items():
        # Update min_value and max_value based on the current DataFrame
        current_min = df['Timestep'].min()
        current_max = df['Timestep'].max()

        if min_value is None or current_min < min_value:
            min_value = current_min

        if max_value is None or current_max > max_value:
            max_value = current_max
    return min_value, max_value

# Energybalances
def ebalance(energybalance, x_min, x_max):
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


# Function to render Page 3 content
def tecoperation(tec_data, x_min, x_max):
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


path = Path('//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/20231201/20231201133820_Baseline')
node_path = Path.joinpath(path, 'nodes')
nodes = [f.name for f in os.scandir(node_path) if f.is_dir()]

# Sidebar navigation
page_options = ["Energy Balance at Node", "Technology Operation"]
selected_page = st.sidebar.selectbox("Select graph", page_options)
selected_node = st.sidebar.selectbox('Select a node:', nodes)

# Read data
energybalance = pd.read_excel(Path.joinpath(node_path, selected_node, 'Energybalance.xlsx'), sheet_name=None,
                         index_col=0)
tec_data = pd.read_excel(Path.joinpath(node_path, selected_node, 'TechnologyOperation.xlsx'), sheet_name=None,
                         index_col=0)
for carrier in energybalance:
    energybalance[carrier]['Timestep'] = [datetime(2030, 1, 1, 0, 0, 0) + timedelta(hours=hour) for hour in energybalance[carrier].index]
for tec in tec_data:
    tec_data[tec]['Timestep'] = [datetime(2030, 1, 1, 0, 0, 0) + timedelta(hours=hour) for hour in tec_data[tec].index]

# Determine plotted daterange
min_date, max_date = get_boundaries_date(energybalance)

st.sidebar.text("Select x-axis range:")
x_min = st.sidebar.slider(
    "Starting time: ",
    min_value=datetime.fromtimestamp(min_date.timestamp()),
    max_value=datetime.fromtimestamp(max_date.timestamp()),
    format="DD.MM, HH",
)
x_max = st.sidebar.slider(
    "Ending time: ",
    min_value=datetime.fromtimestamp(min_date.timestamp()),
    max_value=datetime.fromtimestamp(max_date.timestamp()),
    format="DD.MM, HH",
)

# Render the selected page
if selected_page == "Energy Balance at Node":
    ebalance(energybalance, x_min, x_max)
elif selected_page == "Technology Operation":
    tecoperation(tec_data, x_min, x_max)
