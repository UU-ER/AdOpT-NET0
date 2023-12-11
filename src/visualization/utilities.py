import os
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


def get_boundaries_date(dict):
    min_value = None
    max_value = None
    for run in dict:
        for key, df in dict[run].items():
            # Update min_value and max_value based on the current DataFrame
            current_min = df['Timestep'].min()
            current_max = df['Timestep'].max()

            if min_value is None or current_min < min_value:
                min_value = current_min

            if max_value is None or current_max > max_value:
                max_value = current_max
    return min_value, max_value


def determine_graph_boundaries(dict):
    # Determine plotted daterange
    min_date, max_date = get_boundaries_date(dict)
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
        value=datetime.fromtimestamp(max_date.timestamp()),
        format="DD.MM, HH",
    )
    return x_min, x_max


def select_node(path, nr_pages):
    node_path = {}
    nodes = {}
    selected_node = {}

    for i in [1, len(path)]:
        node_path[i] = Path.joinpath(path[i], 'Nodes')
        nodes[i] = [f.name for f in os.scandir(node_path[i]) if f.is_dir()]

    selected_node[1] = st.sidebar.selectbox('Select a node:', nodes[1], key="node_key1")
    if len(nr_pages) == 2:
        selected_node[2] = st.sidebar.selectbox('Select a node for second result:', nodes[2], key="node_key2")

    return selected_node, node_path


def read_time_series(path):
    data = pd.read_excel(path, sheet_name=None, index_col=0)
    for carrier in data:
        data[carrier]['Timestep'] = [datetime(2030, 1, 1, 0, 0, 0) + timedelta(hours=hour) for
                                                   hour in data[carrier].index]
    return data


def plot_area_chart(df, x_min, x_max):
    df = df[(df['Timestep'] >= x_min) & (df['Timestep'] <= x_max)]
    df = pd.melt(df, value_vars=df.columns, id_vars=['Timestep'])
    chart = alt.Chart(df).mark_area().encode(
        x='Timestep:T',
        y='value:Q',
        color="variable:N").configure_legend(orient='bottom')
    return chart
