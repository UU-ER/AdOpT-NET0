import altair as alt
import pandas as pd
import streamlit as st
import h5py

def determine_graph_boundaries(x_values):
    """
    Returns x_min and x_max for a graph that is determined by using a slider
    :param x_values: x_values available
    :return: x_min, x_max
    """
    x_min = st.sidebar.slider(
        "Starting time: ",
        min_value=min(x_values),
        max_value=max(x_values),
    )
    x_max = st.sidebar.slider(
        "Ending time: ",
        min_value=min(x_values),
        max_value=max(x_values),
    )
    return x_min, x_max


def plot_area_chart(df, x_min, x_max):
    df = df[(df.index >= x_min) & (df.index <= x_max)]
    df = df.reset_index()
    df = pd.melt(df, value_vars=df.columns, id_vars=['index'])

    chart = alt.Chart(df).mark_area().encode(
        x='index:Q',
        y='sum(value):Q',
        color="variable:N").configure_legend(orient='bottom')
    return chart


def plot_line_chart(df, x_min, x_max):
    df = df[(df.index >= x_min) & (df.index <= x_max)]
    df = df.reset_index()
    df = pd.melt(df, value_vars=df.columns, id_vars=['index'])

    chart = alt.Chart(df).mark_line().encode(
        x='index:Q',
        y='value:Q',
        color="variable:N").configure_legend(orient='bottom')
    return chart


def extract_datasets_from_h5_group(group, prefix=()):
    """
    Gets all datasets from a group of an h5 file and writes it to a multi-index dataframe

    :param group: group of h5 file
    :return: dataframe containing all datasets in group
    """
    data = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            data.update(extract_datasets_from_h5_group(value, prefix + (key,)))
        elif isinstance(value, h5py.Dataset):
            if value.shape == ():
                data[prefix + (key,)] = [value[()]]
            else:
                data[prefix + (key,)] = value[:]

    df = pd.DataFrame(data)

    return df


def extract_datasets_from_h5_dataset(dataset):
    """
    Gets dataset from an h5 file

    :param group: group of h5 file
    :return: dataframe containing all datasets in group
    """
    data = [item.decode('utf-8') for item in dataset]

    return data


def export_csv(df, label, filename):
    """
    Makes a button on the side bar that allows for csv export
    :param df: dataframe to export
    :param label: label of button
    :param filename: filename to export
    :return:
    """
    excel_buffer = df.to_csv(index=False)
    st.sidebar.download_button(
        label=label,
        data=excel_buffer,
        file_name=filename,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
