import h5py
import pandas as pd
import streamlit as st

@st.cache_data
def aggregate_time(df, level):
    df = df.groupby(level=level).sum()
    df.index.names = ['Timeslice']
    return df

@st.cache_data
def aggregate_spatial_networks(network_operation, level):
    network_operation = network_operation.T
    if level == 'Country':
        network_operation = network_operation.reset_index()
        network_operation = network_operation[network_operation['FromCountry'] != network_operation['ToCountry']]
        network_operation = network_operation.groupby(['Network', 'FromCountry', 'ToCountry']).sum()
        network_operation = network_operation.rename_axis(index={'FromCountry': 'FromNode', 'ToCountry': 'ToNode'})
        network_operation = network_operation.drop(columns=['FromNode', 'ToNode'])
        network_operation = network_operation.reset_index()
        network_operation = network_operation.set_index(['Network', 'Arc_ID', 'Country_ID', 'Variable', 'FromNode', 'ToNode'])
        network_operation.columns = ['Value']
        network_operation = network_operation.reset_index()

    else:
        network_operation.columns = ['Value']
        network_operation = network_operation.reset_index()
        network_operation = network_operation.drop(columns=['FromCountry', 'ToCountry'])

    return network_operation

@st.cache_data
def aggregate_spatial_balance(balance, level):
    balance = balance.T.reset_index()
    balance = balance.groupby([level, 'Technology', 'Carrier', 'Variable']).sum()
    if level == 'Country':
        balance = balance.drop(columns=['Node'])
        balance = balance.rename_axis(index={'Country': 'Node'})
    else:
        balance = balance.drop(columns=['Country'])

    balance.columns = ['Value']

    return balance

@st.cache_data
def load_nodes_from_h5_results(path):
    """
    Loads all nodes contained in a results file as a list
    """
    with h5py.File(path) as hdf_file:
        nodes = extract_data_from_h5_dataset(hdf_file["topology/nodes"])

    return nodes


@st.cache_data
def load_carriers_from_h5_results(path):
    """
    Loads all carriers contained in a results file as a list
    """
    with h5py.File(path) as hdf_file:
        carriers = extract_data_from_h5_dataset(hdf_file["topology/carriers"])

    return carriers


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


def extract_data_from_h5_dataset(dataset):
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
    excel_buffer = df.to_csv(index=False, sep=';')
    st.sidebar.download_button(
        label=label,
        data=excel_buffer,
        file_name=filename,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
