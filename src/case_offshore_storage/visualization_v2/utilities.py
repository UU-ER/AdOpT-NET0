import h5py
import pandas as pd
import streamlit as st

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
