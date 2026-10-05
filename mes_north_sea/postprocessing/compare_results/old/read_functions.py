import h5py
import numpy as np
import pandas as pd

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


def add_time_steps_to_df(df):
    """
    Adds time index to a df
    """
    hour = df.index.to_list()
    num_rows = len(hour)
    day = np.repeat(np.arange(1, num_rows + 1), 24)[0:num_rows]
    week = np.repeat(np.arange(1, num_rows + 1), 24 * 7)[0:num_rows]
    month = pd.date_range(start='2008-01-01 00:00', end='2008-12-31 00:00', freq='1h').month[0:num_rows].to_list()
    year = np.ones(num_rows)

    df.index = pd.MultiIndex.from_arrays([hour, day, week, month, year],
                                         names=['Hour', 'Day', 'Week', 'Month', 'Year'])

    return df

def read_energy_balance(path_h5):
    """
    Reads energybalance
    """
    with h5py.File(path_h5, 'r') as hdf_file:
        df_bal = extract_datasets_from_h5_group(hdf_file["operation/energy_balance"])


    df_bal = df_bal.rename_axis(columns=['Node', 'Carrier', 'Variable'])
    df_bal = add_time_steps_to_df(df_bal)

    return df_bal

def read_technology_operation(path_h5):
    """
    Reads technology operation
    """
    with h5py.File(path_h5, 'r') as hdf_file:
        df_ope = extract_datasets_from_h5_group(hdf_file["operation/technology_operation"])

    df_ope = df_ope.rename_axis(columns=['Node', 'Technology', 'Variable'])
    df_ope = add_time_steps_to_df(df_ope)

    return df_ope

def read_technology_design(path_h5):
    """
    Reads technology design
    """
    with h5py.File(path_h5, 'r') as hdf_file:
        technology_design = extract_datasets_from_h5_group(hdf_file["design/nodes"])
        technology_design = pd.melt(technology_design)
        technology_design.columns = ['Node', 'Technology', 'Variable', 'Value']

    return technology_design


def read_networks(path_h5):
    with h5py.File(path_h5, 'r') as hdf_file:
        network_design = extract_datasets_from_h5_group(hdf_file["design/networks"])

    network_design = network_design.melt()
    network_design.columns = ['Network', 'Arc_ID', 'Variable', 'Value']
    network_design = network_design.pivot(columns='Variable', index=['Arc_ID', 'Network'], values='Value')
    network_design['FromNode'] = network_design['fromNode'].str.decode('utf-8')
    network_design['ToNode'] = network_design['toNode'].str.decode('utf-8')
    network_design.drop(columns=['fromNode', 'toNode', 'network'], inplace=True)
    network_design = network_design.reset_index()
    # arc_ids = network_design[['Arc_ID', 'FromNode', 'ToNode']]
    #
    # with h5py.File(path_h5, 'r') as hdf_file:
    #     network_operation = extract_datasets_from_h5_group(hdf_file["operation/networks"])
    #
    # network_operation.columns.names = ['Network', 'Arc_ID', 'Variable']
    #
    # network_operation = network_operation.T.reset_index()
    # network_operation = pd.merge(network_operation, arc_ids.drop_duplicates(subset=['Arc_ID']), how='inner', left_on='Arc_ID', right_on='Arc_ID')
    # network_operation = network_operation.set_index(['Network', 'Arc_ID', 'Variable', 'FromNode', 'ToNode']).T
    #
    # network_operation = add_time_steps_to_df(network_operation)

    return (network_design
            # , network_operation
            )