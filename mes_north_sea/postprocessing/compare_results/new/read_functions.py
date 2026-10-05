import h5py
import pandas as pd
import numpy as np

def extract_data_from_h5_dataset(dataset):
    """
    Gets dataset from an h5 file

    :param group: group of h5 file
    :return: dataframe containing all datasets in group
    """
    data = [item.decode("utf-8") for item in dataset]

    return data

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

    return data

def read_energy_balance(path_h5):
    """
    Reads energybalance
    """
    with h5py.File(path_h5, "r") as hdf_file:
        bal = extract_datasets_from_h5_group(hdf_file["operation/energy_balance"])

    return bal


def read_technology_operation(path_h5):
    """
    Reads technology operation
    """
    with h5py.File(path_h5, "r") as hdf_file:
        ope = extract_datasets_from_h5_group(hdf_file["operation/technology_operation"])
    return ope


def read_technology_design(path_h5):
    """
    Reads technology design
    """
    with h5py.File(path_h5, "r") as hdf_file:
        technology_design = extract_datasets_from_h5_group(hdf_file["design/nodes"])

    technology_design = pd.DataFrame(technology_design)
    technology_design = pd.melt(technology_design)
    technology_design.columns = ["Period", "Node", "Technology", "Variable", "Value"]

    return technology_design



def read_networks(path_h5):
    with h5py.File(path_h5, "r") as hdf_file:
        network_design = extract_datasets_from_h5_group(hdf_file["design/networks"])

    network_design = pd.DataFrame(network_design)
    if not network_design.empty:
        network_design = network_design.melt()
        network_design.columns = ["Period", "Network", "Arc_ID", "Variable", "Value"]
        network_design = network_design.pivot(
            columns="Variable", index=["Period", "Arc_ID", "Network"], values="Value"
        )
        network_design["FromNode"] = network_design["fromNode"].str.decode("utf-8")
        network_design["ToNode"] = network_design["toNode"].str.decode("utf-8")
        network_design.drop(columns=["fromNode", "toNode", "network"], inplace=True)
        network_design = network_design.reset_index()
        arc_ids = network_design[["Arc_ID", "FromNode", "ToNode"]]

    # with h5py.File(path_h5, "r") as hdf_file:
    #     network_operation = extract_datasets_from_h5_group(
    #         hdf_file["operation/networks"]
    #     )
    #     # st.text(network_operation)
    #
    # if network_operation:
    #     network_operation = pd.DataFrame(network_operation)
    #
    #     network_operation.columns.names = ["Period", "Network", "Arc_ID", "Variable"]
    #
    #     network_operation = network_operation.T.reset_index()
    #     network_operation = pd.merge(
    #         network_operation,
    #         arc_ids.drop_duplicates(subset=["Arc_ID"]),
    #         how="inner",
    #         left_on="Arc_ID",
    #         right_on="Arc_ID",
    #     )
    #     network_operation = network_operation.set_index(
    #         ["Period", "Network", "Arc_ID", "Variable", "FromNode", "ToNode"]
    #     ).T
    #
    #     network_operation = network_operation.to_dict(orient="list")
    #     ope = {}
    #     for key in network_operation:
    #         ope[key] = np.array(network_operation[key])
    # else:
    #     ope = {}

    return (network_design,
            # ope
            )