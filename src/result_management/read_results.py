import h5py
import pandas as pd
from pathlib import Path


def print_h5_tree(file_path: Path | str):
    """
    Function to print the structure of a h5 file

    The structure of a h5 file is a tree structure: the h5 file is the root group,
    from which all groups stem, and datasets are the leaves contained within a
    group.

    :param Path, str file_path: Path to H5 File
    """
    with h5py.File(file_path, "r") as hdf_file:

        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")

        hdf_file.visititems(print_attrs)


def extract_datasets_from_h5group(group, prefix: tuple = ()) -> pd.DataFrame:
    """
    Extracts datasets from a group within a h5 file

    Gets all datasets from a group of a h5 file and writes it to a multi-index
    dataframe using a recursive function

    :param group: froup of h5 file
    :param tuple prefix: required to search through the structure of the h5 tree if there are multiple subgroups in the
     group you specified, empty by default meaning it starts searching from the group specified.
    :return: dataframe containing all datasets in group
    :rtype: pd.DataFrame
    """
    data = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            data.update(extract_datasets_from_h5group(value, prefix + (key,)))
        elif isinstance(value, h5py.Dataset):
            if value.shape == ():
                data[prefix + (key,)] = [value[()]]
            else:
                data[prefix + (key,)] = value[:]

    df = pd.DataFrame(data)

    return df


def extract_dataset_from_h5(dataset) -> list:
    """
    Extracts values from a dataset within a h5 file

    Gets all values of a dataset in a h5 file and writes it to a list.

    :param dataset: dataset within a h5 file
    :return: list of all values in a dataset
    :rtype: list
    """
    data = [item.decode("utf-8") for item in dataset]

    return data
