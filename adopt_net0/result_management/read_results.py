import h5py
import numpy as np
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


def add_values_to_summary(summary_path: Path, component_set: list = None):
    """
    Collect values of input cost parameters and relevant variables from HDF5 files and add them to the summary Excel file.

    Args:
        summary_path (Path or str): Path to the summary Excel file.
        component_set (list, optional): List of components to extract parameters and variables from.
            Defaults to ["Technologies", "Networks", "Import", "Export"].
    """

    if component_set is None:
        component_set = ["Technologies", "Networks", "Import", "Export"]

    summary_results = pd.read_excel(summary_path)

    # paths to results
    paths = {}
    for timestamp in summary_results["time_stamp"].unique():
        paths[timestamp] = list(
            summary_results.loc[
                summary_results["time_stamp"] == timestamp, "time_stamp"
            ].values
        )

    # dicts to store data
    output_dict = {}

    # Extract data from h5 files

    for case in paths:
        path = Path(case)
        hdf_file_path = path / "optimization_results.h5"
        output_dict[case] = {}
        if hdf_file_path.exists():
            with h5py.File(hdf_file_path, "r") as hdf_file:

                if "Technologies" in component_set:
                    df = extract_datasets_from_h5group(hdf_file["design/nodes"])
                    for period in df.columns.levels[0]:
                        for node in df.columns.levels[1]:
                            for tec in df.columns.levels[2]:
                                parameters = [
                                    "size",
                                    "capex_tot",
                                    "para_unitCAPEX",
                                    "para_fixCAPEX",
                                ]
                                for para in parameters:
                                    if (period, node, tec, para) in df.columns:
                                        tec_output = df[period, node, tec, para].iloc[0]
                                        output_name = f"{period}/{node}/{tec}/{para}"
                                        if output_name not in output_dict[case]:
                                            output_dict[case][output_name] = tec_output

                if "Networks" in component_set:
                    df = extract_datasets_from_h5group(hdf_file["design/networks"])
                    if not df.empty:
                        for period in df.columns.levels[0]:
                            for netw in df.columns.levels[1]:
                                for arc in df.columns.levels[2]:
                                    parameters = [
                                        "para_capex_gamma1",
                                        "para_capex_gamma2",
                                        "para_capex_gamma3",
                                        "para_capex_gamma4",
                                        "size",
                                        "capex",
                                    ]
                                    for para in parameters:
                                        output_name = f"{period}/{netw}/{arc}/{para}"
                                        arc_output = df[period, netw, arc, para].iloc[0]
                                        if output_name not in output_dict[case]:
                                            output_dict[case][output_name] = arc_output

                if "Import" in component_set:
                    df = extract_datasets_from_h5group(
                        hdf_file["operation/energy_balance"]
                    )
                    for period in df.columns.levels[0]:
                        for node in df[period].columns.levels[0]:
                            cars_at_node = (
                                df[period, node].columns.droplevel([1]).unique()
                            )
                            for car in cars_at_node:
                                parameters = ["import", "import_price"]
                                for para in parameters:
                                    car_output = df[period, node, car, para]
                                    if para == "import":
                                        car_output = sum(car_output)
                                        output_name = (
                                            f"{period}/{node}/{car}/{para}_tot"
                                        )
                                        if output_name not in output_dict[case]:
                                            output_dict[case][output_name] = car_output
                                    elif para == "import_price":
                                        car_output_mean = np.mean(car_output)
                                        car_output_std = np.std(car_output)
                                        output_name_mean = (
                                            f"{period}/{node}/{car}/{para}_mean"
                                        )
                                        output_name_std = (
                                            f"{period}/{node}/{car}/{para}_std"
                                        )
                                        if output_name_mean not in output_dict[case]:
                                            output_dict[case][
                                                output_name_mean
                                            ] = car_output_mean
                                        if output_name_std not in output_dict[case]:
                                            output_dict[case][
                                                output_name_std
                                            ] = car_output_std

                if "Export" in component_set:
                    df = extract_datasets_from_h5group(
                        hdf_file["operation/energy_balance"]
                    )
                    for period in df.columns.levels[0]:
                        for node in df[period].columns.levels[0]:
                            cars_at_node = (
                                df[period, node].columns.droplevel([1]).unique()
                            )
                            for car in cars_at_node:
                                parameters = ["export", "export_price"]
                                for para in parameters:
                                    car_output = df[period, node, car, para]
                                    if para == "export":
                                        car_output = sum(car_output)
                                        output_name = (
                                            f"{period}/{node}/{car}/{para}_tot"
                                        )
                                        if output_name not in output_dict[case]:
                                            output_dict[case][output_name] = car_output
                                    elif para == "export_price":
                                        car_output_mean = np.mean(car_output)
                                        car_output_std = np.std(car_output)
                                        output_name_mean = (
                                            f"{period}/{node}/{car}/{para}_mean"
                                        )
                                        output_name_std = (
                                            f"{period}/{node}/{car}/{para}_std"
                                        )
                                        if output_name_mean not in output_dict[case]:
                                            output_dict[case][
                                                output_name_mean
                                            ] = car_output_mean
                                        if output_name_std not in output_dict[case]:
                                            output_dict[case][
                                                output_name_std
                                            ] = car_output_std

    # Add new columns to summary_results
    output_df = pd.DataFrame(output_dict).T
    summary_results = summary_results.set_index("time_stamp")

    # Check for existing columns and overwrite them
    for col in output_df.columns:
        summary_results[col] = output_df[col]

    # Reset the index to ensure time_stamp is a column
    summary_results = summary_results.reset_index().rename(
        columns={"index": "folder_name"}
    )

    # Save the updated summary_results to the Excel file
    summary_results.to_excel(summary_path, index=False)
