import json

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


def extract_datasets_from_h5group(group, prefix: tuple = ()) -> dict:
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

    return data


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
                    data = extract_datasets_from_h5group(hdf_file["design/nodes"])
                    df = pd.DataFrame(data)
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
                    data = extract_datasets_from_h5group(hdf_file["design/networks"])
                    df = pd.DataFrame(data)

                    parameters = [
                        "para_capex_gamma1",
                        "para_capex_gamma2",
                        "para_capex_gamma3",
                        "para_capex_gamma4",
                        "size",
                        "capex",
                    ]

                    df_filtered = df.loc[
                        :, df.columns.get_level_values(3).isin(parameters)
                    ].T
                    for _, row in df_filtered.iterrows():
                        output_dict[case]["/".join(row.name)] = row.values[0]
                    #
                    # if not df.empty:
                    #     for period in df.columns.levels[0]:
                    #         period_data = df[period]
                    #         for netw in period_data.columns.levels[0]:
                    #             netw_data = period_data[netw]
                    #             for arc in netw_data.columns.levels[0]:
                    #                 print(netw_data)
                    #                 print(arc)
                    #                 arc_data = netw_data[arc]
                    #
                    #                 for para in parameters:
                    #                     output_name = f"{period}/{netw}/{arc}/{para}"
                    #                     arc_output = arc_data[para].iloc[0]
                    #                     if output_name not in output_dict[case]:
                    #                         output_dict[case][output_name] = arc_output

                if "Import" in component_set:
                    data = extract_datasets_from_h5group(
                        hdf_file["operation/energy_balance"]
                    )
                    df = pd.DataFrame(data)
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
                    data = extract_datasets_from_h5group(
                        hdf_file["operation/energy_balance"]
                    )
                    df = pd.DataFrame(data)
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


def add_vintage_annualization_to_summary(
    summary_path: Path, casestudy_path: Path, intervals: list
):
    """
        Add the annualized capex of carried-over vintages to the summary Excel file.

        Sums the ``vintage_capex`` entries of each interval's Technologies.json and
        Networks.json (present only while the economic lifetime is running) and adds
        the columns ``cost_annualization_vintage_tecs``, ``cost_annualization_vintage_netw``,
        ``cost_annualization`` and ``total_cost_with_vintage_annualization``.
    5    The same values are also written to each interval's ``optimization_results.h5``
        under the ``summary`` group.

        Args:
            summary_path (Path or str): Path to the summary Excel file, one row per
                interval, in the same order as ``intervals``.
            casestudy_path (Path or str): Path to the case study folder containing
                the ``Case_{interval}`` folders.
            intervals (list): Interval names, in the order they were solved.
    """
    summary_results = pd.read_excel(summary_path)
    if len(summary_results) != len(intervals):
        raise ValueError(
            f"Summary file has {len(summary_results)} rows, but {len(intervals)} "
            "intervals were given. One summary row per interval is required."
        )

    casestudy_path = Path(casestudy_path)
    cost_vintage_tecs = []
    cost_vintage_netw = []

    for interval in intervals:
        interval_path = casestudy_path / ("Case_" + interval) / interval

        cost_tecs = 0.0
        node_data_path = interval_path / "node_data"
        if node_data_path.exists():
            for node_dir in sorted(node_data_path.iterdir()):
                tec_json_path = node_dir / "Technologies.json"
                if not tec_json_path.exists():
                    continue
                with open(tec_json_path) as f:
                    json_tec = json.load(f)
                for vintages in json_tec.get("vintage_capex", {}).values():
                    cost_tecs += sum(vintages.values())

        cost_netw = 0.0
        netw_json_path = interval_path / "Networks.json"
        if netw_json_path.exists():
            with open(netw_json_path) as f:
                json_netw = json.load(f)
            for vintages in json_netw.get("vintage_capex", {}).values():
                cost_netw += sum(vintages.values())

        cost_vintage_tecs.append(cost_tecs)
        cost_vintage_netw.append(cost_netw)

    summary_results["cost_annualization_vintage_tecs"] = cost_vintage_tecs
    summary_results["cost_annualization_vintage_netw"] = cost_vintage_netw
    summary_results["cost_annualization"] = (
        summary_results["cost_annualization_vintage_tecs"]
        + summary_results["cost_annualization_vintage_netw"]
    )
    summary_results["total_cost_with_vintage_annualization"] = (
        summary_results["total_cost"] + summary_results["cost_annualization"]
    )

    # Write the new values to each interval's h5 file
    if "time_stamp" in summary_results.columns:
        h5_keys = [
            "cost_annualization_vintage_tecs",
            "cost_annualization_vintage_netw",
            "cost_annualization",
            "total_cost_with_vintage_annualization",
        ]
        for _, row in summary_results.iterrows():
            hdf_file_path = Path(row["time_stamp"]) / "optimization_results.h5"
            if not hdf_file_path.exists():
                continue
            with h5py.File(hdf_file_path, "a") as hdf_file:
                summary = hdf_file["summary"]
                for key in h5_keys:
                    if key in summary:
                        del summary[key]
                    summary.create_dataset(key, data=row[key])

    summary_results.to_excel(summary_path, index=False)
