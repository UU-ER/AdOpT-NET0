import h5py
import pandas as pd
from pathlib import Path


def print_h5_tree(file_path: Path | str):
    """
    Function to print the structure of a h5 file

    The structure of a h5 file is a tree structure: the h5 file is the root group, from which all groups stem,
    and datasets are the leaves contained within a group.

    :param Path, str file_path: Path to H5 File
    """
    with h5py.File(file_path, "r") as hdf_file:

        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")

        hdf_file.visititems(print_attrs)


def extract_datasets_from_h5group(
    group: h5py.Group, prefix: tuple = ()
) -> pd.DataFrame:
    """
    Extracts datasets from a group within an h5 file

    Gets all datasets from a group of an h5 file and writes it to a multi-index dataframe

    :param h5py.Group group: of h5 file
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


def extract_dataset_from_h5(dataset: h5py.Dataset) -> list:
    """
    Extracts values from a dataset within an h5 file

    Gets all values of a dataset in an h5 file and writes it to a list.

    :param h5py.Dataset dataset: dataset within an h5 file
    :return: list of all values in a dataset
    :rtype: list
    """
    data = [item.decode("utf-8") for item in dataset]

    return data


def add_values_to_summary(summary_path: Path or str, value_set: list = None):
    """Collects values of input parameters and variables from h5 files and adds them to the summary.

    Args:
        summary_path (Path or str): Path to the summary Excel file.
        value_set (list, optional): Set of values to extract. Defaults to ['tec_capex', 'tec_size'].
    """
    if value_set is None:
        value_set = {"tec_capex", "tec_size"}
        # value_set = {'tec_capex', 'tec_size', 'netw_capex', 'netw_size', 'import_price', 'import_limit', 'total_import',
        #              'export_price', 'export_limit', 'total_export'}

    # Ensure summary_path is a Path object
    if not isinstance(summary_path, Path):
        summary_path = Path(summary_path)

    path_root = summary_path.parent.parent
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
    tec_output_dict = {}

    # Extract data from h5 files
    for case in paths:
        path = Path(case)
        hdf_file_path = path / "optimization_results.h5"
        if hdf_file_path.exists():
            with h5py.File(hdf_file_path, "r") as hdf_file:
                df = extract_datasets_from_h5group(hdf_file["design/nodes"]).sum()
                for period in df.index.levels[0]:
                    for node in df.index.levels[1]:
                        for tec in df.index.levels[2]:
                            parameters = [
                                "size",
                                "capex_tot",
                                "para_unitCAPEX",
                                "para_fixCAPEX",
                            ]
                            for para in parameters:
                                output_name = f"{period}/{node}/{tec}/{para}"
                                try:
                                    tec_output = df.loc[period, node, tec, para]
                                    if case not in tec_output_dict:
                                        tec_output_dict[case] = {}
                                    if output_name not in tec_output_dict[case]:
                                        tec_output_dict[case][output_name] = tec_output
                                except KeyError:
                                    pass

    if any(
        value in value_set for value in ["import_price", "import_limit", "total_import"]
    ):
        # TODO add import, export and networks similarly
        pass
        #     if hdf_file_path.exists():
        #         with h5py.File(hdf_file_path, 'r') as hdf_file:
        #             df = extract_datasets_from_h5group(hdf_file["operation/energy_balance"])
        #         df = df.sum()

    # Add new columns to summary_results
    tec_output_df = pd.DataFrame(tec_output_dict).T
    summary_results = summary_results.set_index("time_stamp")
    summary_results_appended = pd.merge(
        summary_results, tec_output_df, right_index=True, left_index=True
    )

    # Save the updated summary_results to the Excel file
    summary_results_appended.to_excel(summary_path, index=False)
