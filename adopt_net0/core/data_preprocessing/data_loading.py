import warnings
import pandas as pd
import numpy as np
import shutil
import os
import json
from pathlib import Path


def fill_carrier_data(
    folder_path: str | Path,
    value_or_data: float | pd.DataFrame,
    columns: list = [],
    carriers: list = [],
    nodes: list = [],
    investment_periods: list = None,
):
    """
    Updates carrier data for a time series based on a provided value or DataFrame and writes it to file.

    Allows you to update Demand, Import limit, Export limit, Import price,
    Export price, Import emission factor, Export emission factor and/or Generic production.

    :param str folder_path: Path to the folder containing the case study data
    :param float | pd.DataFrame value_or_data: A float value to be applied or a DataFrame containing the new values for the carrier data
    :param list columns: Name of the columns that need to be changed
    :param list investment_periods: Name of investment periods to be changed
    :param list nodes: Name of the nodes that need to be changed
    :param list carriers: Name of the carriers that need to be changed
    """
    # Convert to Path
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    # Read the topology json file
    json_file_path = folder_path / "Topology.json"
    with open(json_file_path, "r") as json_file:
        topology = json.load(json_file)

    # Define options
    column_options = [
        "Demand",
        "Import limit",
        "Export limit",
        "Import price",
        "Export price",
        "Import emission factor",
        "Export emission factor",
        "Generic production",
    ]

    for period in (
        investment_periods if investment_periods else topology["investment_periods"]
    ):
        for node_name in nodes if nodes else topology["nodes"]:
            for car in carriers if carriers else topology["carriers"]:

                # Write data to CSV file
                output_folder = (
                    folder_path / period / "node_data" / node_name / "carrier_data"
                )
                filename = car + ".csv"
                output_file = output_folder / filename
                existing_data = pd.read_csv(output_file, sep=";")

                # Fill in existing data with either a constant value or data from the provided DataFrame
                for column in columns if columns else column_options:
                    if isinstance(value_or_data, pd.DataFrame):
                        if column in value_or_data.columns:
                            existing_data[column] = value_or_data[column].values
                        else:
                            raise ValueError(
                                f"Column {column} not found in the provided DataFrame"
                            )
                    else:
                        existing_data[column] = value_or_data * np.ones(
                            len(existing_data)
                        )

                # Save the updated data back to the CSV file
                output_file.parent.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure directory exists
                existing_data.to_csv(output_file, index=False, sep=";")


def fill_carrier_pressure_data(
    folder_path: str | Path,
    pressure_value_bar: float,
    connection: list = [],
    carriers: list = [],
    nodes: list = [],
    investment_periods: list = None,
):
    """
    Updates carrier pressure data for exchange connections for a time series
    based on a provided value and writes it to file.

    Allows you to update Demand pressure, Export pressure and Import pressure.

    :param str folder_path: Path to the folder containing the case study data
    :param float pressure_value_bar: A float value to be applied containing the values of the carrier pressure data
    :param list connection: Name of the connection that need to be changed
    :param list carriers: Name of the carriers that need to be changed
    :param list nodes: Name of the nodes that need to be changed
    :param list investment_periods: Name of investment periods to be changed
    """

    # Convert to Path
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    # Read the Configuration json file
    json_file_path = folder_path / "ConfigModel.json"
    with open(json_file_path, "r") as json_file:
        configuration = json.load(json_file)

    # Read the topology json file
    json_file_path = folder_path / "Topology.json"
    with open(json_file_path, "r") as json_file:
        topology = json.load(json_file)

    if configuration["performance"]["pressure"]["pressure_on"]["value"] == 0:
        warnings.warn(
            "Pressure configuration is turned off and pressure information will not be used"
        )
        return

    for car in carriers:
        if (
            car
            not in configuration["performance"]["pressure"]["pressure_carriers"][
                "value"
            ]
        ):
            raise ValueError(
                f"Carrier '{car} is not configured for pressure information. Change it in configuration"
            )

    for period in (
        investment_periods if investment_periods else topology["investment_periods"]
    ):
        for node_name in nodes:
            # Path to JSON file
            output_file = (
                folder_path
                / period
                / "node_data"
                / node_name
                / "carrier_data"
                / "PressureExchangeData.json"
            )

            for car in carriers:
                # Load existing data if the file exists
                with open(output_file, "r") as f:
                    existing_data = json.load(f)

                for conn in connection:
                    existing_data[car][conn]["value"] = pressure_value_bar

            # Save updated JSON file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(existing_data, f, indent=4)


def copy_technology_data(folder_path: str | Path, tec_data_path: str | Path = None):
    """
    Copies technology JSON files to the node folder for each node and investment period.

    This function reads the topology JSON file to determine the existing and new technologies at each node for
    each investment period. It then searches for the corresponding JSON files in the specified `tec_data_path`
    folder (and its subfolders) using the technology names and copies them to the output folder.

    :param str | Path folder_path: Path to the folder containing the case study data.
    :param str | Path tec_data_path: Path to the folder containing the technology data.
    """
    # Convert to Path
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    if tec_data_path is None:
        tec_data_path = Path(
            os.path.join(
                os.path.dirname(__file__) + "/../database/templates/technology_data"
            )
        )
    else:
        if isinstance(tec_data_path, str):
            tec_data_path = Path(tec_data_path)

    # Reads the topology JSON file
    json_file_path = folder_path / "Topology.json"
    with open(json_file_path, "r") as json_file:
        topology = json.load(json_file)

    for period in topology["investment_periods"]:
        for node_name in topology["nodes"]:
            # Read the JSON technology file
            json_tec_file_path = (
                folder_path / period / "node_data" / node_name / "Technologies.json"
            )
            with open(json_tec_file_path, "r") as json_tec_file:
                json_tec = json.load(json_tec_file)
            tecs_at_node = list(json_tec["existing"].keys()) + json_tec["new"]

            output_folder = (
                folder_path / period / "node_data" / node_name / "technology_data"
            )
            # Copy JSON files corresponding to technology names to output folder
            for tec_name in tecs_at_node:
                _copy_data(tec_data_path, tec_name, output_folder)


def copy_network_data(folder_path: str | Path, ntw_data_path: str | Path = None):
    """
    Copies network JSON files to the network_data folder for each investment period.

    This function reads the topology JSON file to determine the existing and new networks for
    each investment period. It then searches for the corresponding JSON files in the specified `ntw_data_path`
    folder (and its subfolders) using the network names and copies them to folder_path.

    :param str | Path folder_path: Path to the folder containing the case study data.
    :param str | Path ntw_data_path: Path to the folder containing the network data (if left
    empty, standard folder is used).
    :return: None
    """
    # Convert to Path
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    if ntw_data_path is None:
        ntw_data_path = Path(
            os.path.join(
                os.path.dirname(__file__) + "/../database/templates/network_data"
            )
        )
    else:
        if isinstance(ntw_data_path, str):
            ntw_data_path = Path(ntw_data_path)

    # Reads the topology JSON file
    json_file_path = folder_path / "Topology.json"
    with open(json_file_path, "r") as json_file:
        topology = json.load(json_file)

    for period in topology["investment_periods"]:
        # Read the JSON network file
        json_ntw_file_path = folder_path / period / "Networks.json"
        with open(json_ntw_file_path, "r") as json_ntw_file:
            json_ntw = json.load(json_ntw_file)
        ntws_at_node = json_ntw["existing"] + json_ntw["new"]

        output_folder = folder_path / period / "network_data"
        # Copy JSON files corresponding to technology names to output folder
        for ntw_name in ntws_at_node:
            _copy_data(ntw_data_path, ntw_name, output_folder)


def copy_compressor_data(folder_path: str | Path, compr_data_path: str | Path = None):
    """
    Copies compressor JSON files to the compressor_data folder for each investment period.

    This function reads the topology JSON file to determine the existing and new compressors for
    each investment period. It then searches for the corresponding JSON files in the specified `compr_data_path`
    folder (and its subfolders) using the compressor names and copies them to folder_path.

    :param str | Path folder_path: Path to the folder containing the case study data.
    :param str | Path compr_data_path: Path to the folder containing the compressors data (if left
    empty, standard folder is used).
    :return: None
    """
    config_file_path = folder_path / "ConfigModel.json"
    with open(config_file_path, "r") as json_file:
        config = json.load(json_file)
        if config["performance"]["pressure"]["pressure_on"]["value"] == 0:
            return
        else:
            # Convert to Path
            if isinstance(folder_path, str):
                folder_path = Path(folder_path)

            if compr_data_path is None:
                compr_data_path = Path(
                    os.path.join(
                        os.path.dirname(__file__)
                        + "/../database/templates/compressor_data"
                    )
                )
            else:
                if isinstance(compr_data_path, str):
                    compr_data_path = Path(compr_data_path)

            # Reads the topology JSON file
            json_file_path = folder_path / "Topology.json"
            with open(json_file_path, "r") as json_file:
                topology = json.load(json_file)

            for period in topology["investment_periods"]:
                # Read the JSON compressor file
                output_folder = folder_path / period / "compressor_data"
                # Copy JSON files corresponding to compressor names to output folder
                for compr_name in config["performance"]["pressure"][
                    "pressure_carriers"
                ]["value"]:
                    _copy_data(compr_data_path, compr_name, output_folder)


def find_json_path(data_path: str | Path, name: str) -> Path | None:
    """
    Search for a JSON file with the given technology name in the specified path and its subfolders.

    :param str data_path: Path to the folder containing technology JSON files.
    :param str name: Name of the technology.
    :return: Path to the JSON file if found, otherwise None.
    """
    for root, dirs, files in os.walk(data_path.resolve()):
        for file in files:
            if file.lower() == f"{name.lower()}.json":
                return Path(root) / Path(file)


def _copy_data(path, json_name, output_folder):
    """
    Finds the JSON files and copies it to the desired output folder.

    This function finds the JSON file of the component, and it copies it to the output folder.

    :param str | Path path: Path to the folder containing the case study data.
    :param str | Component name json_name: Name of the JSON file.
    :param str | Path output_folder: Path to the folder containing the technology data.
    """
    component_json_path = find_json_path(path, json_name)
    if component_json_path:
        shutil.copy(component_json_path, output_folder)
    else:
        warnings.warn(f"{json_name} not found")
