import warnings
from pathlib import Path
import dill as pickle
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from types import SimpleNamespace
import pvlib
import os
import json
from .import_data import import_jrc_climate_data


def fill_climate_data_from_api(folder_path, dataset="JRC"):
    """
    Reads in climate data for a full year from a folder containing node data,
    where each node data is stored in a subfolder and node locations are provided in a CSV file named NodeLocations.csv

    :param str folder_path: Path to the folder containing node data and NodeLocations.csv
    :param str dataset: Dataset to import from, can be JRC (only onshore) or ERA5 (global)
    :param int year: Optional, needs to be in range of data available. If nothing is specified, a typical year will be loaded
    :return: None
    """
    # Read NodeLocations.csv with node column as index
    node_locations_path = os.path.join(folder_path, "NodeLocations.csv")
    node_locations_df = pd.read_csv(node_locations_path, sep=';', names=["node", "lon", "lat", "alt"])

    # Read nodes and investment_periods from the JSON file
    json_file_path = os.path.join(folder_path, "topology.json")
    with open(json_file_path, 'r') as json_file:
        topology = json.load(json_file)

    year = int(topology["start_date"].split("-")[0]) if topology["start_date"] else "typical_year"

    for key in topology["investment_periods"]:
        for node_name in topology["nodes"]:
            # Read lon, lat, and alt for this node name from node_locations_df
            node_data = node_locations_df[node_locations_df["node"] == node_name]
            lon = node_data["lon"].values[0] if not pd.isnull(node_data["lon"].values[0]) else 5.5
            lat = node_data["lat"].values[0] if not pd.isnull(node_data["lat"].values[0]) else 52.5
            alt = node_data["alt"].values[0] if not pd.isnull(node_data["alt"].values[0]) else 10

            if dataset == "JRC":
                # Fetch climate data for the node
                data = import_jrc_climate_data(lon, lat, year, alt)
            else:
                raise Exception("Other APIs are not available")

            # Write data to CSV file
            output_folder = os.path.join(folder_path, key, "node_data", node_name)
            output_file = os.path.join(output_folder, "ClimateData.csv")
            existing_data = pd.read_csv(output_file, sep=';')

            # Fill in existing data with data from the fetched DataFrame based on column names
            for column, value in data['dataframe'].items():
                existing_data[column] = value.values[:len(existing_data)]

            # Save the updated data back to ClimateData.csv
            existing_data.to_csv(output_file, index=False, sep=';')

        # End of node loop, proceed to next investment period
        print(f"Finished processing investment period: {key}")

    print("All investment periods processed.")