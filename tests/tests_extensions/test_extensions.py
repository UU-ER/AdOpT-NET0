import pandas as pd
import os
import json

from adopt_net0.extensions import load_climate_data_from_api
from adopt_net0.core.data_management.utilities import calculate_dni
from tests.utilities import (
    get_topology_data,
)

def test_climate_data_loading(request):
    """
    Tests standard behavior of load_climate_data_from_api
    - Tests if df is not empty
    - Tests if climate data is the same in investment periods
    """
    case_study_folder_path = request.config.case_study_folder_path

    # Write it to file
    load_climate_data_from_api(case_study_folder_path)

    # Get periods and nodes:
    investment_periods, nodes, carriers = get_topology_data(case_study_folder_path)

    # Verify that climate data is not empty
    climate_data = {}
    for period in investment_periods:
        climate_data[period] = {}
        for node in nodes:
            climate_data[period][node] = pd.read_csv(
                case_study_folder_path
                / period
                / "node_data"
                / node
                / "ClimateData.csv",
                sep=";",
                index_col=0,
            )
            assert not climate_data[period][node].empty

            # calculate dni and check if its ok
            node_locations = pd.read_csv(
                case_study_folder_path / "NodeLocations.csv", sep=";", index_col=0
            )
            lon = node_locations.loc[node, "lon"]
            lat = node_locations.loc[node, "lat"]

            climate_data_check = climate_data[period][node]
            climate_data_check["dni_correct"] = climate_data_check["dni"]
            climate_data_check = climate_data_check.drop(columns=["dni"])
            climate_data_check["dni_check"] = calculate_dni(
                climate_data_check, lon, lat
            )

