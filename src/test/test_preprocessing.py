import pytest
import json
import pandas as pd
from pathlib import Path
import random

from src.data_preprocessing import *

"""
- Test data_loading
- fill carrier_data
- copy technology data
- copy network data
"""


def get_topology_data(folder_path: Path) -> (list, list, list):
    """
    Gets investment periods, nodes and carriers from path
    :param Path folder_path: folder path containing topology
    :return: tuple of lists with investment_period, nodes and carriers
    """
    json_file_path = folder_path / "Topology.json"
    with open(json_file_path, "r") as json_file:
        topology = json.load(json_file)

    investment_periods = topology["investment_periods"]
    nodes = topology["nodes"]
    carriers = topology["carriers"]
    return investment_periods, nodes, carriers


@pytest.mark.data_preprocessing
def test_create_input_data_folder(request):
    """
    Tests standard behavior of
    - create_optimization_templates
    - initialize_configuration_templates
    - initialize_topology_templates
    - create_input_data_folder_template
    - create_empty_network_matrix
    """
    data_folder_path = request.config.data_folder_path
    create_optimization_templates(data_folder_path)
    create_input_data_folder_template(data_folder_path)


@pytest.mark.data_preprocessing
def test_data_climate_data_loading(request):
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
            )
            assert not climate_data[period][node].empty

    # Verify that climate data is the same for both periods
    for node in nodes:
        assert climate_data[investment_periods[0]][node].equals(
            climate_data[investment_periods[1]][node]
        )


@pytest.mark.data_preprocessing
def test_data_fill_carrier_data(request):
    """
    Tests standard behavior of fill_carrier_data
    - Tests if df is indeed filled
    """
    case_study_folder_path = request.config.case_study_folder_path

    # Write to files (should fill some random columns with 1 for all nodes and investment periods
    carriers_to_fill = ["electricity"]
    fill_options = [
        "Demand",
        "Import limit",
        "Export limit",
        "Import price",
        "Export price",
        "Import emission factor",
        "Export emission factor",
        "Generic production",
    ]
    num_items = random.randint(1, len(fill_options))
    series_to_fill = random.sample(fill_options, num_items)

    fill_carrier_data(
        case_study_folder_path, 1, columns=series_to_fill, carriers=carriers_to_fill
    )

    # Check if it is filled indeed
    # Get periods and nodes:
    investment_periods, nodes, carriers = get_topology_data(case_study_folder_path)

    # Verify that climate data is not empty
    for period in investment_periods:
        for node in nodes:
            for carrier in carriers:
                carrier_data = pd.read_csv(
                    case_study_folder_path
                    / period
                    / "node_data"
                    / node
                    / "carrier_data"
                    / (carrier + ".csv"),
                    sep=";",
                )
                for col in carrier_data.columns:
                    if (carrier in carriers_to_fill) and (col in series_to_fill):
                        assert (carrier_data[col] == 1).all()


@pytest.mark.data_preprocessing
def test_data_fill_carrier_data(request):
    """
    Tests standard behavior of fill_carrier_data
    - Tests if df is indeed filled
    """
    pass
