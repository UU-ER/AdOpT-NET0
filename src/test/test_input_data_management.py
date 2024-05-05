import pytest
from src.test.utilities import (
    select_random_list_from_list,
    load_json,
    save_json,
    get_topology_data,
)

from src.data_management import DataHandle


@pytest.mark.data_management
def test_data_handle_reading(request):
    """
    Tests standard behavior of DataHandle Class
    - reads in data
    """
    case_study_folder_path = request.config.case_study_folder_path

    dh = DataHandle()
    dh.read_input_data(case_study_folder_path)


@pytest.mark.data_management
def test_data_handle_clustering(request):
    """
    Tests clutering algorithm of
    - reads in data
    - clusters data
    """
    case_study_folder_path = request.config.case_study_folder_path

    dh = DataHandle()
    dh.read_input_data(case_study_folder_path)
    dh.model_config["optimization"]["typicaldays"]["N"]["value"] = 2
    dh._cluster_data()


@pytest.mark.data_management
def test_data_handle_averaging(request):
    """
    Tests standard behavior of DataHandle Class
    - reads in data
    """
    case_study_folder_path = request.config.case_study_folder_path

    dh = DataHandle()
    dh.read_input_data(case_study_folder_path)
    dh.model_config["optimization"]["timestaging"]["value"] = 2
    dh._average_data()
