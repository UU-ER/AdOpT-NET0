import pytest
from src.test.utilities import (
    select_random_list_from_list,
    load_json,
    save_json,
    get_topology_data,
)

from src.data_management import DataHandle


@pytest.mark.data_preprocessing
def test_data_handle(request):
    """
    Tests standard behavior of DataHandle Class
    - reads in data
    - clusters data
    - averages data
    """
    case_study_folder_path = request.config.case_study_folder_path

    dh = DataHandle(case_study_folder_path)
    dh._cluster_data()
    dh._average_data()
