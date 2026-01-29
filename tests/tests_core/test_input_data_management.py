from adopt_net0.core.data_management.read_input_data import *


def test_data_handle_reading(request):
    """
    Tests standard behavior of DataHandle Class
    - reads in data
    """
    case_study_folder_path = request.config.case_study_folder_path

    topology = read_topology(case_study_folder_path)
    assert topology is not None

    config = read_config(case_study_folder_path, topology)
    assert config is not None

    systems_data = read_system_data(case_study_folder_path, topology)
    assert systems_data is not None

