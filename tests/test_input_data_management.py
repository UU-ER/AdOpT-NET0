import pytest

from adopt_net0.data_management import DataHandle


def test_data_handle_reading(request):
    """
    Tests standard behavior of DataHandle Class
    - reads in data
    """
    case_study_folder_path = request.config.case_study_folder_path

    dh = DataHandle()
    dh.set_settings(case_study_folder_path)


#
# @pytest.mark.data_management
# def test_data_handle_clustering(request):
#     """
#     Tests clustering algorithm:
#     - reads in data
#     - clusters data
#     """
#     case_study_folder_path = request.config.case_study_folder_path
#
#     dh = DataHandle()
#     dh.read_input_data(case_study_folder_path)
#     dh.model_config["optimization"]["typicaldays"]["N"]["value"] = 2
#     dh._cluster_data()
#
#
# @pytest.mark.data_management
# def test_data_handle_averaging(request):
#     """
#     Tests averaging algorithm:
#     - reads in data
#     - averages data
#     """
#     case_study_folder_path = request.config.case_study_folder_path
#
#     dh = DataHandle()
#     dh.read_input_data(case_study_folder_path)
#     dh.model_config["optimization"]["timestaging"]["value"] = 2
#     dh._average_data()
