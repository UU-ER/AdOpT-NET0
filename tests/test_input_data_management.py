from adopt_net0.core.data_management.handle_input_data import DataHandle


def test_data_handle_reading(request):
    """
    Tests standard behavior of DataHandle Class
    - reads in data
    """
    case_study_folder_path = request.config.case_study_folder_path

    dh = DataHandle()
    dh.set_settings(case_study_folder_path)
