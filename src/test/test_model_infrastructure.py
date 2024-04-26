import pytest
import pandas as pd

from src.energyhub import EnergyHub
from src.test.utilities import create_basic_case_study


# @pytest.mark.model_infrastructure
# def test_data_handle_reading(request):
#     """
#     Tests standard behavior of DataHandle Class
#     - reads in data
#     """
#     case_study_folder_path = request.config.case_study_folder_path
#
#     create_basic_case_study(case_study_folder_path)
#
#     dh = DataHandle(case_study_folder_path)
