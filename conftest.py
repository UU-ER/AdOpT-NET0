import pytest
import os
from pathlib import Path
import shutil

from src.data_preprocessing import *


def pytest_configure(config):
    config.data_folder_path = Path("./src/test/test_data")
    config.result_folder_path = Path("./src/test/test_results")
    config.case_study_folder_path = Path("./src/test/test_case")
    config.technology_data_folder_path = Path("./src/test/technology_data")
    config.network_data_folder_path = Path("./src/test/network_data")
    config.root_folder_path = Path(".")
    config.solver = "glpk"


@pytest.fixture(autouse=True)
def setup_before_tests(request):
    """
    Fixture to create the test data before running all tests
    """
    # Create Folder
    data_folder_path = request.config.data_folder_path
    result_folder_path = request.config.data_folder_path
    case_study_folder_path = request.config.case_study_folder_path

    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    if not os.path.exists(case_study_folder_path):
        os.makedirs(case_study_folder_path)

    # Create case study folders for preprocessing
    create_optimization_templates(case_study_folder_path)
    create_input_data_folder_template(case_study_folder_path)

    #
    # # Create Test Data
    # create_data_test_data_handle()
    # create_data_model1()
    # create_data_model2()
    # create_data_emissionbalance1()
    # create_data_emissionbalance2()
    # create_data_technology_type1_PV()
    # create_data_technology_type1_WT()
    # create_data_technology_CONV()
    # create_data_technology_dynamics()
    # create_data_network()
    # create_data_addtechnology()
    # create_data_technologySTOR()
    # create_data_time_algorithms()
    # create_data_optimization_types()
    # create_data_existing_technologies()
    # create_data_existing_networks()
    # create_test_data_dac()
    # create_data_technologyOpen_Hydro()
    # create_data_carbon_tax()
    # create_data_carbon_subsidy()

    # Yield control back to the test functions
    yield

    # Clean up after testing
    if os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
        shutil.rmtree(data_folder_path)
    if os.path.exists(result_folder_path) and os.path.isdir(result_folder_path):
        shutil.rmtree(result_folder_path)
    if os.path.exists(case_study_folder_path) and os.path.isdir(case_study_folder_path):
        shutil.rmtree(case_study_folder_path)
