import pytest
import os
from create_test_data import *
from pathlib import Path
import shutil


@pytest.fixture(autouse=True)
def setup_before_tests():
    """
    Fixture to create the test data before running all tests
    """
    # Create Folder
    data_folder_path = Path("./src/test/test_data")
    result_folder_path = Path("./src/test/results")

    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    # Create Test Data
    create_data_test_data_handle()
    create_data_model1()
    create_data_model2()
    create_data_emissionbalance1()
    create_data_emissionbalance2()
    create_data_technology_type1_PV()
    create_data_technology_type1_WT()
    create_data_technology_CONV()
    create_data_technology_dynamics()
    create_data_network()
    create_data_addtechnology()
    create_data_technologySTOR()
    create_data_time_algorithms()
    create_data_optimization_types()
    create_data_existing_technologies()
    create_data_existing_networks()
    create_test_data_dac()
    create_data_technologyOpen_Hydro()
    create_data_carbon_tax()
    create_data_carbon_subsidy()

    # Yield control back to the test functions
    yield

    # Clean up after testing
    if os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
        shutil.rmtree(data_folder_path)
    if os.path.exists(result_folder_path) and os.path.isdir(result_folder_path):
        shutil.rmtree(result_folder_path)
