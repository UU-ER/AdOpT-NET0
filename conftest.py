import pytest
from create_test_data import *
import os
from pathlib import Path

@pytest.fixture(autouse=True)
def setup_before_tests():
    """
    Fixture to create the test data before running all tests
    """

    # Create Folder
    folder_path = Path("test/test_data")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create Test Data
    create_data_test_data_handle()
    create_data_model1()
    create_data_model2()
    create_data_emissionbalance1()
    create_data_emissionbalance2()
    create_data_technology_type1_PV()
    create_data_technology_type1_WT()
    create_data_technology_CONV()
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

    # Place your teardown code here
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_name}: {e}")
    else:
        print(f"The folder '{folder_path}' does not exist or is not a directory.")
