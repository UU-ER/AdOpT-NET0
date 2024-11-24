import pytest
import os
from pathlib import Path
import shutil

import adopt_net0.data_preprocessing as dp


def pytest_configure(config):
    """
    Creates global data used in tests

    Creates global data used in tests. Contains:

    - data_folder_path: Directory storing temporary test data (deleted after testing)
    - result_folder_path: Directory storing temporary result data (deleted after
      testing)
    - case_study_folder_path: Directory storing temporary case study data (deleted
      after testing)
    - technology_data_folder_path: Directory containing technology data
    - network_data_folder_path: Directory containing network data
    - root_folder_path: root directory
    - solver: Solver to use during testing

    :param config: Configuration for pytest
    """
    config.data_folder_path = Path("adopt_net0/test/test_data")
    config.result_folder_path = Path("adopt_net0/test/test_results")
    config.case_study_folder_path = Path("adopt_net0/test/test_case")
    config.technology_data_folder_path = Path("tests/technology_data")
    config.network_data_folder_path = Path("tests/network_data")
    config.root_folder_path = Path(".")
    config.solver = "glpk"


@pytest.fixture(autouse=True)
def setup_before_tests(request):
    """
    Fixture to make test directories and clean-up after test

    Creates directories required for testing and deletes these directories after
    testing.

    :param request: request containing settings for testing
    """
    # Create Folders
    data_folder_path = request.config.data_folder_path
    result_folder_path = request.config.result_folder_path
    case_study_folder_path = request.config.case_study_folder_path

    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    if not os.path.exists(case_study_folder_path):
        os.makedirs(case_study_folder_path)

    # Create case study folders for preprocessing
    dp.create_optimization_templates(case_study_folder_path)
    dp.create_input_data_folder_template(case_study_folder_path)

    # Yield control back to the test functions
    yield

    # Clean up after testing (deletes folders again)
    if os.path.exists(data_folder_path) and os.path.isdir(data_folder_path):
        shutil.rmtree(data_folder_path)
    if os.path.exists(result_folder_path) and os.path.isdir(result_folder_path):
        shutil.rmtree(result_folder_path)
    if os.path.exists(case_study_folder_path) and os.path.isdir(case_study_folder_path):
        shutil.rmtree(case_study_folder_path)
