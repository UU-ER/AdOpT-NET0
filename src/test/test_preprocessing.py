import pytest

"""
- Test data_loading
- fill carrier_data
- copy technology data
- copy network data
"""


@pytest.mark.data_preprocessing
def test_create_optimization_templates(request):
    """Tests the function create_optimization_templates"""
    data_path = request.config.data_folder_path
