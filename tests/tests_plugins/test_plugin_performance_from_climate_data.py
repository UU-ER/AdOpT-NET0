import shutil

from adopt_net0.plugins.plugin_manager import PluginManager
from adopt_net0.plugins.hooks import Hook
from tests.utilities import create_plugin_testing_mock_data


def _perform_test(request, technology:str):
    """
    Tests test_performance_from_climate_data for a technology

    :param request: request containing testing info
    :param technology: technology name
    """
    period_name = "period1"
    node_name = "node1"
    data_path = request.config.data_folder_path

    create_plugin_testing_mock_data(data_path, node_name, period_name, technology)
    shutil.copy2(request.config.mock_data_path / "ClimateData.csv",
                 data_path / period_name / "node_data" / node_name / "ClimateData.csv")

    pm = PluginManager()
    plugin_list = {"preprocessing_plugins.performance_from_climate_data":
        {
            "config":
                {
                    "technologies": [technology]
                }
        }}

    pm.register(plugin_list)
    pm.emit(Hook.DATA_READ_START, data_path=data_path)

def test_performance_from_climate_data_pv(request):
    """
    Tests pv performance from climate data
    """
    technology = "Photovoltaic"
    _perform_test(request, technology)


def test_performance_from_climate_data_wt(request):
    """
    Tests wind turbine performance from climate data
    """
    technology = "WindTurbine_Onshore_2500"
    _perform_test(request, technology)

def test_performance_from_climate_data_stor(request):
    """
    Tests storage performance from climate data
    """
    technology = "Storage_Battery"
    _perform_test(request, technology)



