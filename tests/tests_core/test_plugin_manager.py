from adopt_net0.plugins.plugin_manager import PluginManager


def test_plugin_manager(request):
    """
    Tests if plugin manager can discover known example plugins.
    """
    pm = PluginManager()
    discovered = set(pm.discover_plugins())
    pm.print_plugins()

    # All known plugins should be discovered
    expected = {
        "adopt_net0.plugins.preprocessing_plugins.performance_from_climate_data",
        "adopt_net0.plugins.modeling_plugins.custom_networks",
        "adopt_net0.plugins.modeling_plugins.custom_technologies",
        "adopt_net0.plugins.modeling_plugins.operational_constraints_technologies",
        "adopt_net0.plugins.modeling_plugins.technology_retrofits",
    }
    assert discovered == expected
