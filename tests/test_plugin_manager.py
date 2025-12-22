from adopt_net0.plugins.plugin_manager import PluginManager


def test_discover_plugins_contains_known_plugins():
    """
    Tests if plugin manager can discover known example plugins.
    """
    pm = PluginManager()
    discovered = pm.discover_plugins()
    # Expect at least the two example plugins under general_plugins
    assert any("general_plugins.end_print_plugin" in p for p in discovered)
    assert any("general_plugins.start_print_plugin" in p for p in discovered)


def test_load_from_config():
    """
    Tests if plugin manager can load plugins from config.
    """
    pm = PluginManager()
    pm.load_from_config(["general_plugins.start_print_plugin"])
    loaded = [getattr(p, "name", None) for p in pm.get_plugins()]




