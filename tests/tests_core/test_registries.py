from adopt_net0.core.registries import TechnologyRegistry, NetworkRegistry

def test_technology_registry():
    """
    Tests standard behavior of TechnologyRegistry Class
    """
    registry = TechnologyRegistry()
    registry.register_builtin_technologies()

def test_network_registry():
    """
    Tests standard behavior of NetworkRegistry Class
    """
    registry = NetworkRegistry()
    registry.register_builtin_networks()