from adopt_net0.core.components.technologies import *
from adopt_net0.core.components.networks import *

class ComponentRegistry:
    """
    Registry for component constructors.

    This class maintains a mapping of component type identifiers to their
    corresponding constructor functions.
    """
    def __init__(self):
        """Initialize an empty technology registry."""
        self._constructors = {}

    def register(self, component_type: str, ctor):
        """
        Register a technology constructor under a given type identifier.

        :param str component_type: Unique identifier for the component model
        :param callable ctor: A constructor or factory function that accepts data and returns the component instance.
        """
        if component_type in self._constructors:
            raise ValueError(f"Component type '{component_type}' already registered")
        self._constructors[component_type] = ctor

    def create(self, component_type: str, component_data: dict):
        """
        Create a component instance using the registered constructor.

        :param str component_type: string with component model
        :param dict component_data: component data
        :return:

        """
        if component_type in self._constructors:
            return self._constructors[component_type](component_data)
        else:
            raise KeyError(f"Unknown component type '{component_type}'. If type is plugin-provided, ensure the plugin is loaded.")

class TechnologyRegistry(ComponentRegistry):
    """
    Registry for technology constructors.

    This class maintains a mapping of technology type identifiers to their
    corresponding constructor functions.

    """
    def register_builtin_technologies(self):
        """
        Register built-in technology constructors in the provided registry.
        """
        self.register("RES", Res)
        self.register("CONV1", Conv1)
        self.register("CONV2", Conv2)
        self.register("CONV3", Conv3)
        self.register("CONV4", Conv4)
        self.register("STOR", Stor)
        self.register("SINK", Sink)



class NetworkRegistry(ComponentRegistry):
    """
    Registry for network constructors.

    This class maintains a mapping of network type identifiers to their
    corresponding constructor functions.

    """
    def register_builtin_networks(self):
        """
        Register built-in network constructors in the provided registry.
        """
        self.register("SIMPLE", Simple)
