"""
Plugin enabling custom technologies.

Each custom technology is implemented as a separate class in ``custom_technologies_helpers``. To use a custom technology,
add its name to the plugin configuration under the "technologies" key:

.. code-block:: json

   "modeling_plugins.custom_technologies":
    {
        "config":
            {
                "technologies": ["Technology_A", "Technology_B"]
            }
    }

Note that some technologies may require additional climate data. Refer to the documentation of each technology class for details.
"""

from adopt_net0.plugins.base import Plugin as PluginBase
from adopt_net0.plugins.hooks import Hook
from adopt_net0.plugins.modeling_plugins.custom_technologies_helpers import *

FACTORY = {
    "DAC_Adsorption": DacAdsorption,
    "GasTurbine": GasTurbine,
    "HeatPump": HeatPump,
    "HydroOpen": HydroOpen,
    "CCPP": CCPP,
    "Conversion_operational_constrained": ConvOC,
}


class Plugin(PluginBase):
    """
    Plugin enabling custom technologies.
    """
    name = "Custom technology models"
    hooks = {
        Hook.TECHNOLOGY_REGISTRATION,
    }
    config_template = {
      "technologies": []
    }

    def technology_registration(self, technology_registry=None) -> None:
        """
        Registers custom technology models for all networks specified in plugin config.

        :param registry: TechnologyRegistry
        """
        for technology in self.config["technologies"]:
            if technology in FACTORY:
                technology_registry.register(FACTORY[technology], technology)
            else:
                raise RuntimeError(f"Unknown custom technology model for {technology}")
