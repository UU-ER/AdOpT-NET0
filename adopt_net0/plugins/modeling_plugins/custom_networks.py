"""
Plugin enabling custom networks.

Each custom network is implemented as a separate class in ``custom_networks_helpers``. To use a custom network,
add its name to the plugin configuration under the "networks" key:

.. code-block:: json

   "modeling_plugins.custom_networks":
    {
        "config":
            {
                "networks": ["Network_A", "Network_B"]
            }
    }

Note that the networks are rely on climate data, and thus they require a csv file with climate data in the
respective nodal directory.
"""

from adopt_net0.plugins.base import Plugin as PluginBase
from adopt_net0.plugins.hooks import Hook
from adopt_net0.plugins.modeling_plugins.custom_networks_helpers import *

FACTORY = {
    "FLUID": Fluid,
    "ELECTRICITY": Electricity,
}


class Plugin(PluginBase):
    """
    Plugin enabling custom networks.

    """
    name = "Custom network models"
    hooks = {
        Hook.NETWORK_REGISTRATION,
    }
    config_template = {
      "networks": []
    }

    def network_registration(self, network_registry=None) -> None:
        """
        Registers custom network models for all networks specified in plugin config.

        :param registry: NetworkRegistry
        """
        for network in self.config["networks"]:
            if network in FACTORY:
                network_registry.register(FACTORY[network], network)
            else:
                raise RuntimeError(f"Unknown custom network model for {network}")
