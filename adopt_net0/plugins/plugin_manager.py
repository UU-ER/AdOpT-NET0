from importlib import import_module
from typing import List, Dict
import logging
import pkgutil

from adopt_net0.plugins.hooks import Hook
from adopt_net0.plugins.base import Plugin as PluginBase

log = logging.getLogger(__name__)

class PluginManager:
    """
    Plugin manager

    Responsibilities (minimal):
    - Load plugins listed in a config
    - Auto-discover plugins under `modeling_plugins` and `preprocessing_plugins`
    - Emit hooks to loaded plugins
    """

    def __init__(self, base_package: str = "adopt_net0.plugins"):
        self.base_package = base_package
        self._plugins: List[PluginBase] = []

    def discover_plugins(self) -> List[str]:
        """
        Auto-discover plugin modules under modeling_plugins and preprocessing_plugins.

        Returns a list of import paths that look like modules exposing a `Plugin` class.
        """
        discovered: List[str] = []

        for sub in ("modeling_plugins", "preprocessing_plugins"):
            sub_pkg_name = f"{self.base_package}.{sub}"

            try:
                sub_pkg = import_module(sub_pkg_name)
            except ImportError:
                log.debug("Plugin subpackage %s not found", sub_pkg_name)
                continue

            for finder, name, ispkg in pkgutil.iter_modules(sub_pkg.__path__):
                if name.startswith("_"):
                    continue

                module_path = f"{sub_pkg_name}.{name}"

                try:
                    mod = import_module(module_path)
                    if hasattr(mod, "Plugin"):
                        discovered.append(module_path)
                        log.info("Discovered plugin %s", module_path)
                except Exception:
                    log.exception("Failed importing discovered plugin %s", module_path)

        return discovered

    def register(self, plugin_ids: dict):
        """
        Register plugins based on the provided list.

        The config may be a list of identifiers
        """
        if plugin_ids is None:
            log.debug("No plugin config provided; skipping")
            return

        for plugin_id, plugin_config in plugin_ids.items():
            plugin_config = plugin_config["config"]
            mod_path = f"{self.base_package}.{plugin_id}"
            if not mod_path:
                log.warning("Could not resolve plugin identifier: %s", plugin_id)
                continue
            try:
                mod = import_module(mod_path)
                cls = getattr(mod, "Plugin", None)
                inst = cls(plugin_config)
                try:
                    inst.activate()
                except Exception:
                    log.exception("Plugin.activate() failed for %s", mod_path)
                self._plugins.append(inst)
                log.info("Loaded plugin %s", getattr(inst, "name", mod_path))
            except Exception:
                log.exception("Failed loading plugin %s", mod_path)

    def emit(self, hook: Hook, **kwargs):
        """
        Call hook on all plugins. One can pass additional arguments via kwargs.
        """
        hook_name = hook.value

        for p in list(self._plugins):
            fn = getattr(p, hook_name, None)
            if callable(fn):
                return fn(**kwargs)

    def get_plugins(self):
        return list(self._plugins)

    def print_plugins(self):
        available_plugins = self.discover_plugins()

        for p in available_plugins:
            print(p)

    def deregister_all(self):
        """
        Deregister all plugins.
        """
        self._plugins.clear()
