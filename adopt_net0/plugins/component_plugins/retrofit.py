from adopt_net0.plugins.base import Plugin as PluginBase


class Plugin(PluginBase):
    """
    Base class implementing a retrofit to a component
    """
    name = "retrofit_plugin"

    def on_fit_performance(self):
        pass

    def on_model_construction(self):
        pass