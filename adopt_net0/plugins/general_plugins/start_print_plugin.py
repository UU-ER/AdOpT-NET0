from adopt_net0.plugins.base import Plugin as PluginBase


class Plugin(PluginBase):
    name = "print_start"

    def on_solve_start(self, modelhub):
        print("[plugin] Starting to solve")
