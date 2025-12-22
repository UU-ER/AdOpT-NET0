from adopt_net0.plugins.base import Plugin as PluginBase


class Plugin(PluginBase):
    name = "print_end"

    def on_solve_end(self, modelhub):
        print("[plugin] Solving finished")
