from typing import Any


class Plugin:
    """Minimal plugin interface.

    Implement any of the hook methods you need. Hooks receive the ModelHub
    instance (or other relevant objects) as argument(s).
    """

    name = "base"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def activate(self) -> None:
        """Called once when plugin is loaded."""
        return None
    #
    # def on_solve_start(self, modelhub: Any) -> None:
    #     """Called at very beginning of ModelHub.solve()."""
    #     return None
    #
    # def on_solve_end(self, modelhub: Any) -> None:
    #     """Called at the very end of ModelHub.solve()."""
    #     return None

