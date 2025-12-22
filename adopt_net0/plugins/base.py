from typing import Any


class Plugin:
    """
    Minimal plugin interface.
    """

    name = "base"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def activate(self) -> None:
        """Called once when plugin is loaded."""
        return None

