import pandas as pd

from ..components.utilities import Economics


class ModelComponent:
    """
    Class to read and manage data for technologies and networks. This class inherits its attributes to the technology
    and network classes.
    """

    def __init__(self, data):
        """
        Initializes component class
        """
        self.name = data["name"]
        self.existing = 0
        self.size_initial = []
        self.size_is_int = data["size_is_int"]
        self.size_min = data["size_min"]
        self.size_max = data["size_max"]
        self.decommission = data["decommission"]
        self.economics = Economics(data["Economics"])
        self.big_m_transformation_required = 0

        self.results = {}
        self.results["time_dependent"] = pd.DataFrame()
        self.results["time_independent"] = pd.DataFrame()
