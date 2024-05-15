from ..components.utilities import Economics
from .utilities import Parameters, ComponentInfo, ComponentOptions, Coefficients


class ModelComponent:
    """
    Class to read and manage data for technologies and networks. This class inherits
    its attributes to the technology and network classes.
    """

    def __init__(self, data: dict):
        """
        Initializes component class

        :param dict data: technology/network data
        """
        self.name = data["name"]
        self.existing = 0
        self.size_initial = []
        self.economics = Economics(data["Economics"])

        self.parameters = Parameters(data)
        self.options = ComponentOptions(data)
        self.info = ComponentInfo(data)
        self.bounds = {"input": {}, "output": {}}
        self.coeff = Coefficients()

        self.big_m_transformation_required = 0
