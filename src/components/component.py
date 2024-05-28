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

        Attributes include:

        - parameters: unfitted parameters from json files
        - options: component options that are unrelated to the performance of the
          component
        - info: component infos, such as carriers, model to use etc.
        - bounds: (for technologies only) containing bounds on input and output
         variables that are calculated in technology subclasses
         - coeff: fitted coefficients

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
