from .utilities import get_attribute_from_dict


class ModelComponent:
    """
    Class to read and manage data for technologies and networks. This class inherits
    its attributes to the technology and network classes.
    """

    def __init__(self, data: dict):
        """
        Initializes component class

        Attributes include:

        - name: technology name
        - existing: if component is existing or not
        - size_initial: if existing, initial size
        - economics: contains economic data
        - bounds: (for technologies only) containing bounds on input and output
           variables that are calculated in technology subclasses
        - processed_coeff: fitted/processed coefficients
        - big_m_transformation_required: flag to use for disjunctive programming

        :param dict data: technology/network data
        """
        self.name = data["name"]
        self.existing = 0
        self.size_min = data["size_min"]
        self.size_max = data["size_max"]
        self.size_is_int = data["size_is_int"]
        self.size_initial = []
        self.decommission = data["decommission"]

        self.economics = data["Economics"]
        self.performance_data = data["Performance"]

        self.bounds = {"input": {}, "output": {}}
        self.processed_coeff = ProcessedCoefficients()

        self.big_m_transformation_required = 0


class ProcessedCoefficients:
    """
    Defines a simple class for fitted/processed coefficients
    """

    def __init__(self):
        self.time_dependent_full = {}
        self.time_dependent_clustered = {}
        self.time_dependent_averaged = {}
        self.time_dependent_used = {}
        self.time_independent = {}
        self.dynamics = {}
