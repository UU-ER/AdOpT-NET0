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
        self.data = data
        self.bounds = {}
        self.processed_coeff = ProcessedCoefficients()
        self.big_m_transformation_required = 0




        # Todo: Remove later
        if "existing" in data:
            self.existing = data["existing"]
            self.size_initial = data["size_initial"]
        else:
            self.existing = 0
            self.size_initial = None

        self.size_min = data["size_min"]
        self.size_max = data["size_max"]
        self.size_is_int = data["size_is_int"]
        self.decommission = data["decommission"]
        self.economics = data["Economics"]
        self.performance_data = data["Performance"]
        self.bounds = {"input": {}, "output": {}}


    def fit_performance(self, modelhub, component_id: tuple, **kwargs):
        """
        Fits technology performance and writes it to self.

        Implementation in subclasses.

        :param modelhub: model hub
        :param tuple component_id: component id containing (period, node, component name)
        """
        pass

    def construct_model(self, model_block, modelhub, set_t_full, set_t_clustered, **kwargs):
        pass



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
