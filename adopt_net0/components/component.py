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
        - input_parameters: unfitted parameters from json files
        - component_options: component options that are unrelated to the performance
          of the component
        - bounds: (for technologies only) containing bounds on input and output
           variables that are calculated in technology subclasses
        - processed_coeff: fitted/processed coefficients
        - big_m_transformation_required: flag to use for disjunctive programming

        :param dict data: technology/network data
        """
        self.name = data["name"]
        self.existing = 0
        self.size_initial = []
        self.economics = Economics(data["Economics"])

        self.input_parameters = InputParameters(data)
        self.component_options = ComponentOptions(data)
        self.bounds = {"input": {}, "output": {}}
        self.processed_coeff = ProcessedCoefficients()

        self.big_m_transformation_required = 0


class Economics:
    """
    Class to manage economic data of technologies and networks.

    Contains capex and opex data
    """

    def __init__(self, economics: dict):
        """
        Constructor

        :param dict economics: Dict containing economic data of component
        """
        if "CAPEX_model" in economics:
            self.capex_model = economics["CAPEX_model"]
        self.capex_data = {}
        if "unit_CAPEX" in economics:
            self.capex_data["unit_capex"] = economics["unit_CAPEX"]
        if "fix_CAPEX" in economics:
            self.capex_data["fix_capex"] = economics["fix_CAPEX"]
        if "piecewise_CAPEX" in economics:
            self.capex_data["piecewise_capex"] = economics["piecewise_CAPEX"]
        if "gamma1" in economics:
            self.capex_data["gamma1"] = economics["gamma1"]
            self.capex_data["gamma2"] = economics["gamma2"]
            self.capex_data["gamma3"] = economics["gamma3"]
            self.capex_data["gamma4"] = economics["gamma4"]
        self.opex_variable = economics["OPEX_variable"]
        self.opex_fixed = economics["OPEX_fixed"]
        self.discount_rate = economics["discount_rate"]
        self.lifetime = economics["lifetime"]
        self.decommission_cost = economics["decommission_cost"]


class InputParameters:
    """
    Class to hold unfitted/unprocessed performance of technologies and networks
    """

    def __init__(self, component_data: dict):
        self.performance_data = component_data["Performance"]
        self.size_min = component_data["size_min"]
        self.size_max = component_data["size_max"]
        self.rated_power = 1

        self.rated_power = get_attribute_from_dict(
            component_data["Performance"], "rated_power", 1
        )
        self.min_part_load = get_attribute_from_dict(
            component_data["Performance"], "min_part_load", 0
        )
        self.standby_power = get_attribute_from_dict(
            component_data["Performance"], "standby_power", -1
        )
        self.pressure = get_attribute_from_dict(
            component_data["Performance"], "pressure", {}
        )


class ComponentOptions:
    """
    Class to hold options for technologies and networks
    """

    def __init__(self, component_data: dict):
        self.modelled_with_full_res = False
        self.lower_res_than_full = False
        self.size_is_int = component_data["size_is_int"]
        self.decommission = component_data["decommission"]
        self.size_based_on = None
        self.emissions_based_on = None

        # TECHNOLOGY
        # Technology Model
        if "tec_type" in component_data:
            self.technology_model = component_data["tec_type"]

        # Input carrier
        self.input_carrier = get_attribute_from_dict(
            component_data["Performance"], "input_carrier", []
        )

        # Output Carriers
        self.output_carrier = get_attribute_from_dict(
            component_data["Performance"], "output_carrier", []
        )

        # Performance Function Type
        self.performance_function_type = get_attribute_from_dict(
            component_data["Performance"], "performance_function_type", None
        )

        # CCS
        if (
            "ccs" in component_data["Performance"]
            and component_data["Performance"]["ccs"]["possible"]
        ):
            self.ccs_possible = True
            self.ccs_type = component_data["Performance"]["ccs"]["ccs_type"]
        else:
            self.ccs_possible = False
            self.ccs_type = None

        # Standby power
        self.standby_power_carrier = get_attribute_from_dict(
            component_data["Performance"], "standby_power_carrier", -1
        )

        # Determined in child classes
        self.main_input_carrier = None
        self.main_output_carrier = None

        # NETWORKS
        # Transported carrier
        if "carrier" in component_data["Performance"]:
            self.transported_carrier = component_data["Performance"]["carrier"]

        if "energyconsumption" in component_data["Performance"]:
            if component_data["Performance"]["energyconsumption"]:
                self.energyconsumption = 1
            else:
                self.energyconsumption = 0

        # disable bidirectional for networks and storage
        if "network_type" in component_data:
            if "bidirectional_network" in component_data["Performance"]:
                self.bidirectional_network = component_data["Performance"][
                    "bidirectional_network"
                ]
            self.bidirectional_network_precise = get_attribute_from_dict(
                component_data["Performance"], "bidirectional_network_precise", 1
            )

        if "allow_only_one_direction" in component_data["Performance"]:
            self.allow_only_one_direction = component_data["Performance"][
                "allow_only_one_direction"
            ]
            if self.allow_only_one_direction:
                self.allow_only_one_direction_precise = get_attribute_from_dict(
                    component_data["Performance"], "allow_only_one_direction_precise", 1
                )
        # other technology specific options
        self.other = {}


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
