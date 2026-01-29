from enum import Enum


class Hook(Enum):
    # Data reading hooks
    DATA_READ_START = "on_data_read_start"
    DATA_READ_END = "on_data_read_end"

    # Custom model component registration
    TECHNOLOGY_REGISTRATION = "technology_registration"
    NETWORK_REGISTRATION = "network_registration"

    # Construction hooks
    TECHNOLOGY_CONSTRUCTION_END = "on_technology_construction_end"
    NODE_CONSTRUCTION_END = "on_node_construction_end"

    # Balance construction hooks
    ADD_TO_ENERGYBALANCE = "on_energybalance_construction"
    ADD_TO_COSTBALANCE_CAPEX = "on_costbalance_capex_construction"
    ADD_TO_COSTBALANCE_OPEX = "on_costbalance_opex_construction"
    ADD_TO_EMISSIONBALANCE = "on_emissionbalance_construction"

    # Result writing hooks
    RESULTS_WRITING_DESIGN = "on_results_writing_design"
    RESULTS_WRITING_OPERATION = "on_results_writing_operation"
