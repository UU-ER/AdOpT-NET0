from enum import Enum


class Hook(Enum):
    DATA_READ_START = "on_data_read_start"
    # Custom modle component registration
    TECHNOLOGY_REGISTRATION = "technology_registration"
    NETWORK_REGISTRATION = "network_registration"
    # Technology construction hooks
    TECHNOLOGY_CONSTRUCTION_END = "on_technology_construction_end"
    # Balance construction hooks
    ADD_TO_ENERGYBALANCE = "on_energybalance_construction"
    ADD_TO_COSTBALANCE_CAPEX = "on_costbalance_capex_construction"
    ADD_TO_COSTBALANCE_OPEX = "on_costbalance_opex_construction"
    ADD_TO_EMISSIONBALANCE = "on_emissionbalance_construction"
    # Result writing hooks
    TECHNOLOGY_RESULTS_WRITING_DESIGN = "on_technology_results_writing_design"
    TECHNOLOGY_RESULTS_WRITING_OPERATION = "on_technology_results_writing_operation"
