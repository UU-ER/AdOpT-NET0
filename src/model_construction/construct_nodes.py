import time

import numpy as np
from pyomo.environ import *


def determine_carriers_from_time_series(time_series):
    """
    Determines carriers that are used in time_series
    """
    carriers = []
    for car in time_series.columns.get_level_values("Carrier"):
        if np.any(time_series[car]):
            carriers.append(car)

    return list(set(carriers))


def determine_carriers_from_technologies(technology_data):
    """
    Determines carriers that are used for technologies
    """
    carriers = []
    for tec in technology_data:
        if "input_carrier" in technology_data[tec].performance_data:
            input_carriers = technology_data[tec].performance_data["input_carrier"]
            carriers.extend(input_carriers)
        output_carriers = technology_data[tec].performance_data["output_carrier"]
        carriers.extend(output_carriers)

    return list(set(carriers))


def determine_carriers_from_networks(network_data):
    """
    Determines carriers that are used for networks
    """
    carriers = []
    for netw in network_data:
        # Todo: This can be further extended to check if node is connected to network
        # Todo: This needs to be written correctly, possibly its buggy, check if energy consumption works
        # Todo: This does not work for copperplate
        for car in network_data[netw].performance_data["carrier"]:
            carriers.append(car)
        for car in network_data[netw].energyconsumption["carrier"]:
            carriers.append(car)

    return list(set(carriers))


def determine_network_energy_consumption(network_data):
    """
    Determines if there is network consumption for a network
    """
    # Todo: This can be further extended to check if node is connected to network
    network_energy_consumption = 0
    for netw in network_data:
        if not network_data[netw].energyconsumption["carrier"]:
            network_energy_consumption = 1

    return network_energy_consumption


def construct_node_block(b_node, data, set_t):
    r"""
    Adds all nodes with respective data to the model

    This function initializes parameters and decision variables for all considered nodes. It also adds all technologies\
    that are installed at the node (see :func:`~add_technologies`). For each node, it adds one block indexed by the \
    set of all nodes. As such, the function constructs:

    node blocks, indexed by :math:`N` > technology blocks, indexed by :math:`Tec_n, n \in N`

    **Set declarations:**

    - Set for all technologies :math:`S_n` at respective node :math:`n` : :math:`S_n, n \in N` (this is a duplicate \
      of a set already initialized in ``self.model.set_technologies``).

    **Parameter declarations:**

    - Demand for each time step
    - Import Prices for each time step
    - Export Prices for each time step
    - Import Limits for each time step
    - Export Limits for each time step
    - Emission Factors for each time step

    **Variable declarations:**

    - Import Flow for each time step
    - Export Flow for each time step
    - Network Inflow for each time step
    - Network Outflow for each time step
    - Cost at node (includes technology costs (CAPEX, OPEX) and import/export costs), see constraint declarations

    **Constraint declarations**

    - Cost at node:

    .. math::
        C_n = \
        \sum_{tec \in Tec_n} CAPEX_{tec} + \
        \sum_{tec \in Tec_n} OPEXfix_{tec} + \
        \sum_{tec \in Tec_n} \sum_{t \in T} OPEXvar_{t, tec} + \\
        \sum_{car \in Car} \sum_{t \in T} import_{t, car} pImport_{t, car} - \
        \sum_{car \in Car} \sum_{t \in T} export_{t, car} pExport_{t, car}

    **Block declarations:**

    - Technologies at node

    :param EnergyHub energyhub: instance of the energyhub
    :return: model
    """

    # PREPROCESSING
    # Collect data for node and period
    config = data["config"]

    # Determine carriers used at node
    carriers = []
    carriers.extend(
        determine_carriers_from_time_series(data["time_series"]["CarrierData"])
    )
    carriers.extend(determine_carriers_from_technologies(data["technology_data"]))
    if not config["energybalance"]["copperplate"]["value"]:
        carriers.extend(determine_carriers_from_networks(data["network_data"]))
    carriers = list(set(carriers))

    # SETS
    b_node.set_technologies = Set(initialize=list(data["technology_data"].keys()))
    b_node.set_carriers = Set(initialize=list(set(carriers)))

    # PARAMETERS
    def create_carrier_parameter(key):
        # Convert to dict/list for performance
        ts = {}
        for car in b_node.set_carriers:
            ts[car] = {}
            ts[car][key] = data["time_series"]["CarrierData"][car][key].to_list()

        def init_carrier_parameter(para, t, car):
            """Rule initiating a carrier parameter"""
            return ts[car][key][t - 1]

        parameter = Param(
            set_t, b_node.set_carriers, rule=init_carrier_parameter, mutable=False
        )
        return parameter

    def create_carbonprice_parameter(key):
        # Convert to dict/list for performance
        ts = data["time_series"]["CarbonCost"][:][key].to_list()

        def init_carbonprice_parameter(para, t):
            """Rule initiating a carrier parameter"""
            return ts[t - 1]

        parameter = Param(set_t, rule=init_carbonprice_parameter, mutable=False)
        return parameter

    b_node.para_demand = create_carrier_parameter("Demand")
    b_node.para_production_profile = create_carrier_parameter("Generic production")
    b_node.para_import_price = create_carrier_parameter("Import price")
    b_node.para_export_price = create_carrier_parameter("Export price")
    b_node.para_import_limit = create_carrier_parameter("Import limit")
    b_node.para_export_limit = create_carrier_parameter("Export limit")
    b_node.para_import_emissionfactors = create_carrier_parameter(
        "Import emission factor"
    )
    b_node.para_export_emissionfactors = create_carrier_parameter(
        "Export emission factor"
    )
    b_node.para_carbon_subsidy = create_carbonprice_parameter("subsidy")
    b_node.para_carbon_tax = create_carbonprice_parameter("price")

    # VARIABLES
    def init_import_bounds(var, t, car):
        return (0, b_node.para_import_limit[t, car])

    b_node.var_import_flow = Var(set_t, b_node.set_carriers, bounds=init_import_bounds)

    def init_export_bounds(var, t, car):
        return (0, b_node.para_export_limit[t, car])

    b_node.var_export_flow = Var(set_t, b_node.set_carriers, bounds=init_export_bounds)

    if not config["energybalance"]["copperplate"]["value"]:
        b_node.var_netw_inflow = Var(set_t, b_node.set_carriers)
        b_node.var_netw_outflow = Var(set_t, b_node.set_carriers)
        network_energy_consumption = determine_network_energy_consumption(
            data["network_data"]
        )
        if network_energy_consumption:
            b_node.var_netw_consumption = Var(set_t, b_node.set_carriers)

    # Generic production profile
    b_node.var_generic_production = Var(
        set_t, b_node.set_carriers, within=NonNegativeReals
    )

    # Emissions
    b_node.var_import_emissions_pos = Var(set_t, b_node.set_carriers)
    b_node.var_import_emissions_neg = Var(set_t, b_node.set_carriers)
    b_node.var_export_emissions_pos = Var(set_t, b_node.set_carriers)
    b_node.var_export_emissions_neg = Var(set_t, b_node.set_carriers)
    b_node.var_car_emissions_pos = Var(set_t, within=NonNegativeReals)
    b_node.var_car_emissions_neg = Var(set_t, within=NonNegativeReals)

    # CONSTRAINTS
    # Generic production constraint
    def init_generic_production(const, t, car):
        if data["energybalance_options"][car]["curtailment_possible"] == 0:
            return (
                b_node.para_production_profile[t, car]
                == b_node.var_generic_production[t, car]
            )
        elif data["energybalance_options"][car]["curtailment_possible"] == 1:
            return (
                b_node.para_production_profile[t, car]
                >= b_node.var_generic_production[t, car]
            )

    b_node.const_generic_production = Constraint(
        set_t, b_node.set_carriers, rule=init_generic_production
    )

    # Emission constraints
    def init_import_emissions_pos(const, t, car):
        if b_node.para_import_emissionfactors[t, car] >= 0:
            return (
                b_node.var_import_flow[t, car]
                * b_node.para_import_emissionfactors[t, car]
                == b_node.var_import_emissions_pos[t, car]
            )
        else:
            return 0 == b_node.var_import_emissions_pos[t, car]

    b_node.const_import_emissions_pos = Constraint(
        set_t, b_node.set_carriers, rule=init_import_emissions_pos
    )

    def init_export_emissions_pos(const, t, car):
        if b_node.para_export_emissionfactors[t, car] >= 0:
            return (
                b_node.var_export_flow[t, car]
                * b_node.para_export_emissionfactors[t, car]
                == b_node.var_export_emissions_pos[t, car]
            )
        else:
            return 0 == b_node.var_export_emissions_pos[t, car]

    b_node.const_export_emissions_pos = Constraint(
        set_t, b_node.set_carriers, rule=init_export_emissions_pos
    )

    def init_import_emissions_neg(const, t, car):
        if b_node.para_import_emissionfactors[t, car] < 0:
            return (
                b_node.var_import_flow[t, car]
                * (-b_node.para_import_emissionfactors[t, car])
                == b_node.var_import_emissions_neg[t, car]
            )
        else:
            return 0 == b_node.var_import_emissions_neg[t, car]

    b_node.const_import_emissions_neg = Constraint(
        set_t, b_node.set_carriers, rule=init_import_emissions_neg
    )

    def init_export_emissions_neg(const, t, car):
        if b_node.para_export_emissionfactors[t, car] < 0:
            return (
                b_node.var_export_flow[t, car]
                * (-b_node.para_export_emissionfactors[t, car])
                == b_node.var_export_emissions_neg[t, car]
            )
        else:
            return 0 == b_node.var_export_emissions_neg[t, car]

    b_node.const_export_emissions_neg = Constraint(
        set_t, b_node.set_carriers, rule=init_export_emissions_neg
    )

    def init_car_emissions_pos(const, t):
        return (
            sum(
                b_node.var_import_emissions_pos[t, car]
                + b_node.var_export_emissions_pos[t, car]
                for car in b_node.set_carriers
            )
            == b_node.var_car_emissions_pos[t]
        )

    b_node.const_car_emissions_pos = Constraint(set_t, rule=init_car_emissions_pos)

    def init_car_emissions_neg(const, t):
        return (
            sum(
                b_node.var_import_emissions_neg[t, car]
                + b_node.var_export_emissions_neg[t, car]
                for car in b_node.set_carriers
            )
            == b_node.var_car_emissions_neg[t]
        )

    b_node.const_car_emissions_neg = Constraint(set_t, rule=init_car_emissions_neg)

    return b_node
