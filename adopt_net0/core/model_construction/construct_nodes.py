import numpy as np
import pandas as pd
import pyomo.environ as pyo

from adopt_net0.core.utilities import get_used_timeseries


def _determine_carriers_from_time_series(time_series: pd.DataFrame) -> list:
    """
    Determines carriers that are used in time_series

    :param pd.Dataframe time_series: Time series
    """
    carriers = []
    series = ["Demand", "Import limit", "Export limit", "Generic production"]
    for car in time_series.columns.get_level_values("Key2").unique():
        if np.any(time_series[car][series]):
            carriers.append(car)

    return list(set(carriers))


def _determine_carriers_from_technologies(technology_data: dict) -> list:
    """
    Determines carriers that are used for technologies

    :param dict technology_data: Dict with technology data
    :return: list of carriers used in technologies
    """
    carriers = []
    for tec in technology_data:
        input_carriers = technology_data[tec].input_carrier
        carriers.extend(input_carriers)
        output_carriers = technology_data[tec].output_carrier
        carriers.extend(output_carriers)

    return list(set(carriers))


def _determine_carriers_from_networks(network_data) -> list:
    """
    Determines carriers that are used for networks

    :param dict network_data: Dict with network data
    :return: list of carriers used in networks
    """
    carriers = []
    for netw in network_data:
        # Todo: This can be further extended to check if node is connected to network
        carriers.extend([network_data[netw].transported_carrier])

        if network_data[netw].energy_consumption:
            carriers.extend(network_data[netw].energy_consumption.keys())

    return list(set(carriers))


def construct_node_block(b_node, modelhub, period, set_t_full, set_t_clustered):
    """
    Adds all nodes with respective data to the model

    This function initializes parameters and decision variables for all considered
    nodes.

    **Set declarations:**

    - set_technologies: Set for all technologies at respective node
    - set_carriers: Set of carriers used at node (this is a subset of all carriers)
    - set_compressor; Set for all compressors needed at respective node

    **Parameter declarations:**

    - para_demand: Demand for each time step and carrier
    - para_production_profile: Maximal generic production profile for each time step
      and carrier
    - para_import_price: Import Prices for each time step and carrier
    - para_export_price: Export Prices for each time step and carrier
    - para_import_limit: Import Limits for each time step and carrier
    - para_export_limit: Export Limits for each time step and carrier
    - para_import_emissionfactors: Emission factors of imports for each time step and
      carrier
    - para_export_emissionfactors: Emission factors of exports for each time step and
      carrier
    - para_carbon_subsidy: Carbon subsidy for negative emissions
    - para_carbon_tax: Carbon tax for positive emissions

    **Variable declarations:**

    - var_import_flow: Import Flow for each time step and carrier
    - var_export_flow: Export Flow for each time step and carrier
    - var_netw_inflow: Network Inflow for each time step and carrier
    - var_netw_outflow: Network Outflow for each time step and carrier
    - var_netw_consumption: Network consumption (if present)
    - var_generic_production: Actual generic production
    - var_import_emissions_pos: Positive emissions from imports
    - var_import_emissions_neg: Negative emissions from imports
    - var_export_emissions_pos: Positive emissions from exports
    - var_export_emissions_neg: Negative emissions from exports
    - var_car_emissions_pos: Sum of positive emissions
    - var_car_emissions_neg: Sum of negative emissions

    **Constraint declarations**

    - Generic production: Equal to the parameter if curtailment is not possible,
      otherwise less-or-equal
    - Calculate import emissions (positive and negative): emissions = import *
      emissions-factor
    - Calculate export emissions (positive and negative): emissions = export *
      emissions-factor
    - Calculate carrier emissions as a sum of positive/negative import and export
      emissions

    :param b_node: pyomo block with node model
    :param dict data: data containing model configuration
    :param set_t_full: pyomo set containing full resolution timesteps
    :param set_t_clustered: pyomo set containing clustered resolution timesteps
    :return: pyomo block with node model
    """

    # PREPROCESSING
    # Collect data for node and period
    node = b_node.index()

    config = modelhub.data["config"]
    time_series_at_node = get_used_timeseries(modelhub.data)[period][node]

    # Determine carriers used at node
    carriers_from_time_series = _determine_carriers_from_time_series(time_series_at_node["CarrierData"])
    carriers_from_technologies = _determine_carriers_from_technologies(modelhub.component_constructors["technology_constructors"][period][node])
    if not config["energybalance"]["copperplate"]["value"]:
        carriers_from_network = _determine_carriers_from_networks(modelhub.component_constructors["network_constructors"][period])
    else:
        carriers_from_network = []

    carriers_at_node = list(set(carriers_from_time_series + carriers_from_technologies + carriers_from_network))

    # SETS
    b_node.set_technologies = pyo.Set(initialize=list(modelhub.component_constructors["technology_constructors"][period][node].keys()))
    b_node.set_carriers = pyo.Set(initialize=list(set(carriers_at_node)))

    # Todo: move to plugin
    # if config["performance"]["pressure"]["pressure_on"]["value"] == 1:
    #     target_carriers = config["performance"]["pressure"]["pressure_carriers"][
    #         "value"
    #     ]
    #     target_carriers = list(set(target_carriers))
    #     b_node.set_carriers_compression = pyo.Set(initialize=target_carriers)
    #     b_node.set_compressor = pyo.Set(initialize=list(data["compressor_data"].keys()))

    # Time aggregation
    if config["optimization"]["typicaldays"]["N"]["value"] == 0:
        set_t = set_t_full
    elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
        set_t = set_t_clustered
    elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
        set_t = set_t_full

    # PARAMETERS
    def create_carrier_parameter(key, par_mutable=False):
        # Convert to dict/list for performance
        ts = {}
        for car in b_node.set_carriers:
            ts[car] = {}
            ts[car][key] = time_series_at_node["CarrierData"][car][key].to_list()

        def init_carrier_parameter(para, t, car):
            """Rule initiating a carrier parameter"""
            return ts[car][key][t - 1]

        parameter = pyo.Param(
            set_t, b_node.set_carriers, rule=init_carrier_parameter, mutable=par_mutable
        )
        return parameter

    def create_carbonprice_parameter(key):
        # Convert to dict/list for performance
        ts = time_series_at_node["CarbonCost"]["global"][key].to_list()

        def init_carbonprice_parameter(para, t):
            """Rule initiating a carrier parameter"""
            return ts[t - 1]

        parameter = pyo.Param(set_t, rule=init_carbonprice_parameter, mutable=False)
        return parameter

    par_mutable = False

    b_node.para_demand = create_carrier_parameter("Demand")
    b_node.para_production_profile = create_carrier_parameter("Generic production")
    b_node.para_import_price = create_carrier_parameter(
        "Import price", par_mutable=par_mutable
    )
    b_node.para_export_price = create_carrier_parameter(
        "Export price", par_mutable=par_mutable
    )
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

    b_node.var_import_flow = pyo.Var(
        set_t, b_node.set_carriers, bounds=init_import_bounds
    )

    def init_export_bounds(var, t, car):
        return (0, b_node.para_export_limit[t, car])

    b_node.var_export_flow = pyo.Var(
        set_t, b_node.set_carriers, bounds=init_export_bounds
    )

    if not config["energybalance"]["copperplate"]["value"]:
        b_node.var_netw_inflow = pyo.Var(set_t, b_node.set_carriers)
        b_node.var_netw_outflow = pyo.Var(set_t, b_node.set_carriers)

        b_node.var_netw_consumption = pyo.Var(set_t, b_node.set_carriers)

    # Generic production profile
    b_node.var_generic_production = pyo.Var(
        set_t, b_node.set_carriers, within=pyo.NonNegativeReals
    )

    # Emissions
    b_node.var_import_emissions_pos = pyo.Var(set_t, b_node.set_carriers)
    b_node.var_import_emissions_neg = pyo.Var(set_t, b_node.set_carriers)
    b_node.var_export_emissions_pos = pyo.Var(set_t, b_node.set_carriers)
    b_node.var_export_emissions_neg = pyo.Var(set_t, b_node.set_carriers)
    b_node.var_car_emissions_pos = pyo.Var(set_t, within=pyo.NonNegativeReals)
    b_node.var_car_emissions_neg = pyo.Var(set_t, within=pyo.NonNegativeReals)

    # CONSTRAINTS
    # Generic production constraint
    def init_generic_production(const, t, car):
        if config["node_config"][period][node][car]["curtailment_possible"] == 0:
            return (
                b_node.para_production_profile[t, car]
                == b_node.var_generic_production[t, car]
            )
        elif config["node_config"][period][node][car]["curtailment_possible"] == 1:
            return (
                b_node.para_production_profile[t, car]
                >= b_node.var_generic_production[t, car]
            )

    b_node.const_generic_production = pyo.Constraint(
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

    b_node.const_import_emissions_pos = pyo.Constraint(
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

    b_node.const_export_emissions_pos = pyo.Constraint(
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

    b_node.const_import_emissions_neg = pyo.Constraint(
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

    b_node.const_export_emissions_neg = pyo.Constraint(
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

    b_node.const_car_emissions_pos = pyo.Constraint(set_t, rule=init_car_emissions_pos)

    def init_car_emissions_neg(const, t):
        return (
            sum(
                b_node.var_import_emissions_neg[t, car]
                + b_node.var_export_emissions_neg[t, car]
                for car in b_node.set_carriers
            )
            == b_node.var_car_emissions_neg[t]
        )

    b_node.const_car_emissions_neg = pyo.Constraint(set_t, rule=init_car_emissions_neg)

    return b_node
