"""
Plugin enabling energy consumption for compression of gases.

This plugin allows to model the energy consumption of gas compression for different pressure levels. E.g.
the pressure level at an electrolyser outlet might need to be increased for transport.

The plugin is activated by adding the following to the Plugins.json configuration file:

.. code-block:: json

   "modeling_plugins.compression_energy":
    {
        "config":
            {
                "carriers": ["Carrier_A"]
            }
    }

The plugin also contains a helper function to copy the necessary compressor data files. You
can import the function `copy_compressor_data` from this module and use it to copy the
compressor data files to your case study data folder before running the model.

``from adopt_net0.plugins.modeling_plugins.compression_energy import copy_compressor_data``


"""
import json
import logging
from pathlib import Path
import pyomo.environ as pyo
import os

from adopt_net0.plugins.base import Plugin as PluginBase
from adopt_net0.plugins.hooks import Hook
from adopt_net0.core.data_management.utilities import open_json
from adopt_net0.core.data_preprocessing.data_loading import _copy_data
from adopt_net0.plugins.modeling_plugins.compression_energy_helpers import *
from adopt_net0.core.utilities import get_set_t

log = logging.getLogger(__name__)


def _determine_flow_existing_compressors(modelhub, compressor, b_period, node):
    """
    Determines the flow capacity of an existing compressor connection by returning
    the minimum available capacity between the output and input components

    :param compressor: tuple with carrier, component1, component 2
    :param b_period: pyomo block data for period
    :param node: pyomo block data for node
    :return float: minimum capacity between input and output component
    """
    component_output_bound = float("inf")
    component_input_bound = float("inf")
    period_name = b_period.index()
    type_component = [compressor.output_type, compressor.input_type]

    if type_component[0] == "Technology":
        var_output = (
            b_period.node_blocks[node]
            .tech_blocks_active[compressor.output_component]
            .var_output
        )
        component_output_bound = max(var_output[idx].ub for idx in var_output)
    elif type_component[0] == "Network":
        component_output_bound = 0
        for arc in b_period.network_block[
            compressor.output_component
        ].set_arcs:
            if arc[0] == node:
                component_output_bound += b_period.network_block[
                    compressor.output_component
                ].arc_block[arc].para_size_initial.value

    elif type_component[0] == "Import":
        component_output_bound = max(
            modelhub.data["time_series_data"]["full_resolution"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Import limit"]
        )

    elif type_component[0] == "Generic production":
        component_output_bound = max(
            modelhub.data["time_series_data"]["full_resolution"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Generic production"]
        )

    if type_component[1] == "Technology":
        var_output = (
            b_period.node_blocks[node]
            .tech_blocks_active[compressor.input_component]
            .var_output
        )
        component_input_bound = max(var_output[idx].ub for idx in var_output)
    elif type_component[1] == "Network":
        component_input_bound = 0
        for arc in b_period.network_block[
            compressor.input_component
        ].set_arcs:
            if arc[1] == node:
                component_input_bound += b_period.network_block[
                    compressor.input_component
                ].arc_block[arc].para_size_initial.value

    elif type_component[1] == "Demand":
        component_input_bound = max(
            modelhub.data["time_series_data"]["full_resolution"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Demand"]
        )
    elif type_component[1] == "Export":
        component_input_bound = max(
            modelhub.data["time_series_data"]["full_resolution"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Export limit"]
        )

    size = min(component_output_bound, component_input_bound)

    return size

def _create_compressor_class(connection_info: dict, carrier: str, load_path: Path):
    """
    Loads the compressor data from load_path and preprocesses it.

    :param dict connection_info: information about the connection
    :param str carrier: compressed carrier
    :param Path load_path: load path

    :return: Compressor Class
    """
    comp_data = open_json(carrier, load_path)

    comp_data["connection_info"] = connection_info
    comp_data["name"] = (
        f"{carrier}_Compressor_{comp_data['connection_info']['components'][0]}_{comp_data['connection_info']['components'][1]}"
    )
    comp_data["size"] = None

    if (
        comp_data["connection_info"]["existing"][0] == 1
        and comp_data["connection_info"]["existing"][1] == 1
    ):
        comp_data["name"] = comp_data["name"] + "_existing"
        comp_data["existing"] = 1
    else:
        comp_data["existing"] = 0

    comp_data = Compressor(comp_data)

    return comp_data


def _get_pressure_info(component, carrier: str, direction: str, type: str) -> dict:
    """
    Obtains pressure-related information for a given component, carrier, and flow direction (input/output).

    :param component: the component from which to extract pressure data
    :param str carrier: the energy carrier for which pressure data is requested
    :param str direction: either 'Input' or 'Output', specifying whether to retrieve inlet or outlet pressure

    :return dict: A dictionary containing:
            - "name": name of the component.
            - "pressure": the inlet or outlet pressure associated with the specified carrier and component.
            - "type": the type of the component ('Technology' or 'Network').
            - "existing": 1 if the component is existing; 0 otherwise.
    """
    pressure_data = component["settings_compression_energy"]["pressure_levels"]
    component_name = component["name"]
    pressure = ()
    if direction == "Input":
        pressure = pressure_data[carrier]["inlet"]
    elif direction == "Output":
        pressure = pressure_data[carrier]["outlet"]
    return {
        "name": component_name,
        "pressure": pressure,
        "type": type,
        "existing": component["existing"],
    }


def _collect_possible_connections_at_node(pressure_data_at_node: dict):
    """
    Generates all possible compression connections between output and input components at a given node.

    :param dict pressure_data_at_node: contains all components that can be inputs or outputs for compression

    :return list: containing all possible connection between input and outputs for each node, with necessary information
    """
    connection_data_at_node = []
    for output_i in pressure_data_at_node["outputs"]:
        for input_i in pressure_data_at_node["inputs"]:
            connection_data_at_node.append(
                {
                    "components": (output_i["name"], input_i["name"]),
                    "pressure": (output_i["pressure"], input_i["pressure"]),
                    "type": (output_i["type"], input_i["type"]),
                    "existing": (output_i["existing"], input_i["existing"]),
                }
            )

    return connection_data_at_node


def copy_compressor_data(folder_path: str | Path, compr_data_path: str | Path = None):
    """
    Copies compressor JSON files to the compressor_data folder for each investment period.

    This function reads the topology JSON file to determine the existing and new compressors for
    each investment period. It then searches for the corresponding JSON files in the specified `compr_data_path`
    folder (and its subfolders) using the compressor names and copies them to folder_path.

    :param str | Path folder_path: Path to the folder containing the case study data.
    :param str | Path compr_data_path: Path to the folder containing the compressors data (if left
    empty, standard folder is used).
    :return: None
    """
    plugin_file_path = folder_path / "Plugins.json"
    with open(plugin_file_path, "r") as json_file:
        plugins = json.load(json_file)
        if "modeling_plugins.compression_energy" in plugins.keys():
            # Convert to Path
            if isinstance(folder_path, str):
                folder_path = Path(folder_path)

            if compr_data_path is None:
                compr_data_path = Path(
                    os.path.join(
                        os.path.dirname(__file__)
                        + "/../database/templates/compressor_data"
                    )
                )
            else:
                if isinstance(compr_data_path, str):
                    compr_data_path = Path(compr_data_path)

            # Reads the topology JSON file
            json_file_path = folder_path / "Topology.json"
            with open(json_file_path, "r") as json_file:
                topology = json.load(json_file)

            for period in topology["investment_periods"]:
                # Read the JSON compressor file
                output_folder = folder_path / period / "compressor_data"
                (output_folder).mkdir(
                    parents=True, exist_ok=True
                )
                # Copy JSON files corresponding to compressor names to output folder
                for compr_name in plugins["modeling_plugins.compression_energy"]["config"]["carriers"]:
                    _copy_data(compr_data_path, compr_name, output_folder)


class Plugin(PluginBase):
    """
    Plugin enabling custom technologies.
    """
    name = "Compression energy for gas compression"
    hooks = {
        Hook.DATA_READ_END,
        Hook.NODE_CONSTRUCTION_END,
        Hook.ADD_TO_ENERGYBALANCE,
        Hook.ADD_TO_COSTBALANCE_CAPEX,
        Hook.ADD_TO_COSTBALANCE_OPEX,
    }
    config_template = {
      "carriers": []
    }

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.connection_pressures = {}
        self.connection_pressure_options = {}
        
    def on_data_read_end(self, modelhub):
        """
        Creates a dictionary that stores all possible compression connections for each carrier with defined pressure levels,
        at each node. It identifies all valid connections where the same carrier is used as both input and output across
        different components (e.g., technologies, networks, demand, import/export).

        The resulting dictionary contains:
            - All feasible input-output connections for the same carrier.
            - Associated pressure levels for each connection.
            - Relevant information about the two components involved in each connection (existing, type).
        """
        target_carriers = self.config["carriers"]

        self._read_pressure_connection_options(modelhub)

        for investment_period in modelhub.data["topology"]["investment_periods"]:
            self.connection_pressures[investment_period] = {}
            for node_i in modelhub.data["topology"]["nodes"]:
                self.connection_pressures[investment_period][node_i] = {}
                for carrier_i in target_carriers:
                    pressure_data_at_node = self._collect_pressure_info_at_node(
                        modelhub, carrier_i, investment_period, node_i
                    )
                    connection_data_at_node = _collect_possible_connections_at_node(
                        pressure_data_at_node
                    )

                    self.connection_pressures[investment_period][node_i][
                        carrier_i
                    ] = connection_data_at_node

        # Log success
        log_msg = "Pressure data read successfully"
        log.info(log_msg)

        self._read_compressor_data(modelhub)

    def on_node_construction_end(self, modelhub, b_node, b_period):
        """
        Adds compressor to the node block during node construction.

        :param modelhub: Modelhub.
        :param b_node: The block representing the node being constructed.
        """
        target_carriers = list(set(self.config["carriers"]))
        period = b_period.index()
        node = b_node.index()
        b_node.set_carriers_compression = pyo.Set(initialize=target_carriers)
        b_node.set_compressor = pyo.Set(initialize=list(self.compressor_data[period][node].keys()))

        # COMPRESSOR BLOCK
        def init_compressor_block(b_compr, car, comp1, comp2):
            """Pyomo rule to initialize a block holding all compressors at node"""
            compressor_constructor = self.compressor_data[period][node][b_compr.index()]
            b_compr = compressor_constructor.construct_compressor_model(
                b_compr, modelhub, b_period.set_t_full, b_period.set_t_clustered
            )
            return b_compr

        b_node.compressor_blocks_active = pyo.Block(
            b_node.set_compressor, rule=init_compressor_block
        )

        # fixing size of existing compressor based on components minimum capacity
        for node in b_period.node_blocks:
            for compr in b_period.node_blocks[node].set_compressor:
                compressor_constructor = self.compressor_data[period][node][compr]
                b_compr = b_period.node_blocks[node].compressor_blocks_active[
                    compr
                ]

                if (compressor_constructor.compression_active == 1) and (
                        compressor_constructor.existing == 1
                ):
                    size = _determine_flow_existing_compressors(
                        modelhub, compressor_constructor, b_period, node
                    )
                    compressor_constructor.fix_size(b_compr, size)

        self._construct_compressor_constrains(modelhub, period, node)

    def on_energybalance_construction(self, b_period: pyo.Block, node:str, t: int, carrier: str) -> pyo.Expression | int:
        """
        Sums over all inputs and outputs of all retrofits at current period, node, and timestep

        :return: Pyomo expression to be added to energy balance
        """
        delta_output = 0
        for compr in b_period.node_blocks[node].set_compressor:
            b_compressor = b_period.node_blocks[node].compressor_blocks_active[
                compr
            ]
            if carrier in b_compressor.set_consumed_carriers:
                delta_output -= b_compressor.var_consumption_energy[t, carrier]

        return delta_output


    def on_costbalance_capex_construction(self, model: pyo.Block, period: str) -> pyo.Expression | int:
        """
        Sums over all var_delta_capex in all retrofits at current period for all technologies

        :return: Pyomo expression to be added to cost balance
        """
        capex_compressors = 0
        b_period = model.periods[period]
        for node in model.set_nodes:
            b_node = b_period.node_blocks[node]
            for compressor in b_node.set_compressor:
                print(compressor)
                capex_compressors += b_node.compressor_blocks_active[compressor].var_capex

        return capex_compressors


    def on_costbalance_opex_construction(self, model: pyo.Block, period: str) -> pyo.Expression | int:
        """
        Sums over all var_delta_opex_var and var_delta_opex_fix in all retrofits at current period for all technologies

        :return: Pyomo expression to be added to cost balance
        """
        opex_compressors = 0
        b_period = model.periods[period]
        for node in model.set_nodes:
            b_node = b_period.node_blocks[node]
            for compressor in b_node.set_compressor:
                print(compressor)
                opex_compressors += b_node.compressor_blocks_active[compressor].var_opex_variable
                opex_compressors += b_node.compressor_blocks_active[compressor].var_opex_fixed

        return opex_compressors

    def on_results_writing_design(self, modelhub, model, period, node, h5_node_group):
        """
        Writes design results of retrofits to h5 file

        :param technology_constructor: Technology constructor
        :param b_tec: Pyomo technology block
        :param h5_group: H5 technology design group
        :return:
        """
        b_node = model.periods[period].node_blocks[node]
        compressor_group = h5_node_group.create_group("compressors")

        for compr_name in b_node.set_compressor:
            b_compr = b_node.compressor_blocks_active[compr_name]
            compressor = self.compressor_data[period][node][compr_name]
            if compressor.compression_active == 1:
                compr_group = compressor_group.create_group(
                    compressor.name_compressor
                )
                compressor.write_results_compressor_design(
                    compr_group, b_compr
                )

    def on_results_writing_operation(self, modelhub, model, period, node, h5_node_group):
        """
        Writes operational results of retrofits to h5 file

        :param technology_constructor: Technology constructor
        :param b_tec: Pyomo technology block
        :param h5_group: H5 technology operation group
        :return:
        """
        b_node = model.periods[period].node_blocks[node]
        compressor_group = h5_node_group.create_group("compressors")

        for compr_name in b_node.set_compressor:
            b_compr = b_node.compressor_blocks_active[compr_name]
            compressor = self.compressor_data[period][node][compr_name]
            if compressor.compression_active == 1:
                compr_group = compressor_group.create_group(
                    compressor.name_compressor
                )
                compressor.write_results_compressor_operation(
                    compr_group, b_compr
                )

    def _collect_pressure_info_at_node(self, modelhub, carrier_i, investment_period, node_i):
        """
        Collects all possible connections at a given node for a specific carrier and investment period.
        This includes connections where the same carrier is used as input and output
        across technologies, networks, demand, import, and export.

        :param carrier_i: energy carrier for which connections are being collected
        :param investment_period: investment period under consideration
        :param node_i: node at which connections are being created
        """
        pressure_data_at_node = {
            "inputs": [],
            "outputs": [],
        }

        for _, network_i in modelhub.data["network_data"][investment_period].items():
            if carrier_i in network_i["Performance"]["carrier"]:
                    if network_i["connection"].loc[node_i, :].sum() >= 1:
                        # means that there is a network starting in this node
                        pressure_data_at_node["inputs"].append(
                            _get_pressure_info(network_i, carrier_i, "Input", "Network")
                        )

                    if network_i["connection"].loc[:, node_i].sum() >= 1:
                        # means that there is a network arriving at this node
                        pressure_data_at_node["outputs"].append(
                            _get_pressure_info(network_i, carrier_i, "Output", "Network")
                        )

        technologies_by_node = modelhub.data["technology_data"][investment_period][node_i]
        for _, technologies_i in technologies_by_node.items():
            # We look at the one that has the gas as input
            if carrier_i in technologies_i["Performance"]["input_carrier"]:
                pressure_data_at_node["inputs"].append(
                    _get_pressure_info(technologies_i, carrier_i, "Input", "Technology")
                )

            if carrier_i in technologies_i["Performance"]["output_carrier"]:
                pressure_data_at_node["outputs"].append(
                    _get_pressure_info(technologies_i, carrier_i, "Output", "Technology")
                )

        info_node_exchange_pressure = self.connection_pressure_options[
            investment_period
        ][node_i][carrier_i]

        if (
                modelhub.data["time_series_data"]["full_resolution"][investment_period][node_i]["CarrierData"][
                    carrier_i
                ]["Demand"].any()
                != 0
        ):
            pressure_data_at_node["inputs"].append(
                {
                    "name": "Demand",
                    "pressure": (info_node_exchange_pressure["Demand"]["value"]),
                    "type": "Demand",
                    "existing": 1,
                }
            )

        if (
                modelhub.data["time_series_data"]["full_resolution"][investment_period][node_i]["CarrierData"][
                    carrier_i
                ]["Export limit"].any()
                != 0
        ):
            pressure_data_at_node["inputs"].append(
                {
                    "name": "Export",
                    "pressure": (info_node_exchange_pressure["Export"]["value"]),
                    "type": "Export",
                    "existing": 1,
                }
            )

        if (
                modelhub.data["time_series_data"]["full_resolution"][investment_period][node_i]["CarrierData"][
                    carrier_i
                ]["Import limit"].any()
                != 0
        ):
            pressure_data_at_node["outputs"].append(
                {
                    "name": "Import",
                    "pressure": (info_node_exchange_pressure["Import"]["value"]),
                    "type": "Import",
                    "existing": 1,
                }
            )

        if (
                modelhub.data["time_series_data"]["full_resolution"][investment_period][node_i]["CarrierData"][
                    carrier_i
                ]["Generic production"].any()
                != 0
        ):
            pressure_data_at_node["outputs"].append(
                {
                    "name": "Generic production",
                    "pressure": (
                        info_node_exchange_pressure["Generic production"]["value"]
                    ),
                    "type": "Generic production",
                    "existing": 1,
                }
            )

        return pressure_data_at_node

    def _read_pressure_connection_options(self, modelhub):
        """
        Reads connection pressure options for demand, import and export
        """
        for investment_period in modelhub.data["topology"]["investment_periods"]:
            self.connection_pressure_options[investment_period] = {}
            for node in modelhub.data["topology"]["nodes"]:
                with open(
                        modelhub.data["data_path"]
                        / investment_period
                        / "node_data"
                        / node
                        / "carrier_data"
                        / "PressureExchangeData.json"
                ) as json_file:
                    connection_pressure_options = json.load(json_file)

                    # Check for correct data
                    for carrier, connections in connection_pressure_options.items():
                        for connection_type, value in connections.items():
                            if not isinstance(value["value"], (int, float)):
                                raise ValueError(
                                    f"Invalid pressure value at node '{node}',"
                                    f" carrier '{carrier}',"
                                    f" connection '{connection_type}'"
                                    f": {value['value']}"
                                )

                self.connection_pressure_options[investment_period][
                    node
                ] = connection_pressure_options

        # Log success
        log_msg = "Connection pressure options read successfully"
        log.info(log_msg)


    def _read_compressor_data(self, modelhub):
        """
        Reads all compressor data and fits it
        """
        # compressor data always fitted based on full resolution
        aggregation_model = "full_resolution"
        target_carriers = self.config["carriers"]

        # Initialize technology_data dict
        compressor_data = {}

        # Loop through all investment_periods, carriers, nodes
        for investment_period in modelhub.data["topology"]["investment_periods"]:
            compressor_data[investment_period] = {}
            for node_i in modelhub.data["topology"]["nodes"]:
                compressor_data[investment_period][node_i] = {}
                for carrier_i in target_carriers:
                    # Compressor
                    for compressor_i in self.connection_pressures[investment_period][
                        node_i
                    ][carrier_i]:
                        comp_data = _create_compressor_class(
                            compressor_i,
                            carrier_i,
                            modelhub.data["data_path"] / investment_period / "compressor_data",
                        )
                        comp_data.fit_compressor_performance()
                        compressor_data[investment_period][node_i][
                            (
                                carrier_i,
                                compressor_i["components"][0],
                                compressor_i["components"][1],
                            )
                        ] = comp_data

        self.compressor_data = compressor_data

        # Log success
        log_msg = "Compressor data read successfully"
        log.info(log_msg)

    def _construct_compressor_constrains(self, modelhub, period, node):
        """
        Construct the compressor constraints to calculate inflow and outflow for each compressor.

        For each **period**, **node**, and **carrier**, compressor flows must satisfy the following constraints.

        Technologies
        ------------

        .. math::

           Output\_tech_{tec}(t, car) =
           \sum_{i \in C^{in}_{node,car,\,tec}} flow^{comp}_i(t, car)

        .. math::

           Input\_tech_{tec}(t, car) =
           \sum_{i \in C^{out}_{node,car,\,tec}} flow^{comp}_i(t, car)


        Networks
        --------

        .. math::

           Input\_{netw}(t, car) =
           \sum_{i \in C^{in}_{node,car,\,netw}} flow^{comp}_i(t, car)

        .. math::

           Output\_{netw}(t, car) =
           \sum_{i \in C^{out}_{node,car,\,netw}} flow^{comp}_i(t, car)


        Demand
        ------

        .. math::

           Demand(t, car) =
           \sum_{i \in C^{out}_{node,car,\,demand}} flow^{comp}_i(t, car)


        Export
        ------

        .. math::

           Export(t, car) =
           \sum_{i \in C^{out}_{node,car,\,export}} flow^{comp}_i(t, car)


        Import
        ------

        .. math::

           Import(t, car) =
           \sum_{i \in C^{in}_{node,car,\,import}} flow^{comp}_i(t, car)


        Generic Production
        ------------------

        .. math::

           Generic Production(t, car) =
           \sum_{i \in C^{in}_{node,car,\,genProd}} flow^{comp}_i(t, car)

        Notation
        --------
        - :math:`t` = time step
        - :math:`car` = carrier (e.g. hydrogen, methane, etc.)
        - :math:`tec` = technology at the node
        - :math:`netw` = network at the node
        - :math:`flow^{comp}_i(t, car)` = flow through compressor :math:`i` at time :math:`t` for carrier :math:`car`
        - :math:`C^{in}_{node,car,x}` = set of compressors at the node that provide inflow to component :math:`x` for :math:`car`
        - :math:`C^{out}_{node,car,x}` = set of compressors at the node that provide outflow to component :math:`x` for :math:`car`
        --------

        :param model: pyomo model
        :param dict config: dict containing model information
        :return: pyomo model
        """
        config = modelhub.data["config"]
        model = modelhub.model[modelhub.info_solving_algorithms["aggregation_model"]]

        b_period = model.periods[period]
        b_node = b_period.node_blocks[node]
        set_t = get_set_t(config, model.periods[period])

        def init_compressor_constraints(b_compr_const, car):
            """Pyomo rule to generate compressor constraint block"""
            if car in b_node.set_carriers_compression:

                def init_compr_inflow_tec(const, tec, t):
                    """Define constrain for the flow input to compressor from technology"""
                    if car in b_node.tech_blocks_active[tec].set_output_carriers:
                        return b_node.tech_blocks_active[tec].var_output[t, car] == sum(
                            b_node.compressor_blocks_active[compressor].var_flow[t]
                            for compressor in b_node.set_compressor
                            if (compressor[0] == car) and (compressor[1] == tec)
                        )
                    else:
                        return pyo.Constraint.Skip

                b_compr_const.const_compr_inflow_tec = pyo.Constraint(
                    b_node.set_technologies, set_t, rule=init_compr_inflow_tec
                )

                def init_compr_inflow_netw(const, netw, t):
                    """Define constrain for the flow input to compressor from network"""
                    if car in b_period.network_block[netw].set_netw_carrier:
                        relevant_compressors = [
                            compressor
                            for compressor in b_node.set_compressor
                            if (compressor[0] == car) and (compressor[1] == netw)
                        ]
                        if not relevant_compressors:
                            return pyo.Constraint.Skip
                        return b_period.network_block[netw].var_inflow[t, car, node] == sum(
                            b_node.compressor_blocks_active[compressor].var_flow[t]
                            for compressor in relevant_compressors
                        )
                    else:
                        return pyo.Constraint.Skip

                b_compr_const.const_compr_inflow_netw = pyo.Constraint(
                    b_period.set_networks, set_t, rule=init_compr_inflow_netw
                )

                def init_compr_outflow_tec(const, tec, t):
                    """Define constrain for the flow output from compressor to technology"""
                    if car in b_node.tech_blocks_active[tec].set_input_carriers:
                        return b_node.tech_blocks_active[tec].var_input[t, car] == sum(
                            b_node.compressor_blocks_active[compressor].var_flow[t]
                            for compressor in b_node.set_compressor
                            if (compressor[0] == car) and (compressor[2] == tec)
                        )
                    else:
                        return pyo.Constraint.Skip

                b_compr_const.const_compr_outflow_tec = pyo.Constraint(
                    b_node.set_technologies, set_t, rule=init_compr_outflow_tec
                )

                def init_compr_outflow_netw(const, netw, t):
                    """Define constrain for the flow output from compressor to network"""
                    if car in b_period.network_block[netw].set_netw_carrier:
                        relevant_compressors = [
                            compressor
                            for compressor in b_node.set_compressor
                            if (compressor[0] == car) and (compressor[2] == netw)
                        ]
                        if not relevant_compressors:
                            return pyo.Constraint.Skip
                        return b_period.network_block[netw].var_outflow[
                            t, car, node
                        ] == sum(
                            b_node.compressor_blocks_active[compressor].var_flow[t]
                            for compressor in relevant_compressors
                        )
                    else:
                        return pyo.Constraint.Skip

                b_compr_const.const_compr_outflow_netw = pyo.Constraint(
                    b_period.set_networks, set_t, rule=init_compr_outflow_netw
                )

                def init_compr_outflow_demand(const, t):
                    """Define constrain for the flow output from compressor to demand"""
                    if any(
                        compressor[0] == car and compressor[2] == "Demand"
                        for compressor in b_node.set_compressor
                    ):
                        return b_node.para_demand[t, car] == sum(
                            b_node.compressor_blocks_active[compressor].var_flow[t]
                            for compressor in b_node.set_compressor
                            if (compressor[0] == car) and (compressor[2] == "Demand")
                        )
                    else:
                        return pyo.Constraint.Skip

                b_compr_const.const_compr_outflow_demand = pyo.Constraint(
                    set_t, rule=init_compr_outflow_demand
                )

                def init_compr_outflow_export(const, t):
                    """Define constrain for the flow output from compressor to export"""
                    if any(
                        compressor[0] == car and compressor[2] == "Export"
                        for compressor in b_node.set_compressor
                    ):
                        return b_node.var_export_flow[t, car] == sum(
                            b_node.compressor_blocks_active[compressor].var_flow[t]
                            for compressor in b_node.set_compressor
                            if (compressor[0] == car) and (compressor[2] == "Export")
                        )
                    else:
                        return pyo.Constraint.Skip

                b_compr_const.const_compr_outflow_export = pyo.Constraint(
                    set_t, rule=init_compr_outflow_export
                )

                def init_compr_inflow_import(const, t):
                    """Define constrain for the flow input to compressor from import"""
                    if any(
                        compressor[0] == car and compressor[2] == "Import"
                        for compressor in b_node.set_compressor
                    ):
                        return b_node.var_import_flow[t, car] == sum(
                            b_node.compressor_blocks_active[compressor].var_flow[t]
                            for compressor in b_node.set_compressor
                            if (compressor[0] == car) and (compressor[1] == "Import")
                        )
                    else:
                        return pyo.Constraint.Skip

                b_compr_const.const_compr_inflow_import = pyo.Constraint(
                    set_t, rule=init_compr_inflow_import
                )

                def init_compr_generic_production(const, t):
                    """Define constrain for the flow input to compressor from generic production"""
                    if any(
                        compressor[0] == car and compressor[2] == "Generic production"
                        for compressor in b_node.set_compressor
                    ):
                        return b_node.var_generic_production[t, car] == sum(
                            b_node.compressor_blocks_active[compressor].var_flow[t]
                            for compressor in b_node.set_compressor
                            if (compressor[0] == car)
                            and (compressor[1] == "Generic production")
                        )
                    else:
                        return pyo.Constraint.Skip

                b_compr_const.const_compr_inflow_generic_production = pyo.Constraint(
                    set_t, rule=init_compr_generic_production
                )

            else:
                return pyo.Block.Skip

        b_node.block_compressor_constraints = pyo.Block(
            model.set_carriers,
            rule=init_compressor_constraints,
        )