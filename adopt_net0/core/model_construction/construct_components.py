import logging
import time

import pyomo.environ as pyo

from .construct_investment_period import construct_investment_period_block
from .construct_nodes import construct_node_block
from .construct_networks import construct_network_block
from .construct_technology import construct_technology_block
from .construct_compressor import construct_compressor_block
from .utilities import get_data_for_node
from adopt_net0.core.utilities import get_data_for_investment_period, determine_flow_existing_compressors

log = logging.getLogger(__name__)

def construct_components(model, modelhub):
    """
    Constructs components of the model based on provided parameters.

    Args:
        model: The model object to which components will be added.

    Returns:
        None
    """
    log_msg = "--- Constructing Model ---"
    print(log_msg)
    log.info(log_msg)
    start = time.time()

    aggregation_model = modelhub.info_solving_algorithms["aggregation_model"]
    aggregation_data = modelhub.info_solving_algorithms["aggregation_data"]
    modelhub.model[aggregation_model] = pyo.ConcreteModel()

    # GET DATA
    model = modelhub.model[aggregation_model]
    topology = modelhub.data.topology
    config = modelhub.data.model_config

    # DEFINE GLOBAL SETS
    # Nodes, Carriers, Technologies, Networks
    model.set_periods = pyo.Set(initialize=topology["investment_periods"])
    model.set_nodes = pyo.Set(initialize=topology["nodes"])
    model.set_carriers = pyo.Set(initialize=topology["carriers"])

    # DEFINE GLOBAL VARIABLES
    model.var_npv = pyo.Var()
    model.var_emissions_net = pyo.Var()

    # INVESTMENT PERIOD BLOCK
    def init_period_block(b_period):
        """Pyomo rule to initialize a block holding all investment periods"""

        # Get data for investment period
        investment_period = b_period.index()
        data_period = get_data_for_investment_period(
            modelhub.data, investment_period, aggregation_data
        )
        # Add sets, parameters, variables, constraints to block
        b_period = construct_investment_period_block(b_period, data_period)

        # NETWORK BLOCK
        if not config["energybalance"]["copperplate"]["value"]:
            def init_network_block(b_netw, netw):
                """Pyomo rule to initialize a block holding all networks"""
                # Add sets, parameters, variables, constraints to block
                b_netw = construct_network_block(
                    b_netw,
                    data_period,
                    model.set_nodes,
                    b_period.set_t_full,
                    b_period.set_t_clustered,
                )

                return b_netw

            b_period.network_block = pyo.Block(
                b_period.set_networks, rule=init_network_block
            )

        # NODE BLOCK
        def init_node_block(b_node, node):
            """Pyomo rule to initialize a block holding all nodes"""
            # Get data for node
            data_node = get_data_for_node(data_period, node)

            # Add sets, parameters, variables, constraints to block
            b_node = construct_node_block(
                b_node, data_node, b_period.set_t_full, b_period.set_t_clustered
            )

            # TECHNOLOGY BLOCK
            def init_technology_block(b_tec, tec):
                """Pyomo rule to initialize a block holding all technologies at node"""
                b_tec = construct_technology_block(
                    b_tec, data_node, b_period.set_t_full, b_period.set_t_clustered
                )

                return b_tec

            b_node.tech_blocks_active = pyo.Block(
                b_node.set_technologies, rule=init_technology_block
            )

            # COMPRESSOR BLOCK
            if config["performance"]["pressure"]["pressure_on"]["value"] == 1:
                def init_compressor_block(b_compr, car, comp1, comp2):
                    """Pyomo rule to initialize a block holding all compressors at node"""
                    b_compr = construct_compressor_block(
                        b_compr,
                        data_node,
                        b_period.set_t_full,
                        b_period.set_t_clustered,
                    )
                    return b_compr

                b_node.compressor_blocks_active = pyo.Block(
                    b_node.set_compressor, rule=init_compressor_block
                )
            return b_node

        b_period.node_blocks = pyo.Block(model.set_nodes, rule=init_node_block)

        if config["performance"]["pressure"]["pressure_on"]["value"] == 1:
            # fixing size of existing compressor based on components minimum capacity
            for node in b_period.node_blocks:
                data_node = get_data_for_node(data_period, node)
                for compr in b_period.node_blocks[node].set_compressor:
                    compressor = data_node["compressor_data"][compr]
                    b_compr = b_period.node_blocks[node].compressor_blocks_active[
                        compr
                    ]

                    if (compressor.compression_active == 1) and (
                            compressor.existing == 1
                    ):
                        size = determine_flow_existing_compressors(
                            modelhub, compressor, b_period, node
                        )
                        compressor.fix_size(b_compr, size)

        return b_period

    model.periods = pyo.Block(model.set_periods, rule=init_period_block)

    log_msg = f"Constructing model completed in {str(round(time.time() - start))}s"
    print(log_msg)
    log.info(log_msg)
