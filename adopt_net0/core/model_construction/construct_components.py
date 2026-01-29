import logging
import time

import pyomo.environ as pyo

from adopt_net0.core.model_construction.construct_investment_period import construct_investment_period_block
from adopt_net0.core.model_construction.construct_nodes import construct_node_block
from adopt_net0.core.model_construction.construct_networks import construct_network_block
from adopt_net0.core.model_construction.construct_technology import construct_technology_block
from adopt_net0.plugins.hooks import Hook

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
    # aggregation_data = modelhub.info_solving_algorithms["aggregation_data"]
    modelhub.model[aggregation_model] = pyo.ConcreteModel()

    # GET DATA
    model = modelhub.model[aggregation_model]
    topology = modelhub.data["topology"]
    config = modelhub.data["config"]

    # DEFINE GLOBAL SETS
    # Nodes, Carriers, Technologies, Networks
    model.set_periods = pyo.Set(initialize=topology["investment_periods"])
    model.set_nodes = pyo.Set(initialize=list(topology["nodes"].keys()))
    model.set_carriers = pyo.Set(initialize=topology["carriers"])

    # DEFINE GLOBAL VARIABLES
    model.var_npv = pyo.Var()
    model.var_emissions_net = pyo.Var()

    # INVESTMENT PERIOD BLOCK
    def init_period_block(b_period):
        """Pyomo rule to initialize a block holding all investment periods"""
        # Add sets, parameters, variables, constraints to block
        construct_investment_period_block(b_period, modelhub)
        period = b_period.index()
        set_t_full = b_period.set_t_full
        set_t_clustered = b_period.set_t_clustered

        # NETWORK BLOCK
        if not config["energybalance"]["copperplate"]["value"]:
            def init_network_block(b_netw, netw):
                """Pyomo rule to initialize a block holding all networks"""
                # Add sets, parameters, variables, constraints to block
                construct_network_block(
                    b_netw,
                    modelhub,
                    period,
                    model.set_nodes,
                    set_t_full,
                    set_t_clustered,
                )

                return b_netw

            b_period.network_block = pyo.Block(
                b_period.set_networks, rule=init_network_block
            )

        # NODE BLOCK
        def init_node_block(b_node, node):
            """Pyomo rule to initialize a block holding all nodes"""
            # Add sets, parameters, variables, constraints to block
            b_node = construct_node_block(
                b_node, modelhub, period, set_t_full, set_t_clustered
            )

            # TECHNOLOGY BLOCK
            def init_technology_block(b_tec, tec):
                """Pyomo rule to initialize a block holding all technologies at node"""
                b_tec = construct_technology_block(
                    b_tec, modelhub, period, node, set_t_full, set_t_clustered
                )

            b_node.tech_blocks_active = pyo.Block(
                b_node.set_technologies, rule=init_technology_block
            )

            modelhub.plugin_manager.emit(Hook.NODE_CONSTRUCTION_END,
                                         modelhub=modelhub,
                                         b_node=b_node,
                                         b_period=b_period)

            return b_node

        b_period.node_blocks = pyo.Block(model.set_nodes, rule=init_node_block)


        return b_period

    model.periods = pyo.Block(model.set_periods, rule=init_period_block)

    log_msg = f"Constructing model completed in {str(round(time.time() - start))}s"
    print(log_msg)
    log.info(log_msg)
