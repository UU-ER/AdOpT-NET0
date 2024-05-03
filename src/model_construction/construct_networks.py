from pyomo.environ import *
from ..components.utilities import perform_disjunct_relaxation


def construct_network_block(b_netw, data, set_nodes, set_t_full, set_t_clustered):
    netw = b_netw.index()
    network = data["network_data"][netw]
    b_netw = network.construct_netw_model(
        b_netw, data, set_nodes, set_t_full, set_t_clustered
    )
    if network.big_m_transformation_required:
        b_netw = perform_disjunct_relaxation(b_netw)

    return b_netw
