from ..components.utilities import perform_disjunct_relaxation


def construct_network_block(b_netw, data: dict, set_nodes, set_t_full, set_t_clustered):
    """
    Construct network block and performs disjunct relaxation if required

    :param b_netw: pyomo block with network model
    :param dict data: data containing model configuration
    :param set_nodes: pyomo set containing all nodes
    :param set_t_full: pyomo set containing timesteps
    :param set_t_clustered: pyomo set containing clustered timesteps
    :return: pyomo block with network model
    """
    netw = b_netw.index()
    network = data["network_data"][netw]
    b_netw = network.construct_netw_model(
        b_netw, data, set_nodes, set_t_full, set_t_clustered
    )
    if network.big_m_transformation_required:
        b_netw = perform_disjunct_relaxation(b_netw)

    return b_netw
