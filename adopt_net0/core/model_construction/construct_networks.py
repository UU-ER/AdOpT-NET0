from adopt_net0.core.components.utilities import perform_disjunct_relaxation


def construct_network_block(
        b_netw, modelhub, period, set_nodes, set_t_full, set_t_clustered
        ):
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

    network_constructor = modelhub.component_constructors["network_constructors"][period][netw]
    network_constructor.construct_model(
        b_netw, modelhub, set_t_full, set_t_clustered,
        set_nodes=set_nodes
    )
    if network_constructor.big_m_transformation_required:
        perform_disjunct_relaxation(b_netw)
