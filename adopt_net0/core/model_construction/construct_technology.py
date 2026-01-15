from adopt_net0.core.components.utilities import perform_disjunct_relaxation


def construct_technology_block(b_tec, modelhub, period, node, set_t_full, set_t_clustered):
    """
    Construct technology block and performs disjunct relaxation if required

    :param b_tec: pyomo block with technology model
    :param dict data: data containing model configuration
    :param set_t_full: pyomo set containing timesteps
    :param set_t_clustered: pyomo set containing clustered timesteps
    :return: pyomo block with technology model
    """
    # Collect data for node and period
    tec = b_tec.index()
    technology_constructor = modelhub.component_constructors["technology_constructors"][period][node][tec]
    technology_constructor.construct_model(b_tec, modelhub, set_t_full, set_t_clustered)
    if technology_constructor.big_m_transformation_required:
        perform_disjunct_relaxation(b_tec)


