from ..components.utilities import perform_disjunct_relaxation


def construct_technology_block(b_tec, data: dict, set_t_full, set_t_clustered):
    """
    Construct technology block and performs disjunct relaxation if required

    :param b_tec: pyomo block with technology model
    :param dict data: data containing model configuration
    :param set_t_full: pyomo set containing timesteps
    :param set_t_clustered: pyomo set containing clustered timesteps
    :return: pyomo block with technology model
    """
    tec = b_tec.index()
    technology = data["technology_data"][tec]
    b_tec = technology.construct_tech_model(b_tec, data, set_t_full, set_t_clustered)
    if technology.big_m_transformation_required:
        b_tec = perform_disjunct_relaxation(b_tec)

    return b_tec
