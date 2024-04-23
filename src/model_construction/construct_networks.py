from pyomo.environ import *
from ..components.utilities import perform_disjunct_relaxation


def construct_network_block(b_netw, netw, data):

    # PREPROCESSING
    network = data["network_data"]["full"][netw]

    # CONSTRUCT NETWORK MODEL
    b_netw = network.construct_general_constraints(b_netw, data)
    if network.big_m_transformation_required:
        b_netw = perform_disjunct_relaxation(b_netw)

    return b_netw
