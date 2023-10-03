from pyomo.environ import *
from ..components.component import perform_disjunct_relaxation

def add_networks(energyhub):
    r"""
    Adds all networks as model blocks.

    :param EnergyHub energyhub: instance of the energyhub
    :return: model
    """
    # COLLECT OBJECTS FROM ENERGYHUB
    model = energyhub.model
    print('_' * 60)
    print('--- Adding Networks... ---')

    def init_network(b_netw, netw):
        """
        Rule to construct network

        :param b_netw: network block
        :param netw: network name
        :return: b_netw: network block
        """
        print('\t - Adding Network ' + netw)

        network = energyhub.data.network_data[netw]
        b_netw = network.construct_general_constraints(b_netw, energyhub)
        if network.big_m_transformation_required:
            b_netw = perform_disjunct_relaxation(b_netw)

        return b_netw

    model.network_block = Block(model.set_networks, rule=init_network)
    return model
