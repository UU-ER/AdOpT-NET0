from pyomo.environ import *
from components.utilities import perform_disjunct_relaxation


def add_technology(energyhub, nodename, set_tecsToAdd):
    r"""
    Adds all technologies as model blocks to respective node.

    :param energyhub EnergyHub: instance of the energyhub
    :param str nodename: name of node for which technology is installed
    :param set set_tecsToAdd: list of technologies to add
    :return: b_node
    """
    def init_technology_block(b_tec, tec):
        """
        Rule to construct technology

        :param b_tec: technology block
        :param tec: technology name
        :return: b_tec: technology block
        """

        technology = energyhub.data.technology_data[nodename][tec]
        b_tec = technology.construct_tech_model(b_tec, energyhub)
        if technology.big_m_transformation_required:
            b_tec = perform_disjunct_relaxation(b_tec)

        return b_tec

    # Create a new block containing all new technologies.
    b_node = energyhub.model.node_blocks[nodename]

    if b_node.find_component('tech_blocks_new'):
        b_node.del_component(b_node.tech_blocks_new)
    b_node.tech_blocks_new = Block(set_tecsToAdd, rule=init_technology_block)

    # If it exists, carry over active tech blocks to temporary block
    if b_node.find_component('tech_blocks_active'):
        b_node.tech_blocks_existing = Block(b_node.set_tecsAtNode)
        for tec in b_node.set_tecsAtNode:
            b_node.tech_blocks_existing[tec].transfer_attributes_from(b_node.tech_blocks_active[tec])
        b_node.del_component(b_node.tech_blocks_active)
    if b_node.find_component('tech_blocks_active_index'):
        b_node.del_component(b_node.tech_blocks_active_index)

    # Create a block containing all active technologies at node
    if not set(set_tecsToAdd).issubset(b_node.set_tecsAtNode):
        b_node.set_tecsAtNode.add(set_tecsToAdd)

    def init_active_technology_blocks(bl, tec):
        if tec in set_tecsToAdd:
            bl.transfer_attributes_from(b_node.tech_blocks_new[tec])
        else:
            bl.transfer_attributes_from(b_node.tech_blocks_existing[tec])

    b_node.tech_blocks_active = Block(b_node.set_tecsAtNode, rule=init_active_technology_blocks)

    if b_node.find_component('tech_blocks_new'):
        b_node.del_component(b_node.tech_blocks_new)
    if b_node.find_component('tech_blocks_new_index'):
        b_node.del_component(b_node.tech_blocks_new_index)
    if b_node.find_component('tech_blocks_existing'):
        b_node.del_component(b_node.tech_blocks_existing)
    if b_node.find_component('tech_blocks_existing_index'):
        b_node.del_component(b_node.tech_blocks_existing_index)
    return b_node

