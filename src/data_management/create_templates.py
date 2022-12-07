import numpy as np
import pandas as pd


def create_empty_topology():
    """
    Creates an empty topology file for easy data entry.

    :return: empty topology file as a dict
    """
    topology = {}
    topology['timesteps'] = []
    topology['timestep_length_h'] = []
    topology['carriers'] = []
    topology['nodes'] = []
    topology['technologies'] = {}
    topology['networks'] = {}
    return topology


def create_empty_network_data(nodes):
    """
    Function creates connection and distance matrix for defined nodes.

    :param list nodes: list of nodes to create matrices from
    :return: dictionary containing two pandas data frames with a distance and connection matrix respectively
    """
    # initialize return data dict
    data = {}

    # construct connection matrix
    matrix = pd.DataFrame(data=np.full((len(nodes), len(nodes)), 0),
                          index=nodes, columns=nodes)
    data['connection'] = matrix

    # construct distance matrix
    matrix = pd.DataFrame(data=np.full((len(nodes), len(nodes)), 0),
                          index=nodes, columns=nodes)
    data['distance'] = matrix
    return data
