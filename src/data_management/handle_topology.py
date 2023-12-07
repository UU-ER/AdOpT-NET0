import pandas as pd
import numpy as np
import json

class SystemTopology:
    """
    Class to define the energy system topology.

    Class to define the energy system topology including time indices, networks, technologies and nodes as an \
    an input.
    """
    def __init__(self):
        """
        Initializer of SystemTopology Class
        """
        self.timesteps = []
        self.timestep_length_h = []
        self.fraction_of_year_modelled = []
        self.timestep_clustered = []
        self.carriers = []
        self.nodes = []
        self.technologies_new = {}
        self.technologies_existing = {}
        self.networks_new = {}
        self.networks_existing = {}

    def define_time_horizon(self, year:int, start_date:str, end_date:str, resolution:float):
        """
        Defines the time horizon of the topology.

        :param int year: calendar year of analysis
        :param str start_date: start date of analysis in format DD-MM HH-MM
        :param str end_date: end date of analysis in format DD-MM HH-MM
        :param float resolution: resolution in hours
        """
        start_interval = str(year) + '-' + start_date
        end_interval = str(year) + '-' + end_date
        time_resolution = str(resolution) + 'h'
        self.timestep_length_h = resolution
        self.timesteps = pd.date_range(start=start_interval, end=end_interval, freq=time_resolution)
        self.fraction_of_year_modelled = (self.timesteps[-1] - self.timesteps[0]) / pd.Timedelta(days=365)

    def define_carriers(self, carriers:list):
        """
        Defines carriers to use in the analysis

        Can be for example ['electricity', 'heat']

        :param list carriers: List of carriers to use in the analysis
        """
        self.carriers = carriers

    def define_nodes(self, nodes:dict):
        """
        Defines nodes to use in the analysis

        Can be for example {'onshore': {'lon': 51.0, 'lat': 5.2}, 'offshore': {'lon': 51.2, 'lat': 6}}. The node names need to be unique.

        :param dict nodes: Dictionary of nodes to use in the analysis
        """
        self.nodes = nodes
        for node in self.nodes.keys():
            self.technologies_new[node] = {}
            self.technologies_existing[node] = {}

    def define_new_technologies(self, node:str, technologies:list):
        """
        Defines technologies that can be constructed in the analysis

        Can be for example ['Photovoltaic', 'Storage_Battery']. All technologies available can be found
        in ./data/Technology_Data
        These technologies come at a size of zero. Its optimal size is determined in the optimization.
        They are added to the node specified.

        :param str node: Node to add technologies to
        :param list technologies: List of technologies
        """
        if node in self.nodes:
            self.technologies_new[node] = technologies
        else:
            raise KeyError('The node you are trying to add technologies to does not exist.')

    def define_existing_technologies(self, node:str, technologies:dict):
        """
        Defines an existing technologies at a node

        Can be for example {'Photovoltaic': 3, 'WindTurbine_Offshore_6000': 4}. ll technologies available can be found
        in ./data/Technology_Data
        These technologies come at a size of zero. Its optimal size is determined in the optimization.
        They are added to the node specified.

        :param str node: Node to add technologies to
        :param dict technologies: List of technologies
        """
        if node in self.nodes:
            self.technologies_existing[node] = technologies
        else:
            raise KeyError('The node you are trying to add technologies to does not exist.')

    def define_new_network(self, network:str, connections:pd.DataFrame, distance:pd.DataFrame, size_max_arcs = None):
        """
        Defines network that can be constructed in the analysis

        All available networks are in ./data/network_data. Empty connection and distance matrices can be generated
        with  :func:`~src.data_management.handle_topology.create_empty_network_matrix`. The resulting matrices can then
        be changed with distance_matrix.at['onshore', 'offshore'] = 100.

        :param str network: Network name
        :param pd connections: pandas dataframe with connections (0 and 1)
        :param pd distance: distance matrix between nodes (in km)
        """
        self.networks_new[network] = {}
        self.networks_new[network]['name'] = network
        self.networks_new[network]['existing'] = 0
        self.networks_new[network]['connection'] = connections
        self.networks_new[network]['distance'] = distance
        self.networks_new[network]['size_max_arcs'] = size_max_arcs

    def define_existing_network(self, network:str, size:pd.DataFrame, distance:pd.DataFrame):
        """
        Defines an existing network

        All available networks are in ./data/network_data. Empty size and distance matrices can be generated
        with  :func:`~src.data_management.handle_topology.create_empty_network_matrix`. The resulting matrices can then
        be changed with distance_matrix.at['onshore', 'offshore'] = 100.

        :param str network: Network name
        :param pd size: pandas dataframe with size of network
        :param pd distance: distance matrix between nodes (in km)
        """
        self.networks_existing[network] = {}
        self.networks_existing[network]['name'] = network
        self.networks_existing[network]['existing'] = 1
        self.networks_existing[network]['size'] = size
        self.networks_existing[network]['size_max_arcs'] = size
        self.networks_existing[network]['distance'] = distance
        connection = size.copy(deep=True)
        connection[connection > 0] = 1
        self.networks_existing[network]['connection'] = connection
        self.networks_existing[network]['size_max_arcs'] = size

def create_empty_network_matrix(nodes:list):
    """
    Function creates matrix for defined nodes.

    :param list nodes: list of nodes to create matrices from
    :return: pandas data frame with nodes
    """
    # construct matrix
    matrix = pd.DataFrame(data=np.full((len(nodes), len(nodes)), 0),
                          index=nodes, columns=nodes)
    return matrix