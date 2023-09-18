from src.data_management.components.utilities import Economics
import json
import pandas as pd
from pathlib import Path

class Network:
    """
    Class to read and manage data for technologies
    """
    def __init__(self, network):
        """
        Initializes technology class from technology name

        The network name needs to correspond to the name of a JSON file in ./data/network_data.

        :param str network: name of technology to read data
        """
        netw_data = read_network_data_from_json(network)

        # General information
        self.name = network
        self.existing = 0
        self.connection = []
        self.distance = []
        self.size_initial = []
        self.size_is_int = netw_data['size_is_int']
        self.size_min = netw_data['size_min']
        self.size_max = netw_data['size_max']
        self.size_max_arcs = []
        self.decommission = netw_data['decommission']
        self.energy_consumption = {}

        # Economics
        self.economics = Economics(netw_data['Economics'])

        # Technology Performance
        self.performance_data = netw_data['NetworkPerf']
        if self.performance_data['energyconsumption']:
            self.calculate_energy_consumption()

    def calculate_energy_consumption(self):
        """
        Fits the performance parameters for a network, i.e. the consumption at each node.
        :param obj network: Dict read from json files with performance data and options for performance fits
        :param obj climate_data: Climate data
        :return: dict of performance coefficients used in the model
        """
        # Get energy consumption at nodes form file
        energycons = self.performance_data['energyconsumption']
        self.performance_data.pop('energyconsumption')

        for car in energycons:
            self.energy_consumption[car] = {}
            if energycons[car]['cons_model'] == 1:
                self.energy_consumption[car]['send'] = {}
                self.energy_consumption[car]['send'] = energycons[car]
                self.energy_consumption[car]['send'].pop('cons_model')
                self.energy_consumption[car]['receive'] = {}
                self.energy_consumption[car]['receive']['k_flow'] = 0
                self.energy_consumption[car]['receive']['k_flowDistance'] = 0
            elif energycons[car]['cons_model'] == 2:
                temp = energycons[car]
                self.energy_consumption[car]['send'] = {}
                self.energy_consumption[car]['send']['k_flow'] = round(temp['c'] * temp['T'] / temp['eta'] / \
                                                                       temp['LHV'] * ((temp['p'] / 30) **
                                                                                   ((temp['gam'] - 1) / temp[
                                                                                       'gam']) - 1), 4)
                self.energy_consumption[car]['send']['k_flowDistance'] = 0
                self.energy_consumption[car]['receive'] = {}
                self.energy_consumption[car]['receive']['k_flow'] = 0
                self.energy_consumption[car]['receive']['k_flowDistance'] = 0

        self.energy_consumption = self.energy_consumption

    def calculate_max_size_arc(self):
        if self.existing == 0:
            if self.size_max_arcs == None:
                # Use max size
                self.size_max_arcs = pd.DataFrame(self.size_max, index=self.distance.index, columns=self.distance.columns)
        elif self.existing == 1:
            # Use initial size
            self.size_max_arcs = self.size_initial





def read_network_data_from_json(network):
    """
    Reads network data from json file
    """
    # Read in JSON files
    path = Path('./data/network_data/')
    network = network + '.json'

    with open(path / network) as json_file:
        network_data = json.load(json_file)
    # Assign name
    network_data['Name'] = network
    return network_data