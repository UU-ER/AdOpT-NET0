from src.data_management.components.utilities import Economics
import json
import pandas as pd

class Network:
    """
    Class to read and manage data for technologies
    """
    def __init__(self, network:dict, path:str):
        """
        Initializes technology class from technology name

        The network name needs to correspond to the name of a JSON file in ./data/network_data.

        :param dict network: name of technology to read data
        :param str path: path to read network data from
        """
        netw_data = read_network_data_from_json(network['name'], path)

        # General information
        self.name = network['name']
        self.existing = network['existing']
        self.connection = network['connection']
        self.distance = network['distance']
        if self.existing:
            self.size_initial = network['size']

        self.size_is_int = netw_data['size_is_int']
        self.size_min = netw_data['size_min']
        self.size_max = netw_data['size_max']
        self.decommission = netw_data['decommission']
        if network['size_max_arcs'] is not None:
            self.size_max_arcs = network['size_max_arcs']
        else:
            self.size_max_arcs = pd.DataFrame(self.size_max, index=self.distance.index, columns=self.distance.columns)

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


def read_network_data_from_json(network, path):
    """
    Reads network data from json file
    """
    # Read in JSON files
    with open(path + network + '.json') as json_file:
        network_data = json.load(json_file)
    # Assign name
    network_data['Name'] = network
    return network_data