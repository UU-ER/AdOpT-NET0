import pandas as pd

from ..component import ModelComponent


class Network(ModelComponent):
    """
    Class to read and manage data for networks
    """
    def __init__(self, netw_data):
        """
        Initializes technology class from technology name

        The network name needs to correspond to the name of a JSON file in ./data/network_data.

        :param str network: name of technology to read data
        """
        super().__init__(netw_data)

        # General information
        self.connection = []
        self.distance = []
        self.size_max_arcs = []
        self.energy_consumption = {}

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
