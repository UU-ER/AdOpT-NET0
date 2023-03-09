from src.data_management.components.utilities import Economics
from src.data_management.components.fit_technology_performance import *
import json

class Technology:
    """
    Class to read and manage data for technologies
    """
    def __init__(self, technology):
        """
        Initializes technology class from technology name

        The technology name needs to correspond to the name of a JSON file in ./data/technology_data.

        :param str technology: name of technology to read data
        """
        tec_data = read_technology_data_from_json(technology)

        # General information
        self.name = technology
        self.existing = 0
        self.technology_model = tec_data['tec_type']
        self.size_initial = []
        self.size_is_int = tec_data['size_is_int']
        self.size_min = tec_data['size_min']
        self.size_max = tec_data['size_max']
        self.decommission = tec_data['decommission']

        # Economics
        self.economics = Economics(tec_data['Economics'])

        # Technology Performance
        self.performance_data = tec_data['TechnologyPerf']
        self.fitted_performance = []

    def fit_technology_performance(self, climate_data):
        """
        Fits performance to respective technology model

        :param pd climate_data: Dataframe of climate data
        """
        # Initialize parameters dict
        parameters = {}

        # Derive performance parameters for respective performance function type
        # GENERIC TECHNOLOGIES
        if self.technology_model == 'RES':  # Renewable technologies
            if self.name == 'PV':
                if 'system_type' in self.performance_data:
                    self.fitted_performance = perform_fitting_PV(climate_data,
                                                           system_data=self.performance_data['system_type'])
                else:
                    self.fitted_performance = perform_fitting_PV(climate_data)
            elif self.name == 'ST':
                self.fitted_performance = perform_fitting_ST(climate_data)
            elif 'WT' in self.name:
                if 'hubheight' in self.performance_data:
                    hubheight = self.performance_data['hubheight']
                else:
                    hubheight = 120
                self.fitted_performance = perform_fitting_WT(climate_data, self.name, hubheight)

        elif self.technology_model == 'CONV1':  # n inputs -> n output, fuel and output substitution
            self.fitted_performance = perform_fitting_tec_CONV1(self.performance_data)

        elif self.technology_model == 'CONV2':  # n inputs -> n output, fuel and output substitution
            self.fitted_performance = perform_fitting_tec_CONV2(self.performance_data)

        elif self.technology_model == 'CONV3':  # n inputs -> n output, fixed ratio between inputs and outputs
            self.fitted_performance = perform_fitting_tec_CONV3(self.performance_data)

        elif self.technology_model == 'STOR':  # storage technologies
            self.fitted_performance = perform_fitting_tec_STOR(self.performance_data, climate_data)

        # SPECIFIC TECHNOLOGIES
        elif self.technology_model == 'DAC_adsorption':  # DAC adsorption
            self.fitted_performance = perform_fitting_tec_DAC_adsorption(self.performance_data, climate_data)


def read_technology_data_from_json(tec):
    """
    Reads technology data from json file
    """
    # Read in JSON files
    with open('./data/technology_data/' + tec + '.json') as json_file:
        technology_data = json.load(json_file)
    # Assign name
    technology_data['Name'] = tec
    return technology_data

