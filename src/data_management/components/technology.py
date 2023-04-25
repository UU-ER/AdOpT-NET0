import src.global_variables as global_variables
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

        self.modelled_with_full_res = None

        # Economics
        self.economics = Economics(tec_data['Economics'])

        # Technology Performance
        self.performance_data = tec_data['TechnologyPerf']
        self.modelled_with_full_res = 1
        technologies_modelled_with_full_res = ['RES', 'STOR']
        if global_variables.clustered_data and (self.technology_model not in technologies_modelled_with_full_res):
            self.modelled_with_full_res = 0

        self.fitted_performance = None

    def fit_technology_performance(self, node_data):
        """
        Fits performance to respective technology model

        :param pd node_data: Dataframe of climate data
        """
        location = node_data.location

        # Which tecs are modelled with full resolution?
        """
        - clustered vs. not-clustered data
        - averaged vs. not-averaged data
        - RES vs not RES        
        """



        if self.modelled_with_full_res:
            climate_data = node_data.data['climate_data']
        else:
            climate_data = node_data.data_clustered['climate_data']


        # Derive performance parameters for respective performance function type
        # GENERIC TECHNOLOGIES
        if self.technology_model == 'RES':  # Renewable technologies
            if self.name == 'Photovoltaic':
                if 'system_type' in self.performance_data:
                    self.fitted_performance = perform_fitting_PV(climate_data, location,
                                                           system_data=self.performance_data['system_type'])
                else:
                    self.fitted_performance = perform_fitting_PV(climate_data, location)
            elif self.name == 'SolarThermal':
                self.fitted_performance = perform_fitting_ST(climate_data)
            elif 'WindTurbine' in self.name:
                if 'hubheight' in self.performance_data:
                    hubheight = self.performance_data['hubheight']
                else:
                    hubheight = 120
                self.fitted_performance = perform_fitting_WT(climate_data, self.name, hubheight)

        elif self.technology_model == 'CONV1':  # n inputs -> n output, fuel and output substitution
            self.fitted_performance = perform_fitting_tec_CONV1(self.performance_data, climate_data)

        elif self.technology_model == 'CONV2':  # n inputs -> n output, fuel and output substitution
            self.fitted_performance = perform_fitting_tec_CONV2(self.performance_data, climate_data)

        elif self.technology_model == 'CONV3':  # n inputs -> n output, fixed ratio between inputs and outputs
            self.fitted_performance = perform_fitting_tec_CONV3(self.performance_data, climate_data)

        elif self.technology_model == 'STOR':  # storage technologies
            self.fitted_performance = perform_fitting_tec_STOR(self.performance_data, climate_data)

        # SPECIFIC TECHNOLOGIES
        elif self.technology_model == 'DAC_Adsorption':  # DAC adsorption
            self.fitted_performance = perform_fitting_tec_DAC_adsorption(self.performance_data, climate_data)

        elif self.technology_model.startswith('HeatPump_'):  # Heat Pump
            self.fitted_performance = perform_fitting_tec_HP(self.performance_data, climate_data, self.technology_model)

        elif self.technology_model.startswith('GasTurbine_'):  # Gas Turbine
            self.fitted_performance = perform_fitting_tec_GT(self.performance_data, climate_data)


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

