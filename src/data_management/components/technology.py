import src.global_variables as global_variables
from src.data_management.components.utilities import Economics
from src.data_management.components.fit_technology_performance import *

class Technology:
    """
    Class to read and manage data for technologies
    """
    def __init__(self, technology, load_path):
        """
        Initializes technology class from technology name

        The technology name needs to correspond to the name of a JSON file in ./data/technology_data.

        :param str technology: name of technology to read data
        """
        tec_data = open_json(technology, load_path)

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

        # Size-input/output constraints
        if self.technology_model == 'CONV1':
            self.performance_data['size_based_on'] = tec_data['size_based_on']
        else:
            self.performance_data['size_based_on'] = 'input'

        # Emissions are based on...
        if (self.technology_model == 'DAC_Adsorption') or \
                (self.technology_model == 'CONV4'):
            self.emissions_based_on = 'output'
        else:
            self.emissions_based_on = 'input'

        self.fitted_performance = None

    def fit_technology_performance(self, node_data):

        """
        Fits performance to respective technology model

        :param pd node_data: Dataframe of climate data
        """
        location = node_data.location
        climate_data = node_data.data['climate_data']

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

        elif self.technology_model == 'CONV2':  # n inputs -> n output, fuel substitution
            self.fitted_performance = perform_fitting_tec_CONV2(self.performance_data, climate_data)

        elif self.technology_model == 'CONV3':  # n inputs -> n output, fixed ratio between inputs and outputs
            self.fitted_performance = perform_fitting_tec_CONV3(self.performance_data, climate_data)

        elif self.technology_model == 'CONV4':  # 0 inputs -> n outputs, fixed ratio between outputs
            self.fitted_performance = perform_fitting_tec_CONV4(self.performance_data, climate_data)

        elif self.technology_model == 'STOR':  # storage technologies
            self.fitted_performance = perform_fitting_tec_STOR(self.performance_data, climate_data)

        # SPECIFIC TECHNOLOGIES
        elif self.technology_model == 'DAC_Adsorption':  # DAC adsorption
            self.fitted_performance = perform_fitting_tec_DAC_adsorption(self.performance_data, climate_data)

        elif self.technology_model.startswith('HeatPump_'):  # Heat Pump
            self.fitted_performance = perform_fitting_tec_HP(self.performance_data, climate_data, self.technology_model)

        elif self.technology_model.startswith('GasTurbine_'):  # Gas Turbine
            self.fitted_performance = perform_fitting_tec_GT(self.performance_data, climate_data)

        elif self.technology_model == 'Hydro_Open':  # Open Cycle Pumped Hydro
            self.fitted_performance = perform_fitting_tec_hydro_open(self.name, self.performance_data, climate_data)



