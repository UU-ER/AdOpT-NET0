import json

import pandas as pd
import copy
import numpy as np
from pathlib import Path

from ..data_management.handle_input_data import DataHandle
from ..data_management.utilities import open_json
from .all_technologies import *
from ..data_management.utilities import select_technology

def select_technology_capex_optimization(tec_data):
    """
    Returns the correct subclass for a technology

    :param str tec_name: Technology Name
    :param int existing: if technology is existing
    :return: Technology Class
    """
    # Generic tecs
    if tec_data['tec_type'] == 'RES':
        return Res(tec_data)
    elif tec_data['tec_type'] == 'CONV1':
        return Conv1(tec_data)
    elif tec_data['tec_type'] == 'CONV2':
        return Conv2(tec_data)
    elif tec_data['tec_type'] == 'CONV3':
        return Conv3(tec_data)
    elif tec_data['tec_type'] == 'CONV4':
        return Conv4(tec_data)
    elif tec_data['tec_type'] == 'STOR':
        return Stor(tec_data)
    # Specific tecs
    elif tec_data['tec_type'] == 'DAC_Adsorption':
        return DacAdsorption(tec_data)
    elif tec_data['tec_type'].startswith('GasTurbine'):
        return GasTurbine(tec_data)
    elif tec_data['tec_type'].startswith('HeatPump'):
        return HeatPump(tec_data)
    elif tec_data['tec_type'] == 'HydroOpen':
        return HydroOpen(tec_data)
    elif tec_data['tec_type'] == 'OceanBattery3':
        return OceanBattery3(tec_data)
    elif tec_data['tec_type'] == 'OceanBattery':
        return OceanBattery(tec_data)


class DataHandleCapexOptimization(DataHandle):

    def read_technology_data(self, load_path='./data/Technology_Data/'):
        """
        Writes new and existing technologies to self and fits performance functions

        For the default settings, it reads in technology data from JSON files located at ``./data/Technology_Data`` for \
        all technologies specified in the topology. When technology data is stored at a different location, the path \
        should be specified as a string.

        :param str path: path to read technology data from
        :return: self at ``self.Technology_Data[node][tec]``
        """
        load_path = Path(load_path)
        self.model_information.tec_data_path = load_path

        for node in self.topology.nodes:
            self.technology_data[node] = {}
            # New technologies
            for technology in self.topology.technologies_new[node]:
                tec_data = open_json(technology, load_path)
                tec_data['name'] = technology

                if 'capex_optimization' in tec_data:
                    select_tec_function = select_technology_capex_optimization
                else:
                    select_tec_function = select_technology

                self.technology_data[node][technology] = select_tec_function(tec_data)
                self.technology_data[node][technology].fit_technology_performance(self.node_data[node])

            # Existing technologies
            for technology in self.topology.technologies_existing[node].keys():
                tec_data = open_json(technology, load_path)
                tec_data['name'] = technology

                if 'capex_optimization' in tec_data:
                    select_tec_function = select_technology_capex_optimization
                else:
                    select_tec_function = select_technology

                self.technology_data[node][technology + '_existing'] = select_tec_function(tec_data)
                self.technology_data[node][technology + '_existing'].existing = 1
                self.technology_data[node][technology + '_existing'].size_initial = \
                    self.topology.technologies_existing[node][technology]
                self.technology_data[node][technology + '_existing'].fit_technology_performance(self.node_data[node])

    def read_single_technology_data(self, node, technologies):
        """
        Reads technologies to DataHandle after it has been initialized.

        :param str node: node name as specified in the topology
        :param list technologies: technologies to add to node
        :param str path: path to read technology data from
        This function is only required if technologies are added to the model after the DataHandle has been initialized.
        """
        load_path = self.model_information.tec_data_path

        for technology in technologies:
            tec_data = open_json(technology, load_path)
            tec_data['name'] = technology

            self.technology_data[node][technology] = select_technology_capex_optimization(tec_data)
            self.technology_data[node][technology].fit_technology_performance(self.node_data[node])
