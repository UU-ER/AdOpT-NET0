import logging
from pathlib import Path
import json

from ..data_management.handle_input_data import DataHandle
from ..data_management.utilities import open_json
from .all_technologies import *
from ..data_management.utilities import select_technology

log = logging.getLogger(__name__)

def read_tec_data(tec_name: str, load_path: Path):
    """
    Loads the technology data from load_path and preprocesses it.

    :param str tec_name: technology name
    :param Path load_path: load path
    :param pd.DataFrame climate_data: Climate Data
    :param dict location: Dictonary with node location
    :return: Technology Class
    """
    tec_data = open_json(tec_name, load_path)
    tec_data["name"] = tec_name
    if 'capex_optimization' in tec_data:
        select_tec_function = select_technology_capex_optimization
    else:
        select_tec_function = select_technology

    tec_data = select_tec_function(tec_data)

    # CCS
    if tec_data.component_options.ccs_possible:
        tec_data.ccs_data = open_json(tec_data.component_options.ccs_type, load_path)
    return tec_data

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
    # elif tec_data['tec_type'] == 'OceanBattery3':
    #     return OceanBattery3(tec_data)
    # elif tec_data['tec_type'] == 'OceanBattery':
    #     return OceanBattery(tec_data)


class DataHandleCapexOptimization(DataHandle):

    def __init__(self, technology_to_optimize):
        """
        Constructor
        """
        super().__init__()
        self.technology_to_optimize = technology_to_optimize


    def _read_technology_data(self):
        """
        Reads all technology data and fits it

        :param str aggregation_model: specifies the aggregation type and thus the dict key to write the data to
        """
        # Technology data always fitted based on full resolution
        aggregation_model = "full"

        # Initialize technology_data dict
        technology_data = {}

        # Loop through all investment_periods and nodes
        for investment_period in self.topology["investment_periods"]:
            technology_data[investment_period] = {}
            for node in self.topology["nodes"]:
                technology_data[investment_period][node] = {}

                # Get technologies at node
                with open(
                        self.data_path
                        / investment_period
                        / "node_data"
                        / node
                        / "Technologies.json"
                ) as json_file:
                    technologies_at_node = json.load(json_file)

                if self.technology_to_optimize[0] == node:
                    technologies_at_node["new"].append(self.technology_to_optimize[1])

                # New technologies
                for technology in technologies_at_node["new"]:
                    tec_data = read_tec_data(
                        technology,
                        self.data_path
                        / investment_period
                        / "node_data"
                        / node
                        / "technology_data",
                    )
                    tec_data.fit_technology_performance(
                        self.time_series[aggregation_model][investment_period][node][
                            "ClimateData"
                        ]["global"],
                        self.node_locations.loc[node, :],
                    )
                    technology_data[investment_period][node][technology] = tec_data

                # Existing technologies
                for technology in technologies_at_node["existing"]:
                    tec_data = read_tec_data(
                        technology,
                        self.data_path
                        / investment_period
                        / "node_data"
                        / node
                        / "technology_data",
                    )
                    tec_data.existing = 1
                    tec_data.input_parameters.size_initial = technologies_at_node[
                        "existing"
                    ][technology]
                    tec_data.fit_technology_performance(
                        self.time_series[aggregation_model][investment_period][node][
                            "ClimateData"
                        ]["global"],
                        self.node_locations.loc[node, :],
                    )
                    technology_data[investment_period][node][
                        technology + "_existing"
                        ] = tec_data

        self.technology_data = technology_data

        # Log success
        log_msg = "Technology data read successfully"
        log.info(log_msg)


class DataHandleEmissionOptimization(DataHandle):

    def __init__(self, technology_to_optimize):
        """
        Constructor
        """
        super().__init__()
        self.technology_to_optimize = technology_to_optimize

    def _read_technology_data(self):
        """
        Reads all technology data and fits it

        :param str aggregation_model: specifies the aggregation type and thus the dict key to write the data to
        """
        # Technology data always fitted based on full resolution
        aggregation_model = "full"

        # Initialize technology_data dict
        technology_data = {}

        # Loop through all investment_periods and nodes
        for investment_period in self.topology["investment_periods"]:
            technology_data[investment_period] = {}
            for node in self.topology["nodes"]:
                technology_data[investment_period][node] = {}

                # Get technologies at node
                with open(
                        self.data_path
                        / investment_period
                        / "node_data"
                        / node
                        / "Technologies.json"
                ) as json_file:
                    technologies_at_node = json.load(json_file)

                if self.technology_to_optimize[0] == node:
                    technologies_at_node["new"].append(self.technology_to_optimize[1])

                # New technologies
                for technology in technologies_at_node["new"]:
                    tec_data = read_tec_data(
                        technology,
                        self.data_path
                        / investment_period
                        / "node_data"
                        / node
                        / "technology_data",
                    )
                    tec_data.fit_technology_performance(
                        self.time_series[aggregation_model][investment_period][node][
                            "ClimateData"
                        ]["global"],
                        self.node_locations.loc[node, :],
                    )
                    technology_data[investment_period][node][technology] = tec_data

                # Existing technologies
                for technology in technologies_at_node["existing"]:
                    tec_data = read_tec_data(
                        technology,
                        self.data_path
                        / investment_period
                        / "node_data"
                        / node
                        / "technology_data",
                    )
                    tec_data.existing = 1
                    tec_data.input_parameters.size_initial = technologies_at_node[
                        "existing"
                    ][technology]
                    tec_data.fit_technology_performance(
                        self.time_series[aggregation_model][investment_period][node][
                            "ClimateData"
                        ]["global"],
                        self.node_locations.loc[node, :],
                    )
                    technology_data[investment_period][node][
                        technology + "_existing"
                        ] = tec_data

        self.technology_data = technology_data

        # Log success
        log_msg = "Technology data read successfully"
        log.info(log_msg)