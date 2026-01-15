"""
Plugin to calculate technology performance time series based on climate data.

Each technology possible is implemented in a separate function in performance_from_climate_data_helpers.
"""

import pandas as pd
from pathlib import Path

from adopt_net0.core.data_management.read_input_data import read_topology, read_technology_data
from adopt_net0.plugins.preprocessing_plugins.performance_from_climate_data_helpers import (
    calculate_performance_pv,
    calculate_performance_wt,
    calculate_performance_stor
)
from adopt_net0.plugins.base import Plugin as PluginBase
from adopt_net0.plugins.hooks import Hook


def _load_climate_data(node_data_path: Path) -> pd.DataFrame:
    """
    Read climate data from CSV file.

    :param node_data_path: datapath to node directory
    :return: Climate data as DataFrame
    :rtype: pd.DataFrame
    """
    try:
        return pd.read_csv(node_data_path / "ClimateData.csv", sep=";", index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"ClimateData.csv not found in {node_data_path}")


def _calculate_performance_for_technology(data_path: Path, component_id: tuple) -> pd.DataFrame:
    """
    Calculate technology performance time series based on climate data.

    :param data_path Path: Path to input data directory
    :param component_id tuple: (period, node, technology)
    :return: Data frame containing all required climate based time series for technology
    :rtype: pd.DataFrame
    """
    period = component_id[0]
    node = component_id[1]
    technology = component_id[2]
    topology = read_topology(data_path)
    technology_data = read_technology_data(data_path, topology)

    # load climate data
    climate_data = _load_climate_data(data_path / period / "node_data" / node)
    climate_data.index = pd.to_datetime(climate_data.index)
    node_location = topology["nodes"][node]

    if technology_data[period][node][technology]["tec_type"] == "RES":
        if "Photovoltaic" in technology:
            if "system_type" in technology_data[period][node][technology]:
                return calculate_performance_pv(
                    climate_data,
                    node_location,
                    system_data=technology_data[period][node][technology]["system_type"],
                )
            else:
                return calculate_performance_pv(climate_data, node_location)

        elif "WindTurbine" in technology:
            if "hubheight" in technology_data[period][node][technology]:
                hubheight = technology_data[period][node][technology]["hubheight"]
            else:
                hubheight = 120
            return calculate_performance_wt(climate_data, hubheight, technology)
    elif technology_data[period][node][technology]["tec_type"] == "STOR":
        return calculate_performance_stor(climate_data, technology_data[period][node][technology])

    else:
        raise RuntimeError(f"Unknown technology {technology}")


class Plugin(PluginBase):

    name = "Technology performance from climate data"
    hooks = {
        Hook.DATA_READ_START,
    }
    config_template = {
      "technologies": []
    }

    def on_data_read_start(self, data_path: Path = None) -> None:
        """
        Calculates technology performance time series based on climate data for all technologies specified in plugin config.

        Loops through all investment periods, nodes and technologies in the input data directory. If a technology is listed in the plugin configuration,
        the corresponding performance time series is calculated using climate data and saved as CSV file in the respective directory.

        :param data_path Path: Path to input data directory
        """
        topology = read_topology(data_path)
        technology_data = read_technology_data(data_path, topology)
        for period in technology_data:
            for node in technology_data[period]:
                for technology in technology_data[period][node]:
                    if technology.replace("_existing", "") in self.config["technologies"]:
                        technology_time_series = _calculate_performance_for_technology(data_path, (period, node, technology))
                        save_path =  data_path / period / "node_data" / node / "technology_time_series" / (technology + ".csv")
                        technology_time_series.to_csv(save_path, sep=";", index=True)
