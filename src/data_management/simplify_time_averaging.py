import src.data_management as dm

import numpy as np
import pandas as pd

class DataHandle_AveragedData(dm.DataHandle):
    """
    DataHandle sub-class for handling averaged data

    This class is used to generate time series of averaged data based on a full resolution
     or clustered input data.
    """
    def __init__(self, data):
        """
        Constructor
        """
        self.topology = data.topology
        self.node_data = {}
        self.node_data_full_resolution = data.node_data
        self.technology_data = data.technology_data
        self.network_data = data.network_data
        self.specifications_time_resolution = {}

    def average_data(self, nr_timesteps_averaged=4):
        RES_tecs = dm.get_RES_technologies(self.topology['technologies'])
        for nodename in self.node_data_full_resolution:
            self.node_data[nodename] = {}
            for series in self.node_data_full_resolution[nodename]:
                self.node_data[nodename][series] = pd.DataFrame()
                if not series == 'climate_data':
                    for carrier in self.node_data_full_resolution[nodename][series]:
                        series_data = dm.reshape_df(self.node_data_full_resolution[nodename][series][carrier],
                                               None, nr_timesteps_averaged)
                        self.node_data[nodename][series][carrier] = series_data.mean(axis=1)
            for tec in RES_tecs[nodename]:
                series_data = dm.reshape_df(self.technology_data[nodename][tec]['fit']['capacity_factor'],
                                       None, nr_timesteps_averaged)
                self.technology_data[nodename][tec]['fit']['capacity_factor'] = series_data.mean(axis=1)

        keys = pd.DataFrame(index=self.topology['timesteps'])
        keys['hourly_order'] = np.repeat(range(1,int(len(keys)/nr_timesteps_averaged)+1), nr_timesteps_averaged)
        self.specifications_time_resolution['keys'] = keys
        self.specifications_time_resolution['factors'] = dm.get_day_factors(keys)


