from .res import Res
import pandas as pd
import numpy as np

class Res_Cap(Res):
    def __init__(self,
                tec_data):

        self.climate_year = None
        self.node = None

        super().__init__(tec_data)

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):

        time_independent = {}

        # Size
        time_independent["size_min"] = self.size_min
        if not self.existing:
            time_independent["size_max"] = self.size_max
        else:
            time_independent["size_max"] = self.size_initial
            time_independent["size_initial"] = self.size_initial

        # Emissions
        time_independent["emission_factor"] = self.performance_data["emission_factor"]

        # Other
        time_independent["rated_capacity"] = 1
        time_independent["min_part_load"] = 0
        time_independent["standby_power"] = -1

        # Dynamics
        dynamics = {}
        dynamics_parameter = [
            "ramping_time",
            "ref_size",
            "ramping_const_int",
            "standby_power",
            "min_uptime",
            "min_downtime",
            "SU_time",
            "SD_time",
            "SU_load",
            "SD_load",
            "max_startups",
        ]
        for p in dynamics_parameter:
            if p in self.performance_data:
                dynamics[p] = self.performance_data[p]

        # Write to self
        self.processed_coeff.time_independent = time_independent
        self.processed_coeff.dynamics = dynamics

        # read in data
        if self.name == 'Offshore_Wind':
            capacity_factor = climate_data["offshore_wind"].values
        elif self.name == 'Onshore_Wind':
            capacity_factor = climate_data["onshore_wind"].values
        elif self.name == 'PV':
            capacity_factor = climate_data["pv"].values

        self.processed_coeff.time_dependent_full["capfactor"] = capacity_factor

