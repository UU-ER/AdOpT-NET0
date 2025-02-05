import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

from ..utilities import fit_piecewise_function
from ..technology import Technology

import logging

log = logging.getLogger(__name__)


class CementHybridCCS(Technology):
    """
    Cement plant with hybrid CCS

    The plant had an oxyfuel combustion in the calciner and post-combustion capture with MEA afterward. The size
    of the oxyfuel is fixed, while the size and capture rate of the MEA are variables of the optimization
    """

    def __init__(self, tec_data: dict):
        """
        Constructor

        :param dict tec_data: technology data
        """
        super().__init__(tec_data)
        self.component_options.emissions_based_on = "output"
        self.component_options.size_based_on = "output"
        self.component_options.size_cement = "size_cement"
        self.component_options.main_input_carrier = tec_data["Performance"][
            "main_input_carrier"
        ]

    def fit_technology_performance(self, climate_data: pd.DataFrame, location: dict):
        """
        Fits the technology performance

        :param pd.Dataframe climate_data: dataframe containing climate data
        :param dict location: dict containing location details
        """
        super(CementHybridCCS, self).fit_technology_performance(climate_data, location)

        # Number of segments
        nr_segments = self.input_parameters.performance_data["nr_segments"]

        # Read performance data from file
        performance_data_path = Path(__file__).parent.parent.parent.parent
        performance_data_path = (
            performance_data_path
            / "data/technology_data/Industrial/CementHybridCCS_data/dac_adsorption_performance.txt"
        )

        performance_data = pd.read_csv(performance_data_path, sep=",")
        performance_data = performance_data.rename(
            columns={"T": "temp_air", "RH": "humidity"}
        )

        # Unit Conversion of input data
        performance_data.E_tot = performance_data.E_tot.multiply(
            performance_data.CO2_Out / 3600
        )  # in MWh / h
        performance_data.E_el = performance_data.E_el.multiply(
            performance_data.CO2_Out / 3600
        )  # in MWh / h
        performance_data.E_th = performance_data.E_th.multiply(
            performance_data.CO2_Out / 3600
        )  # in MWh / h
        performance_data.CO2_Out = performance_data.CO2_Out / 1000  # in t / h

        # Get humidity and temperature
        RH = copy.deepcopy(climate_data["rh"])
        T = copy.deepcopy(climate_data["temp_air"])

        # Set minimum temperature
        T.loc[T < min(performance_data.temp_air)] = min(performance_data.temp_air)

        # Derive performance points for each timestep
        def interpolate_performance_point(t, rh, point_data, var):
            zi = griddata(
                (point_data.temp_air, point_data.humidity),
                point_data[var],
                (T, RH),
                method="linear",
            )
            return zi

        CO2_Out = np.empty(shape=(len(T), len(performance_data.Point.unique())))
        E_tot = np.empty(shape=(len(T), len(performance_data.Point.unique())))
        E_el = np.empty(shape=(len(T), len(performance_data.Point.unique())))
        for point in performance_data.Point.unique():
            CO2_Out[:, point - 1] = interpolate_performance_point(
                T, RH, performance_data.loc[performance_data.Point == point], "CO2_Out"
            )
            E_tot[:, point - 1] = interpolate_performance_point(
                T, RH, performance_data.loc[performance_data.Point == point], "E_tot"
            )
            E_el[:, point - 1] = interpolate_performance_point(
                T, RH, performance_data.loc[performance_data.Point == point], "E_el"
            )

        # Derive piecewise definition
        alpha = np.empty(shape=(len(T), nr_segments))
        beta = np.empty(shape=(len(T), nr_segments))
        b = np.empty(shape=(len(T), nr_segments + 1))
        gamma = np.empty(shape=(len(T), nr_segments))
        delta = np.empty(shape=(len(T), nr_segments))
        a = np.empty(shape=(len(T), nr_segments + 1))
        el_in_max = np.empty(shape=(len(T)))
        th_in_max = np.empty(shape=(len(T)))
        out_max = np.empty(shape=(len(T)))
        total_in_max = np.empty(shape=(len(T)))

        log.info("Deriving performance data for DAC...")

        for timestep in range(len(T)):
            if timestep % 100 == 1:
                print("\rComplete: ", round(timestep / len(T), 2) * 100, "%", end="")
            # Input-Output relation
            y = {}
            y["CO2_Out"] = CO2_Out[timestep, :]
            time_step_fit = fit_piecewise_function(
                E_tot[timestep, :], y, int(nr_segments)
            )
            alpha[timestep, :] = time_step_fit["CO2_Out"]["alpha1"]
            beta[timestep, :] = time_step_fit["CO2_Out"]["alpha2"]
            b[timestep, :] = time_step_fit["CO2_Out"]["bp_x"]
            out_max[timestep] = max(time_step_fit["CO2_Out"]["bp_y"])
            total_in_max[timestep] = max(time_step_fit["CO2_Out"]["bp_x"])

            # Input-Input relation
            y = {}
            y["E_el"] = E_el[timestep, :]
            time_step_fit = fit_piecewise_function(
                E_tot[timestep, :], y, int(nr_segments)
            )
            gamma[timestep, :] = time_step_fit["E_el"]["alpha1"]
            delta[timestep, :] = time_step_fit["E_el"]["alpha2"]
            a[timestep, :] = time_step_fit["E_el"]["bp_x"]
            el_in_max[timestep] = max(time_step_fit["E_el"]["bp_y"])
            th_in_max[timestep] = max(time_step_fit["E_el"]["bp_x"])

        print("Complete: ", 100, "%")

        # Coefficients
        self.processed_coeff.time_dependent_full["alpha"] = alpha
        self.processed_coeff.time_dependent_full["beta"] = beta
        self.processed_coeff.time_dependent_full["b"] = b
        self.processed_coeff.time_dependent_full["gamma"] = gamma
        self.processed_coeff.time_dependent_full["delta"] = delta
        self.processed_coeff.time_dependent_full["a"] = a
        self.processed_coeff.time_dependent_full["out_max"] = out_max
        self.processed_coeff.time_dependent_full["el_in_max"] = el_in_max
        self.processed_coeff.time_dependent_full["th_in_max"] = th_in_max
        self.processed_coeff.time_dependent_full["total_in_max"] = total_in_max

        self.processed_coeff.time_independent["eta_elth"] = (
            self.input_parameters.performance_data["performance"]["eta_elth"]
        )

        # Options
        self.component_options.other["nr_segments"] = (
            self.input_parameters.performance_data["nr_segments"]
        )
        self.component_options.other["ohmic_heating"] = (
            self.input_parameters.performance_data["ohmic_heating"]
        )
