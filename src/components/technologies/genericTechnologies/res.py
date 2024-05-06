import warnings

import pvlib
from timezonefinder import TimezoneFinder
import pandas as pd
from pathlib import Path
from pyomo.environ import *
from scipy.interpolate import interp1d
import numpy as np


from ..technology import Technology
from ..utilities import FittedPerformance


class Res(Technology):
    """
    Resembles a renewable technology with no input. The capacity factors of the technology are determined for each
    individual technology type.

    **Parameter declarations:**

    - Capacity Factor of technology for each time step.

    **Constraint declarations:**

    - Output of technology. The output can be curtailed in three different ways. For ``curtailment == 0``, there is
      no curtailment possible. For ``curtailment == 1``, the curtailment is continuous. For ``curtailment == 2``,
      the size needs to be an integer, and the technology can only be curtailed discretely, i.e. by turning full
      modules off. For ``curtailment == 0`` (default), it holds:

    .. math::
        Output_{t, car} = CapFactor_t * Size
    """

    def __init__(self, tec_data):
        super().__init__(tec_data)

        self.fitted_performance = FittedPerformance(self.performance_data)

    def fit_technology_performance(self, climate_data, location):

        if "Photovoltaic" in self.name:
            if "system_type" in self.performance_data:
                self._perform_fitting_PV(
                    climate_data,
                    location,
                    system_data=self.performance_data["system_type"],
                )
            else:
                self._perform_fitting_PV(climate_data, location)

        elif "SolarThermal" in self.name:
            self._perform_fitting_ST(climate_data)

        elif "WindTurbine" in self.name:
            if "hubheight" in self.performance_data:
                hubheight = self.performance_data["hubheight"]
            else:
                hubheight = 120
            self._perform_fitting_WT(climate_data, hubheight)

    def _perform_fitting_PV(self, climate_data: pd.DataFrame, location: dict, **kwargs):
        """
        Calculates capacity factors and specific area requirements for a PV system using pvlib

        :param climate_data: contains information on weather data
        :param location: contains lon, lat and altitude
        :param PV_type: (optional) can specify a certain type of module
        """
        # Todo: get perfect tilting angle
        if not kwargs.__contains__("system_data"):
            system_data = dict()
            system_data["tilt"] = 18
            system_data["surface_azimuth"] = 180
            system_data["module_name"] = "SunPower_SPR_X20_327"
            system_data["inverter_eff"] = 0.96
        else:
            system_data = kwargs["system_data"]

        def define_pv_system(location, system_data):
            """
            defines the pv system
            :param location: location information (latitude, longitude, altitude, time zone)
            :param system_data: contains data on tilt, surface_azimuth, module_name, inverter efficiency
            :return: returns PV model chain, peak power, specific area requirements
            """
            module_database = pvlib.pvsystem.retrieve_sam("CECMod")
            module = module_database[system_data["module_name"]]

            # Define temperature losses of module
            temperature_model_parameters = (
                pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
                    "open_rack_glass_glass"
                ]
            )

            # Create PV model chain
            inverter_parameters = {
                "pdc0": 5000,
                "eta_inv_nom": system_data["inverter_eff"],
            }
            system = pvlib.pvsystem.PVSystem(
                surface_tilt=system_data["tilt"],
                surface_azimuth=system_data["surface_azimuth"],
                module_parameters=module,
                inverter_parameters=inverter_parameters,
                temperature_model_parameters=temperature_model_parameters,
            )

            pv_model = pvlib.modelchain.ModelChain(
                system, location, spectral_model="no_loss", aoi_model="physical"
            )
            peakpower = module.STC
            specific_area = module.STC / module.A_c / 1000 / 1000

            return pv_model, peakpower, specific_area

        # Define parameters for convinience
        lon = location["lon"]
        lat = location["lat"]
        alt = location["alt"]

        # Get location
        tf = TimezoneFinder()
        tz = tf.timezone_at(lng=lon, lat=lat)
        location = pvlib.location.Location(lat, lon, tz=tz, altitude=alt)

        # Initialize pv_system
        pv_model, peakpower, specific_area = define_pv_system(location, system_data)

        # Run system with climate data
        pv_model.run_model(climate_data)

        # Calculate cap factors
        power = pv_model.results.ac.p_mp
        capacity_factor = power / peakpower

        # Calculate output bounds
        lower_output_bound = np.zeros(shape=(len(climate_data)))
        upper_output_bound = capacity_factor.to_numpy()
        output_bounds = np.column_stack((lower_output_bound, upper_output_bound))

        # Output Bounds
        self.fitted_performance.bounds["output"]["electricity"] = output_bounds
        # Coefficients
        self.fitted_performance.coefficients["capfactor"] = round(capacity_factor, 3)
        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 1
        # Other Data
        self.fitted_performance.other["specific_area"] = specific_area

    def _perform_fitting_ST(self, climate_data):
        """
        Calculates capacity factors and specific area requirements for a solar thermal system

        :param climate_data: contains information on weather data
        :return: returns capacity factors and specific area requirements
        """
        # Todo: code this
        print("Not coded yet")

    def _perform_fitting_WT(self, climate_data, hubheight):
        """
        Calculates capacity factors for a wint turbine

        :param climate_data: contains information on weather data
        :param hubheight: hubheight of wind turbine
        """
        # Load data for wind turbine type
        # FIXME: find nicer way to do this
        WT_path = Path(__file__).parent.parent.parent.parent.parent
        WT_data_path = WT_path / "data/technology_data/RES/WT_data/WT_data.csv"
        WT_data = pd.read_csv(WT_data_path, delimiter=";")

        # match WT with data
        if self.name in WT_data["TurbineName"]:
            WT_data = WT_data[WT_data["TurbineName"] == self.name]
        else:
            WT_data = WT_data[WT_data["TurbineName"] == "WindTurbine_Onshore_1500"]
            warnings.warn(
                "TurbineName not in csv, standard WindTurbine_Onshore_1500 selected."
            )

        # Load wind speed and correct for height
        ws = climate_data["ws10"]

        # TODO: make power exponent choice possible
        # TODO: Make different heights possible
        alpha = 1 / 7
        # if data.node_data.windPowerExponent(node) >= 0
        #     alpha = data.node_data.windPowerExponent(node);
        # else:
        #     if data.node_data.offshore(node) == 1:
        #         alpha = 0.45;
        #     else:
        #         alpha = 1 / 7;

        if hubheight > 0:
            ws = ws * (hubheight / 10) ** alpha

        # Make power curve
        rated_power = WT_data.iloc[0]["RatedPowerkW"]
        x = np.linspace(0, 35, 71)
        y = WT_data.iloc[:, 13:84]
        y = y.to_numpy()

        f = interp1d(x, y)
        ws[ws < 0] = 0
        capacity_factor = f(ws) / rated_power

        # Calculate output bounds
        lower_output_bound = np.zeros(shape=(len(climate_data)))
        upper_output_bound = capacity_factor[0]
        output_bounds = np.column_stack((lower_output_bound, upper_output_bound))

        # Output Bounds
        self.fitted_performance.bounds["output"]["electricity"] = output_bounds
        # Coefficients
        self.fitted_performance.coefficients["capfactor"] = capacity_factor[0].round(3)
        # Time dependent coefficents
        self.fitted_performance.time_dependent_coefficients = 1
        # Other Data
        self.fitted_performance.rated_power = rated_power / 1000

    def construct_tech_model(self, b_tec, data, set_t, set_t_clustered):
        """
        Adds constraints to technology blocks for tec_type RES (renewable technology)

        :param obj b_tec: technology block
        :param Energyhub energyhub: energyhub instance
        :return: technology block
        """
        super(Res, self).construct_tech_model(b_tec, data, set_t, set_t_clustered)

        # DATA OF TECHNOLOGY
        performance_data = self.performance_data
        coeff = self.fitted_performance.coefficients
        rated_power = self.fitted_performance.rated_power

        if "curtailment" in performance_data:
            curtailment = performance_data["curtailment"]
        else:
            curtailment = 0

        # PARAMETERS
        # Set capacity factors
        capfactor = coeff["capfactor"]

        # CONSTRAINTS
        if curtailment == 0:  # no curtailment allowed (default)

            def init_input_output(const, t, c_output):
                return (
                    self.output[t, c_output]
                    == capfactor[t - 1] * b_tec.var_size * rated_power
                )

            b_tec.const_input_output = Constraint(
                self.set_t, b_tec.set_output_carriers, rule=init_input_output
            )

        elif curtailment == 1:  # continuous curtailment

            def init_input_output(const, t, c_output):
                return (
                    self.output[t, c_output]
                    <= capfactor[t - 1] * b_tec.var_size * rated_power
                )

            b_tec.const_input_output = Constraint(
                self.set_t, b_tec.set_output_carriers, rule=init_input_output
            )

        elif curtailment == 2:  # discrete curtailment
            b_tec.var_size_on = Var(
                self.set_t,
                within=NonNegativeIntegers,
                bounds=(b_tec.para_size_min, b_tec.para_size_max),
            )

            def init_curtailed_units(const, t):
                return b_tec.var_size_on[t] <= b_tec.var_size

            b_tec.const_curtailed_units = Constraint(
                self.set_t, rule=init_curtailed_units
            )

            def init_input_output(const, t, c_output):
                return (
                    self.output[t, c_output]
                    == capfactor[t - 1] * b_tec.var_size_on[t] * rated_power
                )

            b_tec.const_input_output = Constraint(
                self.set_t, b_tec.set_output_carriers, rule=init_input_output
            )

        return b_tec

    def write_results_tec_design(self, h5_group, model_block):

        super(Res, self).write_results_tec_design(h5_group, model_block)

        h5_group.create_dataset("rated_power", data=self.fitted_performance.rated_power)

    def write_results_tec_operation(self, h5_group, model_block):

        super(Res, self).write_results_tec_operation(h5_group, model_block)

        rated_power = self.fitted_performance.rated_power
        capfactor = self.fitted_performance.coefficients["capfactor"]

        h5_group.create_dataset(
            "max_out",
            data=[
                capfactor[t - 1] * model_block.var_size.value * rated_power
                for t in self.set_t
            ],
        )

        h5_group.create_dataset(
            "cap_factor", data=self.fitted_performance.coefficients["capfactor"]
        )

        if self.performance_data["curtailment"] == 2:
            h5_group.create_dataset(
                "units_on", data=[model_block.var_size_on[t].value for t in self.set_t]
            )

        for car in model_block.set_output_carriers:
            h5_group.create_dataset(
                "curtailment_" + car,
                data=[
                    capfactor[t - 1] * model_block.var_size.value * rated_power
                    - model_block.var_output[t, car].value
                    for t in self.set_t
                ],
            )
