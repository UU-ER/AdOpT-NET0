"""
Calculates capacity factors for a PV system using pvlib
"""

import pvlib
from timezonefinder import TimezoneFinder
import numpy as np
import pandas as pd

def _define_pv_system(location: dict, system_data: dict):
    """
    defines the pv system
    :param dict location: location information (latitude, longitude, altitude,
    time zone)
    :param dict system_data: contains data on tilt, surface_azimuth,
    module_name, inverter efficiency
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

def calculate_performance_pv(climate_data: pd.DataFrame, location: dict, **kwargs):
    """
    Calculates capacity factors and specific area requirements for a PV system using pvlib

    :param pd.Dataframe climate_data: dataframe containing climate data
    :param dict location: dict containing location details
    :param PV_type: (optional) can specify a certain type of module, angle, ...
    """
    if not kwargs.__contains__("system_data"):
        system_data = dict()
        system_data["tilt"] = 18
        system_data["surface_azimuth"] = 180
        system_data["module_name"] = "SunPower_SPR_X20_327"
        system_data["inverter_eff"] = 0.96
    else:
        system_data = kwargs["system_data"]

    # Define parameters for convinience
    lon = location["lon"]
    lat = location["lat"]
    alt = location["alt"]

    if (
            (np.isnan(location["lon"]))
            or (np.isnan(location["lat"]))
            or (np.isnan(location["alt"]))
    ):
        raise Exception(
            "To use Photovoltaic technology you need to specify a "
            "location in the NodeLocations.csv file"
        )

    # Get location
    tf = TimezoneFinder()
    tz = tf.timezone_at(lng=lon, lat=lat)
    location = pvlib.location.Location(lat, lon, tz=tz, altitude=alt)

    # Initialize pv_system
    pv_model, peakpower, specific_area = _define_pv_system(location, system_data)

    # Run system with climate data
    pv_model.run_model(climate_data)

    # Calculate cap factors
    power = pv_model.results.ac.p_mp
    technology_time_series = pd.DataFrame()
    technology_time_series["capfactor"] = round(power / peakpower, 3)

    # Coefficients
    return technology_time_series