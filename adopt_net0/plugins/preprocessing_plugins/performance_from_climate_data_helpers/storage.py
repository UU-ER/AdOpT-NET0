"""
Calculates efficiency and specific energy requirements for a storage system

Uses the following equation:

.. math::
   ambient\_loss\_factor = (65 - T) / (90 - 65) * theta

"""

import pandas as pd

def calculate_performance_stor(climate_data: pd.DataFrame, technology_data: dict):
    """
    Calculates efficiency and specific energy requirements for a storage system

    :param pd.Dataframe climate_data: climate data
    :param dict technology_data: contains data on ambient loss factor
    :return: returns dataframe ambient loss factor time series
    """
    # For thermal storage, we might consider ambient temperature effects on losses
    theta = technology_data["Performance"]["performance"]["theta"]
    ambient_loss_factor = (65 -  climate_data["temp_air"]) / (90 - 65) * theta

    technology_time_series = pd.DataFrame(index=climate_data.index)
    technology_time_series["ambient_loss_factor"] = ambient_loss_factor[0].round(3)

    return technology_time_series
