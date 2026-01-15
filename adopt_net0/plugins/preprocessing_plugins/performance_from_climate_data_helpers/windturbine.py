"""
Calculates capacity factors for a wind turbine
"""

import warnings
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from importlib.resources import files
from . import __package__

def calculate_performance_wt(climate_data: pd.DataFrame, hubheight: float, turbine_name: str):
    """
    Calculates capacity factors for a wind turbine

    The power curves are located in ``adopt_net0/plugins/preprocessing_plugins/performance_from_climate_data_helpers/data/WT_data.csv``

    :param pd.Dataframe technology_time_series: dataframe containing climate data
    :param float hubheight: hubheight of wind turbine
    """
    # Load data for wind turbine type
    wt_data_path = files(__package__) / "data" / "WT_data.csv"

    wt_data_full = pd.read_csv(wt_data_path, delimiter=";")

    # match WT with data
    wt_data = wt_data_full[wt_data_full["TurbineName"] == turbine_name]

    if len(wt_data) == 0:
        wt_data = wt_data_full[
            wt_data_full["TurbineName"] == "WindTurbine_Onshore_1500"
            ]
    warnings.warn(
        "TurbineName not in csv, standard WindTurbine_Onshore_1500 selected."
    )

    # Load wind speed and correct for height
    if "ws100" in climate_data:
        ws = climate_data["ws100"]
        ws_height = 100
    else:
        ws = climate_data["ws10"]
        ws_height = 10

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
        ws = ws * (hubheight / ws_height) ** alpha

    # Make power curve
    rated_capacity = wt_data.iloc[0]["RatedPowerkW"]
    x = np.linspace(0, 35, 71)
    y = wt_data.iloc[:, 13:84]
    y = y.to_numpy()

    f = interp1d(x, y)
    ws[ws < 0] = 0
    capacity_factor = f(ws) / rated_capacity

    technology_time_series = pd.DataFrame(index=climate_data.index)
    technology_time_series["capfactor"] = capacity_factor[0].round(3)

    return technology_time_series
