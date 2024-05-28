import pandas as pd
import numpy as np


def fit_ccs_data(
    co2_concentration: float, ccs_data: dict, climate_data: pd.DataFrame
) -> dict:
    """
    Obtain bounds and input ratios for CCS

    Calculates the amount of input (and their bounds) required for each unit of CO2 entering the carbon capture (CC)
    object. The minimum and maximum size parameters are multiplied by the CO2 concentration, so that the units
    of the size becomes t/h of CO2 in. These are also the units used for the rest of the model.
    So far, only post-combustion MEA has been modelled (based on Eq. 7 in Weimann et Al. (2023), A thermodynamic-based
    mixed-integer linear model of post-combustion carbon capture for reliable use in energy system optimisation
    https://doi.org/10.1016/j.apenergy.2023.120738).

    :param float co2_concentration: CO2 concentration for ccs
    :param dict ccs_data: data of the CCS technology
    :param pd.Dataframe climate_data: dataframe containing climate data
    :return: ccs data updated with the bounds and input ratios for CCS
    """

    performance_data = ccs_data["TechnologyPerf"]
    time_steps = len(climate_data)
    molar_mass_CO2 = 44.01
    carbon_capture_rate = performance_data["capture_rate"]

    # Recalculate min/max size to have it in t/hCO2_in
    ccs_data["size_min"] = ccs_data["size_min"] * co2_concentration
    ccs_data["size_max"] = ccs_data["size_max"] * co2_concentration

    # Calculate input ratios
    if ccs_data["ccs_type"] == "MEA":
        ccs_data["TechnologyPerf"]["input_ratios"] = {}
        for car in ccs_data["TechnologyPerf"]["input_carrier"]:
            ccs_data["TechnologyPerf"]["input_ratios"][car] = (
                ccs_data["TechnologyPerf"]["eta"][car]
                + ccs_data["TechnologyPerf"]["omega"][car] * co2_concentration
            ) / (co2_concentration * molar_mass_CO2 * 3.6)
    else:
        raise Exception(
            "Only CCS type MEA is modelled so far. ccs_type in the json file of the technology must include MEA"
        )

    # Calculate input and output bounds
    ccs_data["TechnologyPerf"]["bounds"] = {}
    ccs_data["TechnologyPerf"]["bounds"]["input"] = {}
    ccs_data["TechnologyPerf"]["bounds"]["output"] = {}
    for car in ccs_data["TechnologyPerf"]["input_carrier"]:
        ccs_data["TechnologyPerf"]["bounds"]["input"][car] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=(time_steps))
                * ccs_data["TechnologyPerf"]["input_ratios"][car],
            )
        )
    for car in ccs_data["TechnologyPerf"]["output_carrier"]:
        ccs_data["TechnologyPerf"]["bounds"]["output"][car] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=(time_steps)) * carbon_capture_rate,
            )
        )

    return ccs_data
