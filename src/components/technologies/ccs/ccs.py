import pandas as pd
import numpy as np

from ...component import ModelComponent


def fit_ccs_coeff(co2_concentration: float, ccs_data: dict, climate_data: pd.DataFrame):
    """
    Obtain bounds and input ratios for CCS

    Calculates the amount of input (and their bounds) required for each unit of CO2 entering the carbon capture (CC)
    object. The minimum and maximum size parameters are multiplied by the CO2 concentration, so that the units
    of the size becomes t/h of CO2 in. These are also the units used for the rest of the model.
    So far, only post-combustion MEA has been modelled (based on Eq. 7 in Weimann et Al. (2023), A thermodynamic-based
    mixed-integer linear model of post-combustion carbon capture for reliable use in energy system optimisation
    https://doi.org/10.1016/j.apenergy.2023.120738).

    :param float co2_concentration: CO2 concentration for ccs
    :param dict ccs_coeff: data of the CCS technology
    :param pd.Dataframe climate_data: dataframe containing climate data
    :return: ccs data updated with the bounds and input ratios for CCS
    """
    # Recalculate unit_capex
    molar_mass_CO2 = 44.01
    # convert kmol/s of fluegas to ton/h of CO2molar_mass_CO2 = 44.01
    convert2t_per_h = molar_mass_CO2 * co2_concentration * 3.6

    capture_rate = ccs_data["Performance"]["capture_rate"]
    ccs_data["Economics"]["unit_CAPEX"] = (
        (
            ccs_data["Economics"]["CAPEX_kappa"] / convert2t_per_h
            + ccs_data["Economics"]["CAPEX_lambda"]
        )
        * capture_rate
        * co2_concentration
    ) / convert2t_per_h

    ccs_data["Economics"]["fix_CAPEX"] = ccs_data["Economics"]["CAPEX_zeta"]

    ccs_data = ModelComponent(ccs_data)

    time_steps = len(climate_data)
    molar_mass_CO2 = 44.01

    # Recalculate min/max size to have it in t/hCO2_in
    ccs_data.parameters.size_min = ccs_data.parameters.size_min * co2_concentration
    ccs_data.parameters.size_max = ccs_data.parameters.size_max * co2_concentration

    # Calculate input ratios
    if "MEA" in ccs_data.info.technology_model:
        input_ratios = {}
        for car in ccs_data.info.input_carrier:
            input_ratios[car] = (
                ccs_data.parameters.unfitted_data["eta"][car]
                + ccs_data.parameters.unfitted_data["omega"][car] * co2_concentration
            ) / (co2_concentration * molar_mass_CO2 * 3.6)
        ccs_data.coeff.time_independent["input_ratios"] = input_ratios
    else:
        raise Exception(
            "Only CCS type MEA is modelled so far. ccs_type in the json file of the technology must include MEA"
        )

    # Calculate input and output bounds
    for car in ccs_data.info.input_carrier:
        ccs_data.bounds["input"][car] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=(time_steps))
                * ccs_data.coeff.time_independent["input_ratios"][car],
            )
        )
    for car in ccs_data.info.output_carrier:
        ccs_data.bounds["output"][car] = np.column_stack(
            (
                np.zeros(shape=(time_steps)),
                np.ones(shape=(time_steps)) * capture_rate,
            )
        )

    return ccs_data
