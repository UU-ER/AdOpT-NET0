import pandas as pd

from ...component import ModelComponent


class CcsComponent(ModelComponent):
    """
    Class for ccs attachement to technology
    """

    def __init__(self, ccs_data: dict):
        """
        Initializes ccs class from technology data

        :param dict tec_data: technology data
        """
        super().__init__(ccs_data)

        self.technology_model = ccs_data["tec_type"]
        self.input_carrier = ccs_data["Performance"]["input_carrier"]
        self.output_carrier = ccs_data["Performance"]["output_carrier"]


def fit_ccs_coeff(co2_concentration: float, ccs_data: dict, climate_data: pd.DataFrame):
    """
    Obtain bounds and input ratios for CCS

    Calculates the amount of input (and their bounds) required for each unit of CO2 entering the carbon capture (CC)
    object. The minimum and maximum size parameters are multiplied by the CO2 concentration and capture rate, so that the units
    of the size becomes t/h of CO2 out. These are also the units used for the rest of the model (e.g. size of CC unit).
    So far, only post-combustion MEA has been modelled (based on Eq. 7 in Weimann et Al. (2023), A thermodynamic-based
    mixed-integer linear model of post-combustion carbon capture for reliable use in energy system optimisation
    https://doi.org/10.1016/j.apenergy.2023.120738).

    :param float co2_concentration: CO2 concentration for CCS
    :param dict ccs_coeff: data of the CCS technology
    :param pd.Dataframe climate_data: dataframe containing climate data
    :return: CCS data updated with the bounds and input ratios for CCS
    """
    molar_mass_CO2 = 44.01
    # convert kmol/s of fluegas to ton/h of molar_mass_CO2 = 44.01
    convert2t_per_h = molar_mass_CO2 * co2_concentration * 3.6
    capture_rate = ccs_data["Performance"]["capture_rate"]
    # Recalculate unit_capex in EUR/(t_CO2out/h)
    ccs_data["Economics"]["unit_capex"] = (
        (
            ccs_data["Economics"]["CAPEX_kappa"] / convert2t_per_h
            + ccs_data["Economics"]["CAPEX_lambda"]
        )
        * co2_concentration
    ) / convert2t_per_h

    ccs_data["Economics"]["fix_capex"] = ccs_data["Economics"]["CAPEX_zeta"]

    ccs_data = CcsComponent(ccs_data)

    # Recalculate min/max size to have it in t/hCO2_out
    ccs_data.size_min = ccs_data.size_min * co2_concentration * capture_rate
    ccs_data.size_max = ccs_data.size_max * co2_concentration * capture_rate

    # Calculate input ratios
    ccs_data.processed_coeff.time_independent["size_min"] = ccs_data.size_min
    ccs_data.processed_coeff.time_independent["size_max"] = ccs_data.size_max
    ccs_data.processed_coeff.time_independent["capture_rate"] = capture_rate
    if "MEA" in ccs_data.technology_model:
        input_ratios = {}
        for car in ccs_data.input_carrier:
            input_ratios[car] = (
                ccs_data.performance_data["eta"][car]
                + ccs_data.performance_data["omega"][car] * co2_concentration
            ) / (co2_concentration * molar_mass_CO2 * 3.6)
        ccs_data.processed_coeff.time_independent["input_ratios"] = input_ratios
    else:
        raise Exception(
            "Only CCS type MEA is modelled so far. ccs_type in the json file of the "
            "technology must include MEA"
        )

    return ccs_data
