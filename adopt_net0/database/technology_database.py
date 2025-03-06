import os

import pandas as pd
from pathlib import Path

from .technologies import *
from .networks import *
from .data_component import DataComponent_CostModel

PATH_CURRENT_DIR = Path(__file__).parent
PATH_AVAILABLE_COMPONENTS = PATH_CURRENT_DIR / Path("data/available_cost_models.csv")


def write_json(component_name: str, directory: str, options):
    """
    Writes a json file of the component to the specified directory

    :param str component_name: Name of the technology/network
    :param str directory: directory to write to
    :param dict options: options used in the calculations
    :return:
    """
    component = _component_factory(component_name)
    component.write_json(directory, options)
    return component


def calculate_indicators(component_name: str, options: dict) -> dict:
    """
    Calculates financial parameters based on the component and the provided options

    If no options are provided, the default options are used. Calculates:

    - lifetime
    - discount rate
    - capex
    - fixed opex
    - variable opex
    - levelized costs (if available)

    :param str component_name: Name of the technology/network
    :param dict options: options used in the calculations
    :return: dictionary containing financial parameters
    :rtype: dict
    """
    component = _component_factory(component_name)
    return component.calculate_indicators(options)


def _component_factory(component_name: str):
    """
    Creates component class

    :param str component_name: Name of the technology/network
    :return: component class
    """
    if component_name == "DAC_Adsorption":
        return Dac_SolidSorbent_CostModel(component_name)
    elif component_name == "CO2_Pipeline":
        return CO2_Pipeline_CostModel(component_name)
    elif component_name == "CO2_Compressor":
        return CO2_Compression_CostModel(component_name)
    elif "HeatPump" in component_name:
        return HeatPump_CostModel(component_name)
    elif "WindTurbine" in component_name:
        return WindEnergy_CostModel(component_name)
    elif component_name == "Photovoltaic":
        return PV_CostModel(component_name)
    else:
        return DataComponent_CostModel(component_name)


def help(component_name: str = None):
    """
    Provides help on available cost models of technologies and networks.

    - If no argument is provided, it prints all available cost models.
    - If a component name is provided, it prints detailed information about that component.

    :param str component_name: Name of the technology/network (optional)
    """
    if component_name is None:
        _help_available_cost_models()
    else:
        _help_component(component_name)


def _help_available_cost_models():
    """
    Prints available technologies and networks
    """
    available_components = pd.read_csv(PATH_AVAILABLE_COMPONENTS, sep=";")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(available_components.sort_values(by="Component_Name"))


def _help_component(component_name: str):
    """
    Prints component description and options

    :param str component_name: Name of the technology/network
    """
    component = _component_factory(component_name)
    print(component.__doc__)
    print("Default options are:")
    for o in component.default_options:
        print(o, ":", component.default_options[o])


def show_available_networks():
    """
    Prints all available networks
    """
    tec_data_path = Path(
        os.path.join(os.path.dirname(__file__) + "/../database/network_data")
    )

    for root, dirs, files in os.walk(tec_data_path.resolve()):
        for file in files:
            print(file[:-5])


def show_available_technologies():
    """
    Prints all available technologies
    """
    tec_data_path = Path(
        os.path.join(os.path.dirname(__file__) + "/../database/templates")
    )

    for root, dirs, files in os.walk(tec_data_path.resolve()):
        for file in files:
            print(file[:-5])
