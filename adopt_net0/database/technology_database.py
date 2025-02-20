import pandas as pd
from pathlib import Path

from .technologies import *

PATH_CURRENT_DIR = Path(__file__).parent
PATH_AVAILABLE_COMPONENTS = PATH_CURRENT_DIR / Path("./data/available_components.csv")


def write_json(component_name: str, directory: str, options):
    """
    Writes a json file of the component to the specified directory

    :param str component_name: Name of the technology/network
    :param str directory: directory to write to
    :param dict options: options used in the calculations
    :return:
    """
    component = _component_factory(component_name, options)
    component.write_json(directory)


def calculate_financial_indicators(component_name: str, options: dict) -> dict:
    """
    Calculates financial parameters based on the component and the provided options

    If no options are provided, the default options are used

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
    component = _component_factory(component_name, options)
    return component.calculate_financial_indicators()


def _component_factory(component_name: str, options: dict):
    """
    Creates component class

    :param str component_name: Name of the technology/network
    :return: component class
    """
    if component_name == "DAC - Solid Sorbent":
        return Dac_SolidSorbent(options)
    else:
        raise NotImplementedError("Technology not implemented or spelled incorrectly")


def help(component_name: str = None):
    """
    Provides help on available technologies and networks.

    - If no argument is provided, it prints all available components.
    - If a component name is provided, it prints detailed information about that component.

    :param str component_name: Name of the technology/network (optional)
    """
    if component_name is None:
        _help_available_components()
    else:
        _help_component(component_name)


def _help_available_components():
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
    options = {"currency_out": "", "financial_year_out": 9999, "discount_rate": 999}
    component = _component_factory(component_name, options)
    print(component.__doc__)
    print("Default options are:")
    for o in component.default_options:
        print(o, ":", component.default_options[o])
