from datetime import datetime
import os
from pathlib import Path
from ..energyhub import EnergyHub


def create_unique_folder_name(path: Path, name: str) -> Path:
    """
    Creates a unique folder name, in case the specified name already exists in the given path.

    The unique folder name is either the given folder name if the folder name did not exist yet in the given path, or
    the given folder name with an added suffix (_1, _2, _3, etc.).
    :param Path path: path to check
    :param str name: folder name
    :return: path to the folder with the unique folder name
    :rtype: Path
    """
    folder_path = Path.joinpath(path, name)
    counter = 1
    while folder_path.is_dir():
        folder_path = folder_path.with_name(
            f"{folder_path.stem}_{counter}{folder_path.suffix}"
        )
        counter += 1
    return folder_path


def create_save_folder(save_path: Path):
    """
    Creates a new folder at save_path

    :param Path save_path: path at which the folder is created
    :return:
    """
    os.makedirs(save_path)


def get_time_stage(energyhub: EnergyHub) -> int:
    """
    Gets time stage from the configuration

    :param: energyhub
    :return: time stage
    :rtype: int
    """

    # fixme with algorithms
    if config["optimization"]["timestaging"]["value"]:
        time_stage = energyhub.model_information.averaged_data_specs.stage + 1
    else:
        time_stage = 0
    return time_stage
