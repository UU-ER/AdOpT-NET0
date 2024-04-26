from datetime import datetime
import os
from pathlib import Path


def create_unique_folder_name(path, name):
    """
    Creates a unique folder name, in case the specified name already exists in path
    :param path: path to check
    :param name: folder name
    :return:
    """
    folder_path = Path.joinpath(path, name)
    counter = 1
    while folder_path.is_dir():
        folder_path = folder_path.with_name(
            f"{folder_path.stem}_{counter}{folder_path.suffix}"
        )
        counter += 1
    return folder_path


def create_save_folder(save_path):
    """
    Creates a new folder at save_path

    :param str save_path: path to create folder at
    :return:
    """
    os.makedirs(save_path)


def get_time_stage(energyhub):
    """
    Gets time stage
    """
    if config["optimization"]["timestaging"]["value"]:
        time_stage = energyhub.model_information.averaged_data_specs.stage + 1
    else:
        time_stage = 0
    return time_stage
