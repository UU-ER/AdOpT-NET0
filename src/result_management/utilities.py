from datetime import datetime
import os
from pathlib import Path

def create_save_folder(save_path, timestamp):
    folder_name = Path.joinpath(save_path, timestamp)
    os.makedirs(folder_name)

    return folder_name