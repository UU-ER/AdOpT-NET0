from datetime import datetime
import os
from pathlib import Path

def create_save_folder(save_path):
    # Get the current time
    current_time = datetime.now()

    # Format the time as a string (e.g., "2023-09-28_12-34-56")
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")

    # Create the folder with the formatted time as its name
    folder_name = Path.joinpath(save_path, formatted_time)
    os.makedirs(folder_name)

    return folder_name