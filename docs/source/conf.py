import os, sys, json, csv
from pathlib import Path

sys.path.insert(0, os.path.abspath("../.."))
mathjax_path = (
    "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "AdOpT-NET0"
copyright = (
    "2023, Jan F. Wiegner, Julia L. Tiggeloven, Luca Bertoni, Inge M. Ossentjuk, "
    "Matteo Gazzani"
)
author = (
    "Jan F. Wiegner, Julia L. Tiggeloven, Luca Bertoni, Inge M. Ossentjuk, "
    "Matteo Gazzani"
)
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = []

# Dont run notebooks
nb_execution_mode = "off"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_extra_path = [""]
add_module_names = False

# -- create table with configuration settings for documentation --------------------

# Import the function that created the dictionary
from adopt_net0.data_preprocessing.template_creation import (
    initialize_configuration_templates,
)

# Call the function to get the configuration dictionary
config_dict = initialize_configuration_templates()

# Define the path to the csv file
output_path = os.path.join(os.path.dirname(__file__), "config.csv")


# method to flatten the nested dictionary to a list of tuples
def flatten_dict(d, parent_key=()):
    items = []
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            if "description" in v or "options" in v or "value" in v:
                description = v.get("description", "")
                options = v.get("options", "")
                value = v.get("value", "")
                # Check the depth of the parent_key
                if len(new_key) < 3:
                    new_key += ("",)
                items.append(new_key + (description, options, value))
            else:
                items.extend(flatten_dict(v, new_key))
        else:
            items.append(new_key + (v,))
    return items


# Flatten the config_dict into a list of tuples
config_rows = flatten_dict(config_dict)

# Write the flattened data to CSV
with open("advanced_topics/config.csv", "w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write rows
    csv_writer.writerows(config_rows)


# -- create list of technologies and networks for documentation ---------------------
def generate_component_list(directory):
    component_ls = []

    # Walk through the directory and its subfolders
    for root, dirs, files in os.walk(directory):
        print(f"Searching in {root}")
        # Filter JSON files
        json_files = [f for f in files if f.endswith(".json")]
        print(f"Found JSON files: {json_files}")

        # Process each JSON file
        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            print(f"Processing file: {file_path}")
            name = os.path.splitext(os.path.basename(json_file))[0]

            # Open and parse the JSON file
            with open(file_path, "r") as f:
                data = json.load(f)

            if "technology" in str(directory):
                if "tec_type" in data:
                    tec_type = data.get("tec_type", "")
                    component_ls.append((name, tec_type))
            elif "network" in str(directory):
                if "network_type" in data:
                    network_type = data.get("network_type", "")
                    component_ls.append((name, network_type))
    return component_ls


# specify path to technology json files relative to current folder (not user-dependent)
target_dir = Path(__file__).parent.parent.parent / "adopt_net0/data/technology_data"
tech_list = generate_component_list(target_dir)

with open("src_code/model_components/generated_tech_list.csv", "w") as f:
    f.write(f"Technology name; Technology model (Tec_type)\n")
    for tech in tech_list:
        f.write(f"{tech[0]}; {tech[1]}\n")

# specify path to network json files relative to current folder (not user-dependent)
target_dir = Path(__file__).parent.parent.parent / "adopt_net0/data/network_data"
netw_list = generate_component_list(target_dir)


with open("src_code/model_components/generated_netw_list.csv", "w") as f:
    f.write(f"Network name; Network_type \n")
    for netw in netw_list:
        f.write(f"{netw[0]}; {netw[1]} \n")
