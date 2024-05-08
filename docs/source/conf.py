import os, sys, json

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

project = "EnergyHub"
copyright = "2023, Jan Wiegner, Julia Tiggeloven, Luca Bertoni, Inge Ossentjuk"
author = "Jan Wiegner, Julia Tiggeloven, Luca Bertoni, Inge Ossentjuk"
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
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
add_module_names = False

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

            if "tec_type" in data:
                tec_type = data.get("tec_type", "")
                component_ls.append((name, tec_type))
            else:
                component_ls.append(name)

    return component_ls


# specify path to technology json files relative to current folder (not user-dependent)
current_dir = os.path.abspath(os.path.dirname(__file__))
target_dir = os.path.abspath(
    os.path.join(current_dir, "..", "..", "data/technology_data")
)
tech_list = generate_component_list(target_dir)

with open("src_code/model/model_components/generated_tech_list.csv", "w") as f:
    f.write(f"Technology name; Technology model (Tec_type)\n")
    for tech in tech_list:
        f.write(f"{tech[0]}; {tech[1]}\n")

# specify path to network json files relative to current folder (not user-dependent)
target_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "data/network_data"))
netw_list = generate_component_list(target_dir)


with open("src_code/model/model_components/generated_netw_list.csv", "w") as f:
    f.write(f"Network name\n")
    for netw in netw_list:
        f.write(f"{netw}\n")
