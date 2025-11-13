import os, sys, json, csv
import subprocess
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
    "sphinx_rtd_theme",
    "myst_nb",
]
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = []

# Dont run notebooks
nb_execution_mode = "off"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
}

html_extra_path = [""]
add_module_names = False

# html_logo = "logo/SVG/Adopt_icononly.svg"
html_logo = "logo/SVG/Adopt_fulllogo.svg"
html_favicon = "logo/SVG/Adopt_icononly.svg"


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
def get_git_info(repo_path):
    """
    Get Git repository information (remote URL and current branch/commit)
    """
    try:
        # Change to the repository directory
        original_cwd = os.getcwd()
        os.chdir(repo_path)

        # Get the remote origin URL
        remote_url = (
            subprocess.check_output(
                ["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        # Convert SSH URL to HTTPS if needed
        if remote_url.startswith("git@github.com:"):
            remote_url = remote_url.replace("git@github.com:", "https://github.com/")
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]

        # Get current branch name
        try:
            current_branch = (
                subprocess.check_output(
                    ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except:
            # If detached HEAD, get the commit hash
            current_branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )

        # Restore original directory
        os.chdir(original_cwd)

        return remote_url, current_branch

    except Exception as e:
        print(f"Warning: Could not get Git info: {e}")
        # Fallback to default values
        return "https://github.com/UU-ER/AdOpT-NET0", "main"


def generate_component_list(directory, base_github_url=None):
    component_ls = []

    # Get Git repository information if not provided
    if base_github_url is None:
        repo_path = Path(__file__).parent.parent.parent
        remote_url, current_branch = get_git_info(repo_path)
        base_github_url = f"{remote_url}/blob/{current_branch}"
        print(f"Using Git repository: {remote_url}")
        print(f"Using branch/commit: {current_branch}")

    # Walk through the directory and its subfolders
    for root, dirs, files in os.walk(directory):
        # Filter JSON files
        json_files = [f for f in files if f.endswith(".json")]

        # Process each JSON file
        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            name = os.path.splitext(os.path.basename(json_file))[0]

            # Get the technology group (folder name)
            component_group = os.path.basename(root)

            # Create GitHub URL for the JSON file
            # Get relative path from the adopt-net0 root
            rel_path = os.path.relpath(
                file_path, str(Path(__file__).parent.parent.parent)
            )
            github_url = f"{base_github_url}/{rel_path.replace(os.sep, '/')}"

            # Open and parse the JSON file
            with open(file_path, "r") as f:
                data = json.load(f)

            if "technology" in str(directory):
                if "tec_type" in data:
                    tec_type = data.get("tec_type", "")
                    # Create clickable link for technology name in reStructuredText format
                    clickable_name = f"`{name} <{github_url}>`_"
                    component_ls.append((clickable_name, tec_type, component_group))
            elif "network" in str(directory):
                if "network_type" in data:
                    network_type = data.get("network_type", "")
                    # Create clickable link for network name in reStructuredText format
                    clickable_name = f"`{name} <{github_url}>`_"
                    component_ls.append((clickable_name, network_type))
    return component_ls


# specify path to technology json files relative to current folder
target_dir = (
    Path(__file__).parent.parent.parent
    / "adopt_net0/database/templates/technology_data"
)
tech_list = generate_component_list(target_dir)

with open("database/generated_tech_list.csv", "w") as f:
    f.write(f"Technology group; Technology name; Technology model (Tec_type)\n")
    for tech in tech_list:
        f.write(f"{tech[2]}; {tech[0]}; {tech[1]}\n")

# specify path to network json files relative to current folder (not user-dependent)
target_dir = (
    Path(__file__).parent.parent.parent / "adopt_net0/database/templates/network_data"
)
netw_list = generate_component_list(target_dir)


with open("database/generated_netw_list.csv", "w") as f:
    f.write(f"Network name; Network type\n")
    for netw in netw_list:
        f.write(f"{netw[0]}; {netw[1]}\n")


# Generate GitHub link for Components database Excel file
def generate_components_database_link():
    """Generate GitHub link for the Components database Excel file"""
    repo_path = Path(__file__).parent.parent.parent
    remote_url, current_branch = get_git_info(repo_path)
    base_github_url = f"{remote_url}/blob/{current_branch}"

    # Path to the Excel file relative to repository root
    excel_rel_path = "adopt_net0/database/data/Components_database.xlsx"
    # encode spaces (none expected) and build URL
    github_excel_url = f"{base_github_url}/{excel_rel_path}"

    return github_excel_url


def setup_components_database_download():
    """Setup direct download for Components database Excel file"""
    # Source Excel file path
    source_excel = (
        Path(__file__).parent.parent.parent
        / "adopt_net0"
        / "database"
        / "data"
        / "Components_database.xlsx"
    )

    # Create _static directory if it doesn't exist
    static_dir = Path(__file__).parent / "_static"
    static_dir.mkdir(exist_ok=True)

    # Destination path in _static directory
    dest_excel = static_dir / "Components_database.xlsx"

    # Copy Excel file to _static directory for direct download
    import shutil

    if source_excel.exists():
        shutil.copy2(source_excel, dest_excel)
        return True
    else:
        print(f"Warning: Source Excel file not found at {source_excel}")
        return False


# Setup direct download and generate both GitHub and download links
excel_copied = setup_components_database_download()
excel_github_url = generate_components_database_link()

# Auto-regenerate Components database Excel file during documentation build
print("Regenerating Components database Excel file...")
try:
    # Load the docs-only utilities module from the same docs folder
    from importlib.util import spec_from_file_location, module_from_spec

    docs_utils_path = Path(__file__).parent / "utilities_documentation.py"
    if docs_utils_path.exists():
        spec = spec_from_file_location("docs_utilities", str(docs_utils_path))
        docs_utils = module_from_spec(spec)
        spec.loader.exec_module(docs_utils)

        # Call the docs-local generator
        excel_path = docs_utils.create_csv_database_from_json()
        if excel_path:
            print(f"Components database updated successfully: {excel_path}")
            # Re-copy the updated Excel file to _static directory
            excel_copied = setup_components_database_download()
        else:
            print("Warning: Failed to update Components database")
    else:
        print(f"Warning: docs utilities module not found at {docs_utils_path}")

except Exception as e:
    print(f"Warning: Could not auto-regenerate Components database: {e}")

with open("database/components_database_link.rst", "w") as f:
    if excel_copied:
        # Provide both download and GitHub links - use correct path for Sphinx static files
        f.write(
            "All data used in the model are available in an Excel file that can be "
            "downloaded in\n"
        )
        f.write(":download:`Components database </_static/Components_database.xlsx>` ")
        f.write(f"or `view on GitHub <{excel_github_url}>`_.\n")
    else:
        # Fallback to GitHub link only
        f.write(
            f"All data used in the model are available in an Excel file that can be found in "
            f"`Components database <{excel_github_url}>`_.\n"
        )
