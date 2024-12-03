import nbformat
from nbconvert import PythonExporter
import re
import subprocess
from pathlib import Path
from glob import glob
import os
import sys


def test_jupyter_notebooks(request):
    """
    Makes sure that the jupyter notebooks run.

    - produces a new script file for each notebook contained in docs/source/case_studies
    - changes the directories to match the testing environment
    - runs all notebooks
    """

    notebooks_dir = Path(__file__).parent.parent / "docs/source/case_studies"
    notebook_files = glob(os.path.join(notebooks_dir, "*.ipynb"))
    testing_path = request.config.docu_notebook_folder_path

    for notebook in notebook_files:

        file_name = os.path.basename(notebook)
        file_name = file_name.replace(
            "ipynb",
            "py",
        )

        # Extract the code from the Jupyter notebook
        with open(notebook, "r", encoding="utf-8") as f:
            notebook_code = nbformat.read(f, as_version=4)

        # Convert notebook to Python script
        exporter = PythonExporter()
        code, _ = exporter.from_notebook_node(notebook_code)

        # Make sure only one period is used
        modified_code = re.sub(
            r'configuration\["optimization"\]\["typicaldays"\]\["N"\]\["value"\] = 30',
            'configuration["optimization"]["typicaldays"]["N"]["value"] = 1',
            code,
        )

        if (
            len(
                re.findall(
                    r'configuration\["optimization"\]\["typicaldays"\]\["N"\]\['
                    r'"value"\] = 1',
                    modified_code,
                )
            )
            == 0
        ):
            # Make sure only one period is tested
            modified_code = re.sub(
                r"read_data\(input_data_path\)",
                r"read_data(" "input_data_path, " "start_period=1, " "end_period=2)",
                modified_code,
            )

        # replace directories
        result_folder_path = str(request.config.result_folder_path)
        result_folder_path = result_folder_path.replace("\\t", "/t")

        modified_code = re.sub(r"\./userData", "./", modified_code)
        modified_code = re.sub(
            r"\./caseStudies/", "./tests/notebook_data/", modified_code
        )
        modified_code = re.sub(
            r'with open\(input_data_path / "ConfigModel.json", ' r'"w"\) as json_file:',
            'configuration["solveroptions"]["solver"]["value"] = "'
            + request.config.solver
            + '"\n'
            + 'configuration["reporting"]["save_summary_path"]["value"] = "'
            + result_folder_path
            + '"\n'
            + 'configuration["reporting"]["save_path"]["value"] = "'
            + result_folder_path
            + '"\n'
            + 'with open(input_data_path / "ConfigModel.json", "w") as json_file:',
            modified_code,
        )

        # save modified code
        with open(testing_path / file_name, "w", encoding="utf-8") as f:
            f.write(modified_code)

        try:
            result = subprocess.run(
                [sys.executable, testing_path / file_name],
                capture_output=True,
                text=True,
                check=True,
            )
            os.remove(testing_path / file_name)
        except subprocess.CalledProcessError as e:
            os.remove(testing_path / file_name)
            raise Exception(f"{notebook} failed to test")
