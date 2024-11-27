.. _developers-guide:

=====================================
Developer Instruction
=====================================
This page contains general instructions for the developers that are working on the PyHub.

Reporting Issues
-----------------------
If you encounter an issue or a missing feature, you can report it on `github
<https://github.com/UU-ER/AdOpT-NET0/issues>`_. Please
attach a detailed description and a tag (bug, enhancement,...)

Setting Up the Development Environment
-----------------------------------------
To develop actively on the tool, you can follow the subsequent steps (feel free to
chose a different IDE or environment management):

- Make sure you have Python >=3.12 installed on your computer
- Clone the repository to you drive
- Create a virtual environment for this project. You can do this using PyCharm:
    - Open the project in PyCharm
    - Go to File -> Settings -> <project_name> -> Python Interpreter
    - Select Add Local Interpreter from the Add Interpreter
    - In the left-hand pane of the Add Python Interpreter dialog, select Virtualenv Environment
    - Add a Base Interpreter
- Install all required packages to your virtual environment by using poetry. In PyCharm
you can do this with:
    - Move to the terminal in PyCharm, it is located on the bottom. If the virtual environment was correctly installed, you should see a (venv) in front of the path
    - Execute the command ``pip install poetry``
    - Execute the command ``poetry install`` to install all required packages
- Now, you can run main.py after specifying the right paths. In PyCharm, you can do
  this:
    - Click on Edit configuration, in the upper right corner of the screen
    - Click Add new...
    - Name your configuration as you like (e.g. Run main)
    - Select Python as an interpreter and click ok
    - You can run the file.
- If you are planning to contribute to the main version, it is handy to also install
  a pre-commit hook. You can do this by running ``pre-commit install`` from the
  terminal.


Codebase Overview
-----------------------------------------
The codebase is divided into the following parts:

- Data preprocessing: defines the input data, has helper functions to change data easily
- Data management: reads and processes input data before a model is constructed
- Model construction: module containing functions to construct individual parts of
  the model.
- Components: module containing all technology and network models that are available
- Result management: contains functions to export the optimization results
- Test: test suit

Development Workflow
-----------------------------------------
The main branch contains the working and published version. The development branch
contains a version being prepared for the next release. New features or bug fixes are
developed on a separate branch and then merged via a pull request to the development
branch. At each pull request to the development branch, the github CI checks if all
tests succeed and if the code is correctly formatted. Each pull request needs at
least one approved review.

Before a pull request, tests can be locally run with ``pytest`` in
the terminal.

Coding conventions
-----------------------------------------

Naming
^^^^^^^^^^

To keep the code consistent and clear for other developers, try to use the coding
conventions that are described below.

For the Pyomo classes we use:

+-------------+--------------+
| Type        | Code         |
+=============+==============+
| Objective   | objective... |
+-------------+--------------+
| Constraint  | const...     |
+-------------+--------------+
| Piecewise   | const...     |
+-------------+--------------+
| Set         | set...       |
+-------------+--------------+
| Block       | b...         |
+-------------+--------------+
| Var         | var...       |
+-------------+--------------+
| Param       | para...      |
+-------------+--------------+
| Disjunct    | dis...       |
+-------------+--------------+
| Disjunction | disjunction..|
+-------------+--------------+
| rule        | init...      |
+-------------+--------------+

Other names that are regularly used in AdOpT-NET0 are:

+-------------+--------------+
| Type        | Code         |
+=============+==============+
| Timestep    | t...         |
+-------------+--------------+
| Carrier     | car...       |
+-------------+--------------+
| Node        | node...      |
+-------------+--------------+
| Network     | netw...      |
+-------------+--------------+
| Technology  | tec...       |
+-------------+--------------+
| Consumption | cons...      |
+-------------+--------------+
| Input       | input...     |
+-------------+--------------+
| Output      | output...    |
+-------------+--------------+

Documentation
^^^^^^^^^^^^^^^^
We require all classes and functions to have a docstring and type annotations (where
possible/convenient). The following should be noted:

- The doctring starts with a single line describing in brief the class/method
- We use the ``reStructuredText`` format for parameters and returns
- Type annotations to functions should be added for standard types (e.g. str, int, pd.
  DataFrame,...) but not for very specific types (e.g. ModelHub, pyo.Constraint,...).
  The same holds for return types
- Where required, include the documentation in the sqhinx build of the documentation
  that is published alongside this package.

Additionally, refer to the following guides on documentation:

* `PEP 8 - Style Guide for Python Code <https://peps.python.org/pep-0008/>`_
* `PEP 257 <https://peps.python.org/pep-0257/>`_ (also explained well `here <https://pandas.pydata.org/docs/development/contributing_docstring.html>`_)
* `Shinx Cheat Sheets <https://sphinx-tutorial.readthedocs.io/cheatsheet/>`_

As such, the documentation of a function can look like this:

.. testcode::

    def set_capex_model(config: dict, economics) -> int:
        """
        Sets the capex model of a technology

        Takes either the global capex model or the model defined in respective technology
        :param dict config: dict containing model information
        :param economics: Economics class
        :return: CAPEX model
        :rtype: int
        """
        capex_model = economics.capex_model
        if capex_model != 4:
            if config["economic"]["global_simple_capex_model"]["value"]:
                capex_model = 1

        return capex_model

