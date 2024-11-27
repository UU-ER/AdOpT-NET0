.. _workflow:

=====================================
Get Started
=====================================

For a more detailed description of each of the steps mentioned below, see the
following pages:

.. toctree::
    :maxdepth: 1

    workflow/CreateModelTemplates
    workflow/DefineTopology
    workflow/CreateDataTemplates
    workflow/LoadData
    workflow/DefineModelConfiguration
    workflow/SolveModel
    workflow/CheckModelDiagnostics
    workflow/ManageResults

In addition to the workflow documentation below, we provide three documented :ref:`case studies <case-studies>`
which can be useful to see how AdOpT-NET0 can be used to implement your optimization problem.

In short
----------------------
This workflow documentation will guide you through all the steps that are required to
prepare an exemplary model. Here, we optimize an energy system with a flat
electricity demand over a full year that can be served by imports from the grid, or
by installing photovoltaic panels and a battery.

The modelling steps are as follows:

- Create an *empty working directory* for all input data. The subsequent functions
  will create the data structure you need for the model

- :ref:`Create the templates<workflow_create-model-templates>` for the system
  topology and the model configuration.

    .. testcode::

        import adopt_net0 as adopt
        import json
        from pathlib import Path

        input_data_path = Path("path_to_your_input_data_folder")
        adopt.create_optimization_templates(input_data_path)

- :ref:`Define your system topology<workflow_define-topology>` and the
  :ref:`model configuration<workflow_model-configuration>`.

    .. testcode::

        with open(input_data_path / "Topology.json", "r") as json_file:
            topology = json.load(json_file)
        # Nodes
        topology["nodes"] = ["node1"]
        # Carriers:
        topology["carriers"] = ["electricity"]
        # Investment periods:
        topology["investment_periods"] = ["period1"]
        # Save json template
        with open(input_data_path / "Topology.json", "w") as json_file:
            json.dump(topology, json_file, indent=4)

- :ref:`Create the folder structure<workflow_create-data-templates>` and templates
  for the input data files.

    .. testcode::

        adopt.create_input_data_folder_template(input_data_path)

- :ref:`Load and define input data<workflow_load-data>` (e.g., weather data,
  technology performance, demand data, etc.).

    .. testcode::

        # Define new technologies
        with open(input_data_path / "period1" / "node_data" / "node1" / "Technologies.json", "r") as json_file:
            technologies = json.load(json_file)
        technologies["new"] = ["Photovoltaic", "Storage_Battery"]
        with open(input_data_path / "period1" / "node_data" / "node1" / "Technologies.json", "w") as json_file:
            json.dump(technologies, json_file, indent=4)

        # Copy over technology files
        adopt.copy_technology_data(input_data_path)

        # Define climate data
        adopt.load_climate_data_from_api(input_data_path)

        # Define demand
        adopt.fill_carrier_data(input_data_path, value_or_data=0.01, columns=['Demand'], carriers=['electricity'], nodes=['node1'])
        adopt.fill_carrier_data(input_data_path, value_or_data=100, columns=['Import price'], carriers=['electricity'], nodes=['node1'])
        adopt.fill_carrier_data(input_data_path, value_or_data=1, columns=['Import limit'], carriers=['electricity'], nodes=['node1'])

- :ref:`Define model configuration<workflow_model-configuration>` if you want to
  change something from the defaults. Make sure that the
  result folder path in ``reporting/save_path`` refers to an existing folder

- :ref:`Construct and solve the model<workflow_solve-model>` and, possibly,
  incorporate options to lower the complexity of the model.

    .. testcode::

        m = adopt.ModelHub()
        m.read_data(input_data_path, start_period=None, end_period=None)
        m.quick_solve()

- If something unexpected happens: check the :ref:`model
  diagnostics<model_diagnostics>`.
- :ref:`Obtain and interpret the optimization results<workflow_manage-results>`.
  Therefore, you can upload the H5 file saved on the visualization platform provided:
  https://resultvisualization.streamlit.app/.

An elaborate example of how to set up the model accordingly can be found below. To
understand what happens behind the scenes, please take a look at
the :ref:`Source Code Documentation<src-code>`.

