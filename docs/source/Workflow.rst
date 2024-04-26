.. _workflow:

Users Workflow
=====================================
This workflow documentation will guide you through all the steps that are required to prepare the model for your
application. In short, these steps are as follows:

- Set up a working directory of your case study for all input data preprocessing.
- Create the templates for the system topology and the model configuration.
- Define your system topology and model configuration.
- Create the folder structure and templates for the input data files.
- Load and define input data (e.g., weather data, technology performance, demand data, etc.).
- Construct and solve the model and, possibly, incorporate options to lower the complexity of the model.
- Check the model diagnostics.
- Obtain and interpret the optimization results.

An elaborate example of how to set up the model accordingly can be found :ref:`below <workflow_example-usage>`. To
understand what happens behind the scenes, please take a look :ref:`here<src-code>`. For a more detailed description of
each of the aforementioned steps, see the following pages:

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

..  _workflow_example-usage:

Example Usage
^^^^^^^^^^^^^^^^

To get started with your optimization, you first need to obtain the templates in which you can define the system
topology and the model configuration, as explained :ref:`here<workflow_create-model-templates>`. We specify a path to
our working directory and run the following code:


.. testcode::

    import src.data_preprocessing as dp

    path = "path_to_your_input_data_folder"

    dp.create_optimization_templates(path)

Next, you can define your system topology in the topology.JSON file. For this example, the topology is as follows:

.. code-block:: console

    "nodes": "onshore", "offshore"
    "carriers": "electricity", "heat", "gas"
    "investment_periods": "year1", "year2"
    "start_date": "2022-01-01 00:00",
    "end_date": "2022-12-31 23:00",
    "resolution": "1h",
    "investment_period_length": 1


Now, you can run the following command (the path is the same as before) to obtain the input data folder structure:

.. testcode::

    dp.create_input_data_folder_template(path)

Then, it is time to further specify the system (i.e., node locations, technologies, networks) and collect all input data
required for your optimization. We start by specifying the geographic locations in terms of longitude, latitude and
altitude, of our two nodes ("onshore" and "offshore") in the ``NodeLocations.csv``. Note: longitude and latitude should
be written in decimal degrees, and altitude in metres.

Next, for each investment period we specify the network types and their topology. For network types, the names should be
the same as the JSON file names in ``.\data\network_data``. For our example, let's say that there are no networks currently
in place, but an offshore electricity cable may be installed between our two nodes. In ``Networks.JSON``, this can be
specified as follows:

.. code-block:: console

    "existing": [],
    "new": ['electricityOffshore']

Now, we have to make sure that there is a topology for this network. For this, you go into the network_topology folder,
into the corresponding sub-folder (in our case, "new"), in which we create a new folder with the name of the network.
In this new folder "electricityOffshore", we copy the CSV files ``connection.csv``, ``distance.csv`` and
``size_max_arcs.csv`` from the "new" folder. In these, you specify the possibility of a network connection (0 = no;
1 = yes) between those nodes in a specific direction (``connection.csv``), the distance between the nodes
(``distance.csv``) and the maximum installed capacity of the network that can be installed between the nodes
(``size_max_arcs.csv``). For example:

.. code-block:: console

    Connection: Example - Unidirectional (offshore to onshore only)
                            onshore    offshore
        onshore             0          0
        offshore            1          0

    Connection: Example - Bidirectional
                            onshore    offshore
        onshore             0          1
        offshore            1          0

    Distance [km]:
                            onshore    offshore
        onshore             0          100
        offshore            100        0

    Maximum size [MW]:
                            onshore    offshore
        onshore             0          20
        offshore            20         0

Note: all these files are matrices with the columns and the rows being the nodes you specified in the initial topology.
You should read it as "from row, to column". Networks can only exist between nodes, so the diagonal of this matrix
always consists of 0s.

In order to read in the required data for our "electricityOffshore" network, the JSON file of that network type has to
be copied from the model repository (``.\data\network_data``) into the "network_data" folder in your input data folder.
You can do this manually, but if you have many different network types in your system, you can do it by running the
following code:

.. testcode::

    dp.copy_network_data(path, "path_to_model_repository")

Again, the path is the same as before, and it will automatically copy all networks that are specified in ``Networks.JSON``.
In the network's specific JSON file, you can specify the cost data, lifetime, losses, etc. Note: if you want to have the
same network data for all investment periods, it is quicker to adapt the data in the JSON in the model repository before
copying to your own folder.

Next, for each node we specify the technology types, their input data, the carrier data, climate data and carbon costs.

Climate data can easily be retrieved for all nodes and investment periods at once, by running the following command, as
explained :ref:`here<workflow_load-data>`:

.. testcode::

    dp.load_climate_data_from_api(path)

Note: the only exception to this is the hydro_inflow data. If your system contains any technologies based on the
technology type "Hydro_Open" (see :ref:`here<technologies>`), you should specify this manually. For this, replace the
"TECHNOLOGYNAME" in the column name with the technology in your system, e.g., "PumpedHydro_Open" and load a profile for
water flow into the reservoir.

For technologies, same as for networks, you specify the existing and new technologies in ``Technologies.JSON``, using
the same names as the JSON file names in ``.\data\technology_data``. For our example, we allow for the installment of a
battery, solar PV and a natural gas-fired furnace in the onshore node. Besides, we add two existing technologies and
their respective sizes, as follows:

.. code-block:: console

    "existing": ['WindTurbine_Onshore_1500': 2, 'Photovoltaic': 2.4],
    "new": ['Storage_Battery', 'Photovoltaic', 'Furnace_NG']

Note: For wind turbines, the capacity of one turbine is specified in the name (1500 W), and the size is an integer. Here,
we thus have two 1.5MW wind turbines installed (totalling to 3MW), and 2.4MW of solar PV.

In order to read in the required data for our four different technologies, the JSON files thereof have to be copied from
the model repository (``.\data\technology_data``) into the "technology_data" folder in your input data folder. This can
be done by running the following code:

.. testcode::

    dp.copy_technology_data(path, "path_to_model_repository")

Again, the path is the same as before, and it will automatically copy all technologies that are specified in
``Technologies.JSON``. In the technologies' specific JSON files, you can specify the cost data, lifetime, losses, etc.
Note: if you want to have the same technology data for all investment periods and nodes, it is quicker to adapt the data
in the JSON in the model repository before copying to your own folder. For existing technologies, you can decide if they
can be decommissioned and at what cost in the respective technology's JSON file as well.

In ``CarbonCost.csv``, you can specify carbon costs and subsidies for carbon reduction. In the "carrier_data" folder,
the demand, import/export limits, prices and emission factors, and generic production can be specified per carrier for
that node and investment period. In the same folder, you can specify if curtailment of this general production is
possible in ``EnergybalanceOptions.JSON`` (0 = not possible; 1 = possible). For all this data, you can either set a
fixed value for all time steps, or you can manually upload a profile for that parameter over time. For the former option,
you can run the following piece of code, in this example to set the onshore electricity demand to 10MW for all time steps:

.. testcode::

    dp.fill_carrier_data(path, 10, columns='Demand', carriers='electricity', nodes='onshore', investment_periods=['year1', 'year2'])

Note that data for carriers and nodes not specified will be set to zero.

Now that you have completely set up your system and defined all input data, you can set the model configuration as you
wish for the optimization (this also includes specifying the path for your outputs) in ``ConfigModel.JSON``. Now, you
can call an instance of the EnergyHub class, read in all the data, construct the model and solve the model as follows:

.. testcode::

    modelname = EnergyHub()
    modelname.read_data(path, start_period=None, end_period=None)
    modelname.construct_model()
    modelname.quick_solve()

Note: the start and end period are the time steps you wish to solve the model for (if you do not want to solve over the
complete time horizon as specified in the topology), in which 0 is the first time step in your time horizon.

If the optimization with the full resolution of the data takes too long or is too large, the input data can be clustered
into a number of typical days. 20-50 typical days hereby usually gives a close enough optimization result. Storage
technologies work with the full resolution, so that required seasonal storage is accounted for. Below is an example of
how to use the k-means algorithm (by setting N=50 in ``ConfigModel.JSON``):

.. testcode:: console

        "typicaldays": {
            "N": {
                "description": "Determines number of typical days (0 = off).",
                "value": 50
            },
            "method": {
                "description": "Determine method used for modeling technologies with typical days.",
                "options": [
                    2
                ],
                "value": 2
            }

.. testcode::

    modelname.read_data(path, start_period=None, end_period=None)
    modelname.construct_model()
    modelname.quick_solve()

If your model solved to optimality, you can continue with checking out your results. If your model did not solve to
optimality, or took longer than expected for the complexity of your system, you can take a look at the
:ref:`model diagnostics<model_diagnostics>`.

For a simple means to get a grip on your results, you can :ref:`start the streamlit visualisation<results_visualization>`.
From there, you can also download CSV files with your result data. Alternatively, you can :ref:`export<export_excel>'
datasets from your .h5 files to Excel.