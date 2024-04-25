.. _workflow_create-data-templates:

Creating Input Data Templates
=====================================
After you have defined your system topology, you can now retrieve the templates for the input data required for the
optimization with this specific topology. For this, you call the ``create_input_data_folder_template`` method,
passing your input data folder path (which must be the same folder path as for the :ref:`Model Templates<workflow_create-model-templates>`).

.. automodule:: src.data_preprocessing.template_creation
    :members: create_input_data_folder_template
    :exclude-members:

This yields a CSV file ``NodeLocations`` and an input data folder for each investment period.

Note: all methods related to template creation can be found in the ``template_creation.py`` module in the
``data_preprocessing`` directory.

Your directory should now contain the following files:

- ``Topology.JSON``, in which you have :ref:`specified your system topology<workflow_define-topology>`.
- ``ConfigModel.JSON``, in which you can :ref:`define the model configuration settings<model_configuration>`
- ``NodeLocations.csv``, in which you can specify the geographical coordinates of your nodes.
- A folder for each investment period that you specified in the topology, containing:

    - ``Networks.JSON``, in which you specify the networks that are existing and that may be newly installed in the
      optimization. For each of the networks that you specify, an input data folder should be :ref:`added and filled <workflow_load-data>`
      in the corresponding folder (existing or new) in the network_topology folder.
    - A folder called ``network_data``, in which you :ref:`upload JSON files with network data <workflow_load-data>`
      for each network specified in the ``Networks.JSON``.
    - A folder called ``network_topology``, which itself contains:

        - A folder called ``existing``: containing the data templates that should be copied and :ref:`filled in<workflow_load-data>`
          for all existing network types.
        - A folder called ``new``: containing the data templates that should be copied and :ref:`filled in<workflow_load-data>`
          for all new network types.
    - A folder called ``node_data``, containing:

        - A folder for each node that you specified in the topology, containing:

            - ``Technologies.JSON``, in which you specify technologies that are existing and that may be newly installed
              in the optimization. For each of the technologies that you specify, an input data folder should be
              :ref:`added and filled <workflow_load-data>` in the technology_data folder.
            - ``CarbonCost.csv``, in which you :ref:`specify carbon prices and subsidies<workflow_load-data>`
              for each timestep.
            - ``ClimateData.csv``, in which you :ref:`specify climate data <workflow_load-data>`
              for each timestep.
            - A folder called ``carrier_data``, containing:

                - A ``carrier_name.csv`` file for each carrier, in which you can specify the balance constraints
                  (demand, import/export limits, etc.) for that carrier at the specific node in each timestep.
                - ``EnergybalanceOptions.JSON``, in which you specify for each carrier whether or not curtailment of
                  production is possible.
            - A folder called ``technology_data``, in which you :ref:`upload JSON files with technology data <workflow_load-data>`
              for each technology specified in the ``Technologies.JSON``.

If you have all the templates, you can now continue with :ref:`filling in the data <workflow_load-data>`
required to run the model.
