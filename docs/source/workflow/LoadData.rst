.. _workflow_load-data:

Load Input Data
=====================================

Now that you have the :ref:`correct folder structure and templates for the data <workflow_create-data-templates>`,
you can start to actually fill the templates and folders with data. There are four options for loading data:

#. :ref:`From API<load-data_from-api>`: for climate data.
#. :ref:`From model<load-data_from-model>`: for technologies and networks, you can copy the required JSON files into the technology_data and network_data folders, respectively.
#. :ref:`Specifying a fixed value<load-data_fixed-value>`: for carriers and carbon costs, if the values do not change over time.
#. :ref:`Specifying a profile<load-data_profile>`: for carriers and carbon costs, if the values do change over time.

A detailed description of these different methods can be found in the sections below. Note: all methods related to
loading in data can be found in the ``data_loading.py`` module in the ``data_preprocessing`` directory.

..  _load-data_from-api:

From API
^^^^^^^^^^^^^^^^
For importing climate data from the JRC PVGIS database, the method ``load_climate_data_from_api`` can be used, passing
your :ref:`input data folder path<workflow_create-data-templates>`. This imports the climate data for each node,
accounting for the location of the nodes as specified in ``NodeLocations.csv``. If no location is specified, it takes the
default coordinates (52.5, 5.5) with an altitude of 10m.

.. automodule:: src.data_preprocessing.data_loading
    :members: load_climate_data_from_api
    :exclude-members:

..  _load-data_from-model:

From model
^^^^^^^^^^^^^^^^
For the technologies and networks, you can copy the JSON files in automatically using the ``copy_technology_data`` and
``copy_network_data`` methods below. Note: the method automatically checks which technologies and networks it has to copy
from the model repository by reading in the ``Technology.JSON`` and ``Network.JSON`` files, respectively. Thus, make sure
to use the naming conventions as in the JSON files in the model repository.


.. automodule:: src.data_preprocessing.data_loading
    :members: copy_technology_data, copy_network_data
    :exclude-members:

..  _load-data_fixed-value:

Specifying a fixed value
^^^^^^^^^^^^^^^^^^^^^^^^^

For carrier data, think of demand or import/export limits of a specific carrier at a specific node, you can use the
``fill_carrier_data`` method if your value does not vary over time.

.. automodule:: src.data_preprocessing.data_loading
    :members: fill_carrier_data
    :exclude-members:

..  _load-data_profile:

Specifying a profile
^^^^^^^^^^^^^^^^^^^^^
If the carrier data values do change over time (e.g., you want to have a demand profile resembling actual load) you have
to manually specify the profiles in the carrier csv files in your input data folder. For example, you can copy a demand
profile from a national database (if your nodes are countries) into the correct column in the csv.



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