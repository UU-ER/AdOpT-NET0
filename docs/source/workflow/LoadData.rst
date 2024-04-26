.. _workflow_load-data:

Load Input Data
=====================================

Now that you have the :ref:`correct folder structure and templates for the data <workflow_create-data-templates>`,
you can start to fill the templates and folders with data. First, you need to further define your energy
system by setting:

#. The geographical coordinates of your nodes in ``NodeLocations.csv``.

#. The networks in ``Networks.JSON`` for each investment period, where we distinguish between new (to be installed)
   networks and existing networks, using the network names as in ``.\data\network_data``.
    - Then, for each of the networks that you specify, an input data folder with that network name should be added
      in the corresponding folder ("existing" or "new") in the network_topology folder. To these folders, you should
      copy the CSV files ``connection.csv``, ``distance.csv`` and ``size_max_arcs.csv`` from the corresponding folder
      ("existing" or "new"). You then define the topology for all of your networks by filling in these files (as
      illustrated in an elaborate example :ref:`here`<workflow_example-usage>`) in their respective folders. Note that
      for new networks the ``size_max_arcs.csv`` contains an upper limit, the actual size is determined by the optimization.
    - Finally, you can adjust the network data in the json files of the specific network types, either (1) in the model
      repository, before :ref:`copying these<load-data_from-model>` to the network_data folder in your input data folder,
      or (2) in the network_data folder in your input data folder, after :ref:`copying them from the model repository <load-data_from-model>`.
      Note that option (1) is more efficient if you want the network data to be the same at each investment period,
      while option (2) is more convenient if you want it to change per investment period.

#. The technologies in ``Technologies.JSON`` for each investment period and each node, where we distinguish between new
   (to be installed) technologies and existing technologies using the technology names as in ``.\data\technology_data``.
    - Then, you can adjust the technology data in the json files of the specific technology types, either (1) in the model
      repository, before :ref:`copying these<load-data_from-model>` to the technology_data folder in your input data folder,
      or (2) in the technology_data folder in your input data folder, after :ref:`copying them from the model repository <load-data_from-model>`.
      Note that option (1) is more efficient if you want the technology data to be the same at each node and in each investment period,
      while option (2) is more convenient if you want it to change per node and/or investment period.

#. For the carriers, whether or not curtailment of general production is possible in ``EnergybalanceOptions.JSON``.

After the complete system topology and system characteristics are finalised, the final data can be loaded into the input
data folder. This remaining data covers:

- Carbon costs: Prices of carbon emissions and/or subsidies for emission reductions. These are defined for each investment
  period and node.
- Climate data: global horizontal irradiance (ghi), direct normal irradiance (dni), diffuse horizontal irradiance (dhi),
  air temperature, relative humidity, inflow of water, wind speed at 10 metres. The data is defined for each investment
  period and node based on the geographical location.
- Carrier data: Data on demand, import/export limits/prices/emission factors, and generic production for each carrier at
  each node in each investment period.

All this data is time-dependent, so you need to specify data for all time steps of your model run.

There are four options for loading data:

#. :ref:`From API<load-data_from-api>`: for climate data.
#. :ref:`From model<load-data_from-model>`: for technologies and networks, you can copy the required JSON files into the technology_data and network_data folders, respectively.
#. :ref:`Specifying a fixed value<load-data_fixed-value>`: for carriers and carbon costs, if the values do not change over time.
#. :ref:`Specifying a profile<load-data_profile>`: for carriers and carbon costs, if the values do change over time.

A detailed description of these different methods can be found in the sections below. Note: all methods related to
loading in data can be found in the ``data_loading.py`` module in the ``data_preprocessing`` directory.

..  _load-data_from-api:

From API
^^^^^^^^^^^^^^^^
For importing climate data from the JRC PVGIS database, the method :func:`load_climate_data_from_api` can be used, passing
your :ref:`input data folder path<workflow_create-data-templates>`. This imports the climate data for each node,
accounting for the location of the nodes as specified in ``NodeLocations.csv``. If no location is specified, it takes the
default coordinates (52.5, 5.5) with an altitude of 10m.

.. automodule:: src.data_preprocessing.data_loading
    :members: load_climate_data_from_api
    :exclude-members:

NB: this imports all climate data, except for the hydro inflow. Hydro inflow needs to be specified for any technologies
based on the technology type "OpenHydro" (see :ref:`here<technologies>`). For this, replace the "TECHNOLOGYNAME" in the
column name with the technology in your system, e.g., "PumpedHydro_Open" and :ref:`load a profile<load-data_profile>`
for water flow into the reservoir.

..  _load-data_from-model:

From model
^^^^^^^^^^^^^^^^
For the technologies and networks, you can copy the JSON files in automatically using the :func:`copy_technology_data` and
:func:`copy_network_data` methods below. Note: the method automatically checks which technologies and networks it has to copy
from the model repository by reading in the ``Technology.JSON`` and ``Network.JSON`` files, respectively. Thus, make sure
to use the naming conventions as in the JSON files in the model repository.


.. automodule:: src.data_preprocessing.data_loading
    :members: copy_technology_data, copy_network_data
    :exclude-members:

..  _load-data_fixed-value:

Specifying a fixed value
^^^^^^^^^^^^^^^^^^^^^^^^^

For carrier data, you can use the :func:`fill_carrier_data` method if your value does not vary over time.

.. automodule:: src.data_preprocessing.data_loading
    :members: fill_carrier_data
    :exclude-members:

..  _load-data_profile:

Specifying a profile
^^^^^^^^^^^^^^^^^^^^^
If the carrier data values do change over time (e.g., you want to have a demand profile resembling actual load) you have
to manually specify the profiles in the carrier csv files in your input data folder. For example, you can copy a demand
profile from a national database (if your nodes are countries) into the correct column in the csv.
