.. _workflow_load-data:

Define Input Data
=====================================

Now that you have the :ref:`correct folder structure and templates for the data <workflow_create-data-templates>`,
you can start to fill the templates and folders with data. First, you need to further
define your energy system by following the steps below. Note that the package
includes predefined technologies and networks with respective performance and cost
parameters, but the advanced user is free to define new technologies or adapt the
templates provided.

#. The geographical coordinates of your nodes in ``NodeLocations.csv`` in terms of
   longitude, latitude and altitude. Note: longitude and latitude should
   be written in decimal degrees, and altitude in metres. Climate data is loaded from
   this data. Additionally it is used to calculate the position of the sun for PV
   modelling (using pvlib). Note that the distance between nodes is not based on the
   provided locations.

   .. testcode::

     # Define node locations (here an exemplary location in the Netherlands)
     node_locations = pd.read_csv(input_data_path / "NodeLocations.csv", sep=";", index_col=0)
     node_locations.loc["node1", "lon"] = 5.5
     node_locations.loc["node1", "lat"] = 52.5
     node_locations.loc["node1", "alt"] = 10
     node_locations.to_csv(input_data_path / "NodeLocations.csv", sep=";")

#. The networks in ``Networks.JSON`` for each investment period, where we distinguish between new (to be installed)
   networks and existing networks, using the network names as in ``.\data\network_data`` (a list of these can be found
   :ref:`here <network_list>`) or by:

   .. testcode::

     adopt.show_available_networks()

   Note that all networks are based on the :ref:`Network Class <networks>`. Specify
   the networks in the json file like this:

   .. code-block:: console

     "existing": [],
     "new": ["electricityOffshore"]

   - Then, for each of the networks that you specify, an input data folder with that
     network name should be added  in the corresponding folder ("existing" or
     "new") in the network_topology folder. To these folders, you should
     copy the CSV files ``connection.csv`` and ``distance.csv``. You then define the
     topology for all of your networks by filling in these files (as
     illustrated in an elaborate example :ref:`here<workflow_example-usage>`) in
     their respective folders. Note that the distance needs to be added manually,
     even if the node locations were specified previously.
     It is also possible to specify a maximum size for each
     arc individually by creating another file called ``size_max_arcs.csv`` having the
     same layout as ``connection.csv`` or ``distance.csv``.  Note that for new networks
     the ``size_max_arcs.csv`` contains an upper limit, the actual size is determined
     by the optimization.

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

     Note: all these files are matrices with the columns and the rows being the nodes
     you specified in the initial topology. You should read it as "from row, to
     column". Networks can only exist between nodes, so the diagonal of this matrix
     always consists of 0s.

   - In order to read in the required data for our "electricityOffshore" network, the JSON file of that network type has to
     be either copied into the "network_data" folder in your input data folder or
     made from scratch (based on the template). You can do this manually, but if you
     have many different network types in your system, you can do it by running the
     following code (see also :ref:`here<load-data_from-model>`). After copying the
     file you can also change the performance and cost parameters provided.

     .. testcode::

         adopt.copy_network_data(input_data_path)

#. The technologies in ``Technologies.JSON`` for each investment period and each
   node, where we distinguish between new (to be installed) technologies and
   existing technologies. A list of these can be found :ref:`here
   <technologies_list>`) or by running:

   .. testcode::

       adopt.show_available_technologies()

   Specify the technologies used in this model e.g. as:

   .. code-block:: console

       "existing": {"WindTurbine_Onshore_1500": 2, "Photovoltaic": 2.4},
       "new": ["Storage_Battery", "Photovoltaic", "Furnace_NG"]

   Note: For wind turbines, the capacity of one turbine is specified in the name
   (1500 W), and the size is an integer. Here, we thus have two 1.5MW wind turbines
   installed (totalling to 3MW), and 2.4MW of solar PV.

    Similar to the network data, we can now copy the required technology data files
    by running (see also :ref:`here<load-data_from-model>`). After copying the
    files you can also change the performance and cost parameters provided.

    .. testcode::

       adopt.copy_technology_data(input_data_path)

#. For the carriers, whether or not curtailment of generic production is possible in
   ``EnergybalanceOptions.JSON`` (0 = not possible; 1 = possible).

#. After the complete system topology and system characteristics are finalised, time
   dependent data can be loaded into the input data folder. This remaining data covers:

   - Carbon costs: Prices of carbon emissions and/or subsidies for emission reductions.
     These are defined for each investment period and node.
   - Climate data: global horizontal irradiance (ghi), direct normal irradiance (dni),
     diffuse horizontal irradiance (dhi), air temperature, relative humidity, inflow
     of water, wind speed at 10 metres. The data is defined for each investment
     period and node based on the geographical location.
   - Carrier data: Data on demand, import/export limits/prices/emission factors, and
     generic production for each carrier at each node in each investment period.

   All this data is time-dependent, so you need to specify data for all time steps of
   your model run.

   All data can be simply changed directly in the csv file. For example, you can copy a
   demand profile from a national database (if your nodes are countries) into the
   correct column in the csv. Additionally, we provide a couple of functions to make
   defining the time dependent data more convenient:

   - For climate data, you can use the API to a :ref:`JRC database for onshore
     locations in Europe<load-data_from-api>`. E.g.:

     .. testcode::

         adopt.load_climate_data_from_api(input_data_path)

   - For all other time series, you can :ref:`specify a fixed
     value<load-data_fixed-value>`, if the values do not change over time. E.g.:

     .. testcode::

         adopt.fill_carrier_data(path, value_or_data=10, columns=["Demand"], carriers=["electricity"], nodes=["onshore"], investment_periods=['period1'])


.. _load-data_from-api:

Climate data from API
^^^^^^^^^^^^^^^^^^^^^^^^
For importing climate data from the `JRC PVGIS database
<https://joint-research-centre.ec.europa
.eu/photovoltaic-geographical-information-system-pvgis_en>`_, the method
:func:`load_climate_data_from_api` can be used, passing
your :ref:`input data folder path<workflow_create-data-templates>`. This imports the climate data for each node,
accounting for the location of the nodes as specified in ``NodeLocations.csv``. If no location is specified, it takes the
default coordinates (52.5, 5.5) with an altitude of 10m.

.. automodule:: adopt_net0.data_preprocessing.data_loading
    :members: load_climate_data_from_api
    :exclude-members:

NB: this imports all climate data, except for the hydro inflow. Hydro inflow needs to be specified for any technologies
based on the technology type "OpenHydro" (see :ref:`here<technologies>`). For this, replace the "TECHNOLOGYNAME" in the
column name with the technology in your system, e.g., "PumpedHydro_Open" and :ref:`load a profile<load-data_profile>`
for water flow into the reservoir.

.. _load-data_fixed-value:

Specifying a fixed value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For carrier data, you can use the :func:`fill_carrier_data` method if your value does not vary over time.

.. automodule:: adopt_net0.data_preprocessing.data_loading
    :members: fill_carrier_data
    :exclude-members:

.. _load-data_from-model:

Copy technology and network data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For the technologies and networks, you can copy the JSON files automatically using the :func:`copy_technology_data` and
:func:`copy_network_data` methods below. Note: the method automatically checks which technologies and networks it has to copy
from the model repository by reading in the ``Technology.JSON`` and ``Network.JSON`` files, respectively. Thus, make sure
to use the naming conventions as in the JSON files in the model repository.

.. automodule:: adopt_net0.data_preprocessing.data_loading
    :members: copy_technology_data, copy_network_data
    :exclude-members:

