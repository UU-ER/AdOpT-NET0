Data Preprocessing
=====================================
In order to prepare the model for your application, you will have to set up a working directory for all data
preprocessing and take the following steps:

- Create the templates for the system topology and the model configuration.
- Define your system topology.
- Create the templates for the input data files.
- Load and define input data.

An example of how to set up the model accordingly can be found :ref:`below <data-preprocessing_example-usage>`. For a more
detailed description of each of the aforementioned steps, see the following pages:

.. toctree::
    :maxdepth: 1

    data_preprocessing/CreateModelTemplates
    data_preprocessing/DefineTopology
    data_preprocessing/CreateDataTemplates
    data_preprocessing/LoadData

..  _data-preprocessing_example-usage:

Example Usage
^^^^^^^^^^^^^^^^
Fist, we create an empty topology and fill it with a system design. Hereby note:

- Node names can be chosen freely
- Carrier names need to follow the same naming convention as in the technology files
- Technology names need to be the same as the JSON file names in ``.\data\technology_data``. You can also use a
  different directory to read in the technology data.

.. testcode::

    import src.data_management as dm

    topology = dm.SystemTopology()
    topology.define_time_horizon(year=2001,start_date='01-01 00:00', end_date='01-04 23:00', resolution=1)
    topology.define_carriers(['electricity', 'heat'])
    topology.define_nodes(['onshore', 'offshore'])
    topology.define_new_technologies('onshore', ['Storage_Battery', 'Photovoltaic', 'Furnace_NG'])

It is also possible to add a technology that exists already at a certain size to a node. Note, that you need to
pass the technology as a dictonary with the respective size instead of a simple list. You can specify if these technologies
can be decommissioned and at what cost in the respective json data file.
Similarly, it is possible to add an existing network to the model. See the
:ref:`System Topology Documentation <data-management-system_topology>` for more details.


.. testcode::

    topology.define_existing_technologies('onshore', {'WindTurbine_Onshore_1500': 2, 'Photovoltaic': 2.4})

Let's create an electricity network connecting the onshore and offshore node:

.. testcode::

    distance = dm.create_empty_network_matrix(topology.nodes)
    distance.at['onshore', 'offshore'] = 100
    distance.at['offshore', 'onshore'] = 100

    connection = dm.create_empty_network_matrix(topology.nodes)
    connection.at['onshore', 'offshore'] = 1
    connection.at['offshore', 'onshore'] = 1
    topology.define_new_network('electricitySimple', distance=distance, connections=connection)

The topology has now been defined. We can initialize an instance of the ``src.data_management.handle_input_data.DataHandle``\
class and read in all input data. Note that data for carriers and nodes not specified will be\
set to zero.

.. testcode::

    # Initialize instance of DataHandle
    data = dm.DataHandle(topology)

    # CLIMATE DATA
    lat = 52
    lon = 5.16
    data.read_climate_data_from_api('onshore', lon, lat,save_path='.\data\climate_data_onshore.txt')
    lat = 52.2
    lon = 4.4
    data.read_climate_data_from_api('offshore', lon, lat,save_path='.\data\climate_data_offshore.txt')

    # DEMAND
    electricity_demand = np.ones(len(topology['timesteps'])) * 10
    data.read_demand_data('onshore', 'electricity', electricity_demand)

    # PRINT DATA
    data.pprint()

Printed output:

.. code-block:: console

    ----- NODE onshore -----
	 demand
		                     Mean       Min       Max
		electricity          10.0      10.0      10.0
	 import_prices
		                     Mean       Min       Max
		electricity           0.0         0         0
    ...
	 emission_factors
		                     Mean       Min       Max
		electricity           0.0         0         0

If the optimization with the full resolution of the data takes too long or is too large, the input data
can be clustered into a number of typical days. 20-50 typical days hereby usually gives a close enough
optimization result. Storage technologies work with the full resolution, so that required seasonal storage
is accounted for. Below is an example of how to use the k-means algorithm:

.. testcode::

    # Load a DataHandle instance
    data = dm.load_data_handle(r'./userdata/systemData.p')

    # Load ModelConfiguration
    configuration = ModelConfiguration()

    # Specify a number of typical days
    configuration.optimization.typicaldays.N = 50

    # Solve
    energyhub = EnergyHub(data, configuration)
    energyhub.quick_solve()