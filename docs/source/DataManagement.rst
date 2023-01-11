Data Management
=====================================

Input Data Management
-----------------------
Input data management works with the class ``src.data_management.handle_input_data.DataHandle`` class. It lets you import
and manage input data. The module ``src.data_management.create_templates`` contains functions
creating empty templates for specifiying the system topology and network. The module
``src.data_management.import_functions`` contains functions importing data from external
sources.

Input data can be clustered to reduce the spatial resolution. This can be done using a k-means algorithm
that is provided in the subclass ``src.data_management.handle_input_data.ClusteredDataHandle``. See also below
for example usage.

.. toctree::
    :maxdepth: 1

    data_management/CreateTemplates
    data_management/DataHandle
    data_management/ImportFunctions


Example Usage
^^^^^^^^^^^^^^^^
Fist, we create an empty topology and fill it with a system design. Hereby note:

- Node names can be chosen freely
- Carrier names need to follow the same naming convention as in the technology files
- Technology names need to be the same as the JSON file names in ``.\data\technology_data``

.. testcode::

    from src.data_management.create_templates import create_empty_topology

    modeled_year = 2001

    topology = {}
    topology['timesteps'] = pd.date_range(start=str(modeled_year)+'-01-01 00:00', end=str(modeled_year)+'-12-31 23:00', freq='1h')

    topology['timestep_length_h'] = 1
    topology['carriers'] = ['electricity']
    topology['nodes'] = ['onshore', 'offshore']
    topology['technologies'] = {}
    topology['technologies']['onshore'] = ['battery', 'PV']
    topology['technologies']['offshore'] = []

Let's create an electricity network connecting the onshore and offshore node:

.. testcode::

    from src.data_management.create_templates import create_empty_network_data

    topology['networks'] = {}
    topology['networks']['electricitySimple'] = {}
    network_data = dm.create_empty_network_data(topology['nodes'])
    network_data['distance'].at['onshore', 'offshore'] = 100
    network_data['distance'].at['offshore', 'onshore'] = 100
    network_data['connection'].at['onshore', 'offshore'] = 1
    network_data['connection'].at['offshore', 'onshore'] = 1
    topology['networks']['electricitySimple'] = network_data

The topology has now been defined. We can initialize an instance of the ``src.data_management.handle_input_data.DataHandle``\
class and read in all input data. Note that data for carriers and nodes not specified will be\
set to zero.

.. testcode::

    # Initialize instance of DataHandle
    data = DataHandle(topology)

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

    # Initialize an instance of the ClusteredDataHandle
    clustered_data = dm.ClusteredDataHandle()

    # Specify number of typical days
    nr_days_cluster = 20

    # Perform clustering
    clustered_data.cluster_data(data, nr_days_cluster)


Result Data Management
-----------------------
Result data management works with the class ``src.data_management.handle_optimization_results.ResultsHandle``
class. It lets you export results to dataframes and to excel.

Example Usage
^^^^^^^^^^^^^^^^
To export data from the pyomo model to an instance of the ResultsHandle class, i.e. a class containing
data frames:

.. testcode::

    results = energyhub.write_results()

To write to a respective excel file:

.. testcode::

    results.write_excel(r'.\userData\results')

