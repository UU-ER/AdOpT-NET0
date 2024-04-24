.. _data-preprocessing_load-data:

Load Input Data
=====================================

Now that you have the :ref:`correct folder structure and templates for the data <data-preprocessing_create-data-templates>`,
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
your :ref:`input data folder path<data-preprocessing_create-data-templates>`. This imports the climate data for each node,
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
^^^^^^^^^^^^^^^^

For carrier data, think of demand or import/export limits of a specific carrier at a specific node, you can use the
``fill_carrier_data`` method if your value does not vary over time.

.. automodule:: src.data_preprocessing.data_loading
    :members: fill_carrier_data
    :exclude-members:

..  _load-data_profile:

Specifying a profile
^^^^^^^^^^^^^^^^
If the carrier data values do change over time (e.g., you want to have a demand profile resembling actual load) you have
to manually specify the profiles in the carrier csv files in your input data folder. For example, you can copy a demand
profile from a national database (if your nodes are countries) into the correct column in the csv.
