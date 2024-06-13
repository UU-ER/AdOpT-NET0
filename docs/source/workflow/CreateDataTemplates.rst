.. _workflow_create-data-templates:

3 Creating Input Data Templates
=====================================
After you have defined your system topology, you can now create the folder structure and the templates for the
optimization based on the specified topology. For this, you call the :func:`create_input_data_folder_template` method,
passing your input data folder path (which must be the same folder path as for the :ref:`Model Templates<workflow_create-model-templates>`).

.. automodule:: src.data_preparation.template_creation
    :members: create_input_data_folder_template
    :exclude-members:

This yields a CSV file ``NodeLocations`` and an input data folder for each investment period and each node.

Note: See a complete documentation of all template creation functions in the
:ref:`source code documentation<src-code_data-preparation>`.

Your directory should now contain the following files:

- ``Topology.JSON``
- ``ConfigModel.JSON``
- ``NodeLocations.csv``
- A folder for each investment period that you specified in the topology, containing:

    - ``Networks.JSON``
    - A folder called ``network_data``
    - A folder called ``network_topology``, which itself contains:

        - A folder called ``existing``: containing the data templates for all existing network types.

            - Note that for each of the networks that you specify as existing in ``Networks.JSON``, an additional folder
              called after the network name has to be created containing the data templates for the existing networks.
        - A folder called ``new``: containing the data templates for all new network types.

            - Note that for each of the networks that you specify as new in ``Networks.JSON``, an additional folder
              called after the network name has to be created containing the data templates for the new networks.

    - A folder called ``node_data``, containing:

        - A folder for each node that you specified in the topology, containing:

            - ``Technologies.JSON``
            - ``CarbonCost.csv``
            - ``ClimateData.csv``
            - A folder called ``carrier_data``, containing:

                - A ``carrier_name.csv`` file for each carrier.
                - ``EnergybalanceOptions.JSON``
            - A folder called ``technology_data``
