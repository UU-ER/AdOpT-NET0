.. _src-code_result_management:

Result Management
=====================================

Export to h5
^^^^^^^^^^^^^^^^

Results obtained from the model (in case it is solved) are exported by default to an h5 file as specified in
``Configuration.reporting.save_path``. Additionally, a summary is written to an excel file specified in
``Configuration.reporting.save_summary_path``. In case this excel file exists already, the new summary is appended
as a new row. Documentation on the h5py library and how to handle h5 files can be found
`here <https://docs.h5py.org/en/stable/index.html#>`_

.. automodule:: src.result_management.save_results
    :members:

The structure (object tree) of the resulting HDF5 file is as follows:

Root group (top-level, being the .h5-file) [group] - contains:

* **Summary** [group]: Contains one dataset [leaf] for each variable.

* **Design** (for all time-independent results) [group]

    * Networks [group]

        * Specific network [group]: For each specific network that is present in your model, a separate group is created.
          e.g., "ElectricitySimple".

            * Arc [group]: For each arc that contains that network, another group is created, e.g., "NodeANodeB".

                * Datasets [leaves]: datasets, one for each variable.

    * Nodes [group]

        * Specific node [group]: For each specific node that is present in your model, a separate group is created,
          e.g., "Node A".

            * Technology [group]: For each technology that is present at this node, a separate group is created, e.g.,
              "SteamTurbine".

                * Datasets [leaves]: datasets, one for each variable.

* **Operation** (for all time-dependent results) [group]

    * Networks [group]

        * Specific network [group]: For each specific network that is present in your model, a separate group is created.
          e.g., "ElectricitySimple".

            * Arc [group]: For each arc that contains that network, another group is created, e.g., "NodeANodeB".

                * Datasets [leaves]: datasets, one for each variable.

    * Nodes [group]

        * Specific node [group]: For each specific node that is present in your model, a separate group is created,
          e.g., "Node A".

            * "energy_balance" [group]: A group for the energy balances of all carriers present at that node.

                * Carrier [group]: For each carrier, a specific group is made, e.g., "Electricity".

                    * Datasets [leaves]: datasets of the relevant variables over time.

            * "technology_operation" [group]: a group for the technology operation of all energy technologies present at
              that node.

                * Technology [group]: For each technology, a specific group is made, e.g., "SteamTurbine".

                    * Datasets [leaves]: datasets of the relevant variables over time.

Note: for the time-independent results, one dataset contains only one value, while for the time-dependent results one
dataset contains a value for each timestep in your model run.

.. _export_excel:

Export to Excel
^^^^^^^^^^^^^^^^

We do not provide a direct export to excel/csv files from the model interface, however, you can read results
from the h5 file previously exported. For this, two functions are provided in ``src.result_management.read_results``.
The two functions are documented below.

.. automodule:: src.result_management.read_results
    :members:

.. _results_visualization:

Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^
Provided an h5 file was saved, the data can be visualized on a visualization platform. You can access this platform
by going to https://resultvisualization.streamlit.app/. Further instructions are on the web page.

Note: from the visualization platform, the results can also be downloaded in csv format.

