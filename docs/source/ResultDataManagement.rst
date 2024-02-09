..  _result_data_management:

Result Management
=====================================

Export to h5
^^^^^^^^^^^^^^^^

Results obtained from the model (in case it solved) are exported by default to an h5 file as specified in
``Configuration.reporting.save_path``. Additionally, a summary is written to an excel file specified in
``Configuration.reporting.save_summary_path``. In case this excel file exists already, the new summary is appended
as a new row (see example below). Documentation on the h5py library and how to handle h5 files can be found
`here <https://docs.h5py.org/en/stable/index.html#>`_

The structure (object tree) of the resulting HDF5 file is as follows:

Root group (top-level, being the .h5-file) [group] - contains:

* Summary [group]
    Contains one dataset [leaf] for each variable.
* Design (for all time-independent results) [group]
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
* Operation (for all time-dependent results) [group]
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


Example Usage
^^^^^^^^^^^^^^^^

Set a folder for saving the results (save_path) and the summary results (save_summary_path) in the Model Configuration.

.. testcode::

    configuration = ModelConfiguration()
    configuration.reporting.save_path = './userData/'
    configuration.reporting.save_summary_path = './userData/'

In this example, the results folder (named with a timestamp of the model run), containing both the Gurobi log and the
HDF5 file, is saved in ``userData``. The Excel file with the summary of each run (one row per run), is saved to the
``userData`` folder as well, but not in a timestamp specified sub-folder.


Export to Excel
^^^^^^^^^^^^^^^^

We do not provide a direct export to excel/csv files from the model interface, however, you can read results
from the h5 file previously exported. Therefore, two functions are provided in ``src.result_management.read_results``.
The two functions are documented below
An examplary usage (can be used after the optimization was done):

.. testcode::

    file_path = './userData/20240206140357/optimization_results.h5'
    save_path = 'whereveryouwanttosaveit'
    print_h5_tree(file_path)
    with h5py.File(file_path, 'r') as hdf_file:
        data = extract_datasets_form_h5(hdf_file["operation/energy_balance/offshore"])
        data.to_excel(save_path)
        print(data)

.. automodule:: src.result_management.read_results
    :members:
