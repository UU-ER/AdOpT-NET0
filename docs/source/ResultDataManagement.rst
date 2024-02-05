..  _result_data_management:

Result Data Management
=====================================

The result data management is done with the method ``write_optimization_results_to_h5``, called from the energyhub.
This function is written in the python file ``handle_optimization_results.py`` and it serves two purposes. The first
being: it creates a dictionary for the results summary (i.e., information about costs, emissions and the run
specifications), which is returned by the method to the energyhub. There, the summary results are appended to a
"Summary" Excel, the file path for which can be specified in ``configuration.reporting.save_summary_path``
(see example below). The second purpose is to collect all results from the model to save into a single HDF5 file, using
the h5py library. Documentation for this package can be found `here <https://docs.h5py.org/en/stable/index.html#>`_.
For clear code, any pre-processing to obtain the resulting values for specific variables are moved to the
``utilities.py`` file. Again, the path of the HDF5 file can be specified in the configuration, with:
``configuration.reporting.save_path`` (see example below).

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

