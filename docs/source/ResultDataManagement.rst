..  _result_data_management:

Result Data Management
=====================================

Result data management works with the class ``src.result_management.handle_optimization_results.ResultsHandle``
class. It lets you export results to dataframes and to excel. In the model configuration, the user can chose to
what level of detail results should be saved to the disk: for :python:`configuration.reporting.save_detailed = 1`,
all results are saved to the folder specified in :python:`configuration.reporting.save_path`. For
:python:`configuration.reporting.save_detailed = 0`, only a summary is saved (i.e. technology sizes, network sizes,
and other high level results. The dataframes for a specific optimization results can also be retrieved. (see
example below)

Example Usage
^^^^^^^^^^^^^^^^
Set a folder for saving the results in the Model Configuration and set the level of detail.

.. testcode::

    configuration = ModelConfiguration()
    configuration.reporting.save_detailed = 1
    configuration.reporting.save_path = './result_folder/'
    results = energyhub.quick_solve()

:python:`results` now holds detailed results on the last optimization run (see e.g. :python:`results.energybalance`  or
:python:`self.detailed_results.nodes`

