..  _result_data_management:

Result Data Management
=====================================

Result data management works with the class ``src.result_management.handle_optimization_results.ResultsHandle``
class. It lets you export results to dataframes and to excel.

Example Usage
^^^^^^^^^^^^^^^^
Set a folder for saving the results in the Model Configuration and set the level of detail.

.. testcode::

    configuration = ModelConfiguration()
    configuration.reporting.save_detailed = 1
    configuration.reporting.save_path = './result_folder/'
    results = energyhub.quick_solve()

To write to a summary excel file, do (it will write it to the same folder):

.. testcode::

    results.write_excel('results')

