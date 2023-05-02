..  _result_data_management:

Result Data Management
=====================================

Result data management works with the class ``src.data_management.handle_optimization_results.ResultsHandle``
class. It lets you export results to dataframes and to excel.

Example Usage
^^^^^^^^^^^^^^^^
To export data from the pyomo model to an instance of the ResultsHandle class, i.e. a class containing
data frames:

.. testcode::

    results = energyhub.quick_solve()

To write to a respective excel file:

.. testcode::

    results.write_excel(r'.\userData\results')

