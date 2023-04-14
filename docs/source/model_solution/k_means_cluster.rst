.. _clustering:

Typical Day Clustering
=====================================
To reduce the temporal resolution of the model, the input data can be clustered into typical design days as described in
Gabrielli et al. (2018). Optimal design of multi-energy systems with seasonal storage. Applied Energy, 219, 408–424.
`doi.org/10.1016/j.apenergy.2017.07.142 <doi.org/10.1016/j.apenergy.2017.07.142>`_.

The algorithm is implemented as a sub-class of the DataHandle class.

Examplary Usage
^^^^^^^^^^^^^^^^^^
You need to define the system topology and the input data before clustering the data. Then you can initialize
an instance of the :func:`.ClusteredDataHandle` class and pass it to the :func:`.EnergyHub` class:

.. testcode::

    import src.data_management as dm

    # Perform Clustering
    typical_days = 10
    data_clustered = dm.ClusteredDataHandle(data, typical_days)

    # Construct Model and solve
    ehub = EnergyHub(data_clustered)
    ehub.quick_solve_model()


ClusteredDataHandle Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: src.data_management.handle_input_data.ClusteredDataHandle