..   _clustering:

Typical Day Clustering
=====================================
To reduce the temporal resolution of the model, the input data can be clustered into typical design days as described in
Gabrielli et al. (2018). Optimal design of multi-energy systems with seasonal storage. Applied Energy, 219, 408–424.
`doi.org/10.1016/j.apenergy.2017.07.142 <doi.org/10.1016/j.apenergy.2017.07.142>`_.

The algorithm is implemented in a sub-class of the DataHandle class. The energybalance, storage technologies 'STOR' and
renewable technologies 'RES' as well as networks are always modelled at full resolution. Other technologies are
modelled with a reduced resolution.

To implement this method, you need to adjust the model configuration by setting a number of typical days N
in ``ConfigModel.json`` as shown in :ref:`this example <workflow_example-usage>`.