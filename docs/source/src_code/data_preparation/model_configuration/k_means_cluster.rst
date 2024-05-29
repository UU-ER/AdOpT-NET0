..   _clustering:

Time series aggregation
=====================================
To reduce the temporal resolution of the model, the input data can be clustered into typical design days as described in
`Gabrielli et al. (2018). Optimal design of multi-energy systems with seasonal
storage. Applied Energy, 219, 408–424. <https://doi.org/10.1016/j.apenergy.2017.07
.142>`_

We have implemented method 1 and 2 of the aforementioned paper using the `tsam <https://tsam.readthedocs.io/en/latest/>`_
package. The clustering is performed in a routine of the DataHandle class.

Method 1 (M1)
------------------------
Input data as well as technology and network performances are all reduced in their time
resolution. Storage levels, however, are modelled on a full time-scale to account for
seasonal changes.

Method 1 (M2)
------------------------
Input data is kept at full resolution while the user is free to decide which
technologies should be modelled at reduced resolution. The default, as implemented in
the code, only storage technologies and renewable technologies are modelled at full
resolution. While this method comes with a high performance advantage for models with
a large number of integers or binaries, it is also not trivial to make the right
choices of technologies being modelled at full resolution. This is due to
infeasibilities caused by technologies operating at reduced resolution while required
to satisfy demands at full resolution. A simple solution for this problem is allowing
for a violation of the energy balance or allowing for import.


To use this method, you need to adjust the model configuration by setting a number of
typical days N and the clustering method. in ``ConfigModel.json`` as shown in
:ref:`this example <workflow_example-usage>`.