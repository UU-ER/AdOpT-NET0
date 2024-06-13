.. _time_aggregation:

=========================
Aggregation Algorithms
=========================

Clustering into typical days
------------------------------
To reduce the temporal resolution of the model, the input data can be clustered into typical design days as described in
`Gabrielli et al. (2018). Optimal design of multi-energy systems with seasonal
storage. Applied Energy, 219, 408–424. <https://doi.org/10.1016/j.apenergy.2017.07
.142>`_

We have implemented method 1 and 2 of the aforementioned paper using the `tsam <https://tsam.readthedocs.io/en/latest/>`_
package. The clustering is performed in a routine of the DataHandle class.

Method 1 (M1)
^^^^^^^^^^^^^^^^^^^^
Input data as well as technology and network performances are all reduced in their time
resolution. Storage levels, however, are modelled on a full time-scale to account for
seasonal changes.

Method 1 (M2)
^^^^^^^^^^^^^^^^^^^^
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


Two-stage time averaging algorithm
------------------------------------
This algorithm can help to speed up the optimization. It solves the model in two
stages. In the first stage, input data is averaged over multiple hours and the
model is solved in the first stage with this reduced resolution. In the second stage modelled at full temporal
resolution, technology and network sizes from the first stage serve as a lower bound on the sizes. The approach
is described in Weimann, Gazzani (2022). A novel time discretization method for solving complex multi-energy
system design and operation problems with high penetration of renewable energy.
Computers & Chemical Engineering, 107816.
`doi.org/10.1016/J.COMPCHEMENG.2022.107816 <https://doi.org/10.1016/J.COMPCHEMENG.2022
.107816>`_.

The algorithm is implemented as a method of the DataHandle class and the ModelHub
class. For using this method in your solve, adjust the value for the "timestaging"
setting in ``ConfigModel.json``, accordingly. For example:


.. testcode::

        "timestaging": {
            "description": "Defines number of timesteps that are averaged (0 = off).",
            "value": 4
        },
