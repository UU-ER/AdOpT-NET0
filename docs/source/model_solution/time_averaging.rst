.. _time_averaging:

Two-stage time averaging algorithm
=====================================
This algorithm solves the model in two stages. In the first stage, input data is averaged over multiple hours and the
model is solved in the first stage with this reduced resolution. In the second stage modelled at full temporal
resolution, technology and network sizes from the first stage serve as a lower bound on the sizes. The approach
is described in Weimann, Gazzani (2022). A novel time discretization method for solving complex multi-energy
system design and operation problems with high penetration of renewable energy.
Computers & Chemical Engineering, 107816.
`doi.org/10.1016/J.COMPCHEMENG.2022.107816 <doi.org/10.1016/J.COMPCHEMENG.2022.107816>`_.

The algorithm is implemented as a sub-class of the DataHandle class and the EnergyHub class

Examplary Usage
^^^^^^^^^^^^^^^^^^
You need to define the system topology and the input data before using the algorithm. Then you can pass the DataHandle
to the :func:`.EnergyHub` class:

.. testcode::

    from src.energyhub import *

    # Construct Model and solve in two stages
    energyhub = EnergyHub_two_stage_time_average(data)
    energyhub.solve_model()


EnergyHubTwoStageTimeAverage Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: src.energyhub.EnergyHubTwoStageTimeAverage
    :members:
    :exclude-members: impose_size_constraints, write_results