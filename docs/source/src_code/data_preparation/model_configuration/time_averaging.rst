..   _time_averaging:

Two-stage time averaging algorithm
=====================================
This algorithm can help to speed up the optimization. It solves the model in two
stages. In the first stage, input data is averaged over multiple hours and the
model is solved in the first stage with this reduced resolution. In the second stage modelled at full temporal
resolution, technology and network sizes from the first stage serve as a lower bound on the sizes. The approach
is described in Weimann, Gazzani (2022). A novel time discretization method for solving complex multi-energy
system design and operation problems with high penetration of renewable energy.
Computers & Chemical Engineering, 107816.
`doi.org/10.1016/J.COMPCHEMENG.2022.107816 <https://doi.org/10.1016/J.COMPCHEMENG.2022
.107816>`_.

The algorithm is implemented as a method of the DataHandle class and the EnergyHub
class. For using this method in your solve, adjust the value for the "timestaging"
setting in ``ConfigModel.json``, accordingly. For example:


.. testcode::

        "timestaging": {
            "description": "Defines number of timesteps that are averaged (0 = off).",
            "value": 4
        },
