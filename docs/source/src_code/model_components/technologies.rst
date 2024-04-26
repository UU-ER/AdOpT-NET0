.. _technologies:

Technologies
=====================================
Note that in the model, we distinguish between generic technologies and specific technologies.
Generic technologies offer a framework that can be adjusted for multiple technologies, alone the
performance, the input and output carriers change, while the equations remain the same.

The specific technologies are subclasses for ModelComponent -> Technology, respectively

Specific technologies share equations for 'general technology construction' (with exceptions).

General Technology Construction (same for all technologies)
-------------------------------------------------------------

.. automodule:: src.components.technologies.technology
    :members: Technology


Generic Technologies
--------------------------------
.. automodule:: src.components.technologies.genericTechnologies.res
    :members: Res

.. automodule:: src.components.technologies.genericTechnologies.conv1
    :members: Conv1

.. automodule:: src.components.technologies.genericTechnologies.conv2
    :members: Conv2

.. automodule:: src.components.technologies.genericTechnologies.conv3
    :members: Conv3

.. automodule:: src.components.technologies.genericTechnologies.conv4
    :members: Conv4

.. automodule:: src.components.technologies.genericTechnologies.stor
    :members: Stor


Solid Sorbent Direct Air Capture
----------------------------------------------
.. automodule:: src.components.technologies.specificTechnologies.dac_adsorption
    :members: DacAdsorption

Heat Pump
----------------------------------------------
.. automodule:: src.components.technologies.specificTechnologies.heat_pump
    :members: HeatPump

Gas Turbine
----------------------------------------------
.. automodule:: src.components.technologies.specificTechnologies.gas_turbine
    :members: GasTurbine

Hydro Open
----------------------------------------------
.. automodule:: src.components.technologies.specificTechnologies.hydro_open
    :members: HydroOpen
