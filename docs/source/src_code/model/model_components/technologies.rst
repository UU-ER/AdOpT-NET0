..  _technologies:

Technologies
=====================================

The technology class is a subclass of the ModelComponent class. In the model, we further distinguish between
generic technologies and specific technologies. Generic technologies offer a framework that can be adjusted for multiple
technologies: only the performance parameters, and the input and output carriers change, while the equations remain the same.

Technology types, divided by generic and specific technologies:

- Generic technologies:

    - RES
    - CONV1
    - CONV2
    - CONV3
    - CONV4
    - STOR

- Specific technologies:

    - DAC adsorption
    - Gas turbine
    - Heat pump
    - Hydro open

All technology types listed above are modelled as subclasses of the :class:`Technology` class. An overview of all
technologies that are currently modelled, and the technology classes / types used to model them, can be found
:ref:`below <technologies_list>`.


Technology Class
-------------------------------------------------------------
As mentioned, the technology class is a subclass of the ModelComponent class. In general, all technology subclasses
share the equations of this class, though some exceptions are there for specific technologies (the subclass then
overwrites the class method).

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

Specific Technologies
----------------------------------------------

**Solid Sorbent Direct Air Capture**

.. automodule:: src.components.technologies.specificTechnologies.dac_adsorption
    :members: DacAdsorption

**Heat Pump**

.. automodule:: src.components.technologies.specificTechnologies.heat_pump
    :members: HeatPump

**Gas Turbine**

.. automodule:: src.components.technologies.specificTechnologies.gas_turbine
    :members: GasTurbine

**Hydro Open**

.. automodule:: src.components.technologies.specificTechnologies.hydro_open
    :members: HydroOpen

Carbon Capture and Storage Technologies
----------------------------------------------

Explanation of CCS & Sink

..  _technologies_list:

List of Technologies
----------------------

All technologies that are modelled are listed below, as well as their respective technology models (i.e.,
types of technologies that follow similar constraints, which are explained :ref:`here<technologies>`).

.. csv-table::
   :file: generated_tech_list.csv
   :header-rows: 1
   :delim: ;