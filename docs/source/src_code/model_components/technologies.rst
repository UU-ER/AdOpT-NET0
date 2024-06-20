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

Additionally, you can attach a post combustion CCS to any technology (see :ref:`here
<ccs_docu>`)


..  _technologies_list:

List of Technologies
----------------------

All technologies that are modelled are listed below, as well as their respective technology models (i.e.,
types of technologies that follow similar constraints, which are explained :ref:`here<technologies>`).

.. csv-table::
   :file: generated_tech_list.csv
   :header-rows: 1
   :delim: ;


Technology Class
-------------------------------------------------------------
As mentioned, the technology class is a subclass of the ModelComponent class. In general, all technology subclasses
share the equations of this class, though some exceptions are there for specific technologies (the subclass then
overwrites the class method).

.. automodule:: adopt_net0.components.technologies.technology
    :members: Technology


Generic Technologies
--------------------------------
.. automodule:: adopt_net0.components.technologies.genericTechnologies.res
    :members: Res

.. automodule:: adopt_net0.components.technologies.genericTechnologies.conv1
    :members: Conv1

.. automodule:: adopt_net0.components.technologies.genericTechnologies.conv2
    :members: Conv2

.. automodule:: adopt_net0.components.technologies.genericTechnologies.conv3
    :members: Conv3

.. automodule:: adopt_net0.components.technologies.genericTechnologies.conv4
    :members: Conv4

.. automodule:: adopt_net0.components.technologies.genericTechnologies.stor
    :members: Stor

Specific Technologies
----------------------------------------------

**Solid Sorbent Direct Air Capture**

.. automodule:: adopt_net0.components.technologies.specificTechnologies.dac_adsorption
    :members: DacAdsorption

**Heat Pump**

.. automodule:: adopt_net0.components.technologies.specificTechnologies.heat_pump
    :members: HeatPump

**Gas Turbine**

.. automodule:: adopt_net0.components.technologies.specificTechnologies.gas_turbine
    :members: GasTurbine

**Hydro Open**

.. automodule:: adopt_net0.components.technologies.specificTechnologies.hydro_open
    :members: HydroOpen

..  _ccs_docu:

Carbon Capture
----------------------

The carbon capture object (CCS, even though it refers just to the capture technology), which does not constitute an independent technology itself, can be attached to any technology with a positive emission factor.  To do this, you need to add (if not already present) the following lines of code to the json file under the “Performance” section of the technology you wish to equip with CCS:

"ccs": {
  "possible": 1,
  "co2_concentration" : 0.08,
  "ccs_type": "MEA_medium"
},

To see an example of how this is done, you can look at the json file of the GasTurbine_simple_CCS technology. When you want to have the possibility of installing CCS, you need to set the “possible” option to 1. Moreover, you can specify the CO2 concentration in the flue gas of your emitting technology; this will influence the costs and energy performance of the CCS. With “ccs_type” you can specify the specific capture technology you wish to use. So far, only post combustion capture with MEA is modelled (following the work of Weimann et Al. 2023 https://doi.org/10.1016/j.apenergy.2023.120738), and you can choose the between small, medium and large according to the size range that you expect for the capture plant.

The CCS needs heat and electricity as input to run. Therefore, you have to make sure to have those carriers available. Moreover, the CCS will produce an extra carrier as output of the emitting technology called “CO2captured” in amount equal to the CO2 captured from the flue gas.
The CO2captured needs to be sent to a storage site or exported. In AdOpT-NET0 a storage site is represented with the SINK class of technologies. So far, only the “PermanentStorage_CO2_simple” SINK technology is available. This storage takes CO2captured and electricity (for compression and injection) as input and it has no output. The costs of this sink are given only by the amount of CO2 stored times a fixed cost per ton of CO2.

To summarize, if you want to add the CCS option to a technology you have to:
-	Add “ccs” section in the “Performance” of the json file as above
-	Have electricity, heat (with import possibility if required) and CO2captured as carriers
-	Have a SINK or CO2captured export option
-	Add CO2 transport option if sink and capture are in different nodes
