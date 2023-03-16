Technology Construction
=====================================
The module ``src.model_construction.construct_technology.py`` contains the rule to construct each
technology and the module ``src.model_construction.technology_constraints.py`` contains constraints for all
technologies.

Note that in the model, we distinguish between generic technologies and specific technologies.
Generic technologies offer a framework that can be adjusted for multiple technologies, alone the
performance, the input and output carriers change, while the equations remain the same.

Specific technologies share equations for 'general technology construction' (with exceptions).

General Technology Construction (same for all technologies)
-------------------------------------------------------------

.. automodule:: src.model_construction.construct_technology
    :members:


Generic Technology Constraints
--------------------------------
.. automodule:: src.model_construction.technology_constraints.generic_technology_constraints
    :members:


Solid Sorbent Direct Air Capture
----------------------------------------------
.. automodule:: src.model_construction.technology_constraints.dac_adsorption
    :members:

Heat Pump
----------------------------------------------
.. automodule:: src.model_construction.technology_constraints.heat_pump
    :members:

Gas Turbine
----------------------------------------------
.. automodule:: src.model_construction.technology_constraints.gas_turbine
    :members:
