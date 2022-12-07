Technology Construction
=====================================
The module ``src.model_construction.construct_technology.py`` contains the rule to construct each
technology and the module ``src.model_construction.generic_technology_constraints.py`` contains specific
constraints for all generic technologies.

Note that in the model, we distinguish between generic technologies and specific technologies.
Generic technologies offer a framework that can be adjusted for multiple technologies, alone the
performance, the input and output carriers change, while the equations remain the same.

Specific technologies are contained in the module
``src.model_construction.specific_technology_constraints.py`` and resemble a specific technology
only.

General Technology Construction (same for all technologies)
-------------------------------------------------------------

.. automodule:: src.model_construction.construct_technology
    :members:


Generic Technology Constraints
--------------------------------
.. automodule:: src.model_construction.generic_technology_constraints
    :members:
