..  _src-code_balances:

Balances
=====================================

All model blocks and components are linked through "balances". These are calculations of total emissions and costs,
and carrier (i.e., energy and/or material) balances. All carriers must be in balance for each node and on the global level.
Violation of balances is only possible if specifically allowed for in the configuration (see
:ref:`here <workflow_model-configuration>`).

Balance Construction
----------------------

The module ``.\src\model_construction\construct_balances`` contains the rules to construct these balances. These
functions are called after the nodes and networks have been initialized, i.e. after the blocks have been constructed.

.. automodule:: src.model_construction.construct_balances
    :members:
