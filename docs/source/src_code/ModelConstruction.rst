.. _src-code_model-constructing:

====================
Model Construction
====================

The model is constructed by calling the ``EnergyHub.construct_model()`` method. This method clearly illustrates the
structure of the model. There are sets of investment periods, nodes and carriers based on the ``topology.json``. Then,
a block is created for each investment period (all investment periods in the set), each network, each node, and for each technologies. The
methods for the block creation are called by the aforementioned ``EnergyHub.construct_model()`` method from the
``.\src\model_construction`` directory (with a separate module for each block type).

Then, these blocks are all linked through :ref:`cost, emissions and energy / material balances<src-code_balances>` by
calling the ``EnergyHub.construct_balances()`` method (again retrieved from the ``.\src\model_construction`` directory).

Finally, the model can be solved using ``EnergyHub.solve()``.

Note: the three steps above are combined in the ``EnergyHub.quick_solve()`` method to quickly construct and solve the
model at once. All of the methods are documented :ref:`here <energyhub_class>`.


..  _energyhub_class:

Energy Hub Class
================

.. automodule:: src.energyhub
    :members:
    :exclude-members: calculate_occurance_per_hour


..  _src-code_blocks:

Model Structure
==========================

In order to define your hierarchical model, `Pyomo Blocks <https://pyomo.readthedocs.io/en/stable/library_reference/kernel/block.html>`_
are used. There are multiple (nested) blocks present:

- ``b_period``: a block holding all investment periods as specified in your
  ``topology.json`` file.

    - ``b_node``: a block per investment period holding all nodes as specified in
      your ``topology.json`` file. The rule
      to construct this is held in ``src.model_construction.construct_nodes.py``:

        - ``b_tec``: a block per node holding all technologies as specified in your
          ``technology.json`` file for that node.
          These are added to the respective nodes through the ``src.model_construction.construct_technology.py`` module.

    - ``b_netw``: a block per investment period holding all networks as specified in
      your ``network.json`` file. The
      rule to construct this is held in ``src.model_construction.construct_networks.py``:


Block Construction
==========================

The aforementioned blocks are constructed based on the rules in their respective modules. These are:

For investment periods, contained in the ``src.model_construction.construct_investment_period.py`` module:

    .. automodule:: src.model_construction.construct_investment_period
        :members:

For nodes, from the ``src.model_construction.construct_nodes.py`` module:

    .. automodule:: src.model_construction.construct_nodes
        :members: construct_node_block

For technologies, contained in the ``src.model_construction.construct_technology.py`` module.

    .. automodule:: src.model_construction.construct_technology
        :members:

For networks, contained in the ``src.model_construction.construct_networks.py`` module.

    .. automodule:: src.model_construction.construct_networks
        :members:

..  _src-code_balances:

Balance Construction
==========================

All model blocks and components are linked through "balances". These are calculations of total emissions and costs,
and carrier (i.e., energy and/or material) balances. All carriers must be in balance for each node and on the global level.
Violation of balances is only possible if specifically allowed for in the configuration (see
:ref:`here <workflow_model-configuration>`).

The module ``.\src\model_construction\construct_balances`` contains the rules to construct these balances. These
functions are called after the nodes and networks have been initialized, i.e. after the blocks have been constructed.

.. automodule:: src.model_construction.construct_balances
    :members:



