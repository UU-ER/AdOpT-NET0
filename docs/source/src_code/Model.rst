.. _src-code_model:

Model
====================

In order to understand how the model is structured and constructed, the documentation pages below will guide you:

.. toctree::
    :maxdepth: 1

    model/EnergyHubClass
    model/ModelBlocks
    model/Balances

.. _src-code_model-constructing:

Model Construction & Solving
------------------------------

The model is constructed by calling the ``EnergyHub.construct_model()`` method. This method clearly illustrates the
structure of the model. There are sets of investment periods, nodes and carriers based on the ``topology.json``. Then,
a block is created for the investment periods (holding all investment periods in the set), the networks (holding all
network types), the nodes (holding all nodes in the set), and for the technologies (holding all technology types). The
methods for the block creation are called by the aforementioned ``EnergyHub.construct_model()`` method from the
``.\src\model_construction`` directory (with a separate module for each block type).

Then, these blocks are all linked through :ref:`cost, emissions and energy / material balances<src-code_balances>` by
calling the ``EnergyHub.construct_balances()`` method (again retrieved from the ``.\src\model_construction`` directory).

Finally, the model can be solved using ``EnergyHub.solve()``.

Note: the three steps above are combined in the ``EnergyHub.quick_solve()`` method to quickly construct and solve the
model at once. All of the methods are documented :ref:`here <energyhub_class>`.
