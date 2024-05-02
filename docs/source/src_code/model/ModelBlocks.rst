..  _src-code_blocks:

Model Blocks & Components
==========================

Model Blocks
-------------
In order to define your hierarchical model, `Pyomo Blocks <https://pyomo.readthedocs.io/en/stable/library_reference/kernel/block.html>`_
are used. There are multiple (nested) blocks present:

- ``B_period``: a block holding all investment periods as specified in your ``topology.json`` file.

    - ``B_node``: a block per investment period holding all nodes as specified in your ``topology.json`` file. The rule
      to construct this is held in ``src.model_construction.construct_nodes.py``:

        - ``B_tec``: a block per node holding all technologies as specified in your ``technology.json`` file for that node.
          These are added to the respective nodes through the ``src.model_construction.construct_technology.py`` module.

    - ``B_netw``: a block per investment period holding all networks as specified in your ``network.json`` file. The
      rule to construct this is held in ``src.model_construction.construct_networks.py``:


Block Construction
-------------------

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

Model Components
------------------

In the model, a class ``ModelComponent`` is present. All components (i.e., technologies and networks) are modelled as
subclasses thereof.

.. automodule:: src.components.component
    :members: ModelComponent

For the component-specific documentation for technologies and networks, see their respective documentation pages:

.. toctree::
    :maxdepth: 1

    model_components/technologies
    model_components/networks




