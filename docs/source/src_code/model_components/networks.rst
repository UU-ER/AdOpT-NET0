..  _networks:

Networks
=====================================

Networks models available are:

.. contents::
   :local:
   :depth: 2

Not that the plugin :ref:`custom networks <plugin_custom_networks>` allows to add user-defined network
models.

The network class is a subclass of the ModelComponent class.
An overview of all networks that are currently modelled, along with their respective types, can be found
:ref:`here <network_list>`.

A network is defined as the set of all arcs (i.e., connections between nodes) of a
specific network type (e.g., "electricitySimple"). In addition to the performance and
cost parameters in defined in the respective json file of the network, networks can
generally be modelled as either bi- or uni-directional by modifying the parameter
``bidirectional_network`` and ``bidirectional_network_precise`` in the
json file of the respective technology.

If ``bidirectional_network = 1`` the following properties of the network are
enforced:

- The size of an arc in both direction needs to be equal.
- The capex and the fixed opex are only counted once for each arc. As such a
  connection once build can be used in both directions.
- In each time step the flow can only be in one of the two directions.

    - With ``bidirectional_network_precise = 1`` this is enforced with a
      disjunction and a cut adding integers and thus computational complexity in the
      solving.
    - With ``bidirectional_network_precise = 0`` this is enforced with a cut, thus
      not completly eliminating a flow in both directions at the same time.

Any network is a subclass of the :class:`Network` class, which is subclass of the ModelComponent class. In general, all network
subclasses share the equations of this class.

.. automodule:: adopt_net0.core.components.network
    :members: Network

Simple Network
--------------------------------

.. automodule:: adopt_net0.core.components.networks.simple
    :members: Simple

