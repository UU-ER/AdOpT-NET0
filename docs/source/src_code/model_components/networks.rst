..  _networks:

Networks
=====================================

The network class is a subclass of the ModelComponent class.
Furthermore, the network class has different subclasses that model different types of networks (fluid, electricity, simple).
An overview of all networks that are currently modelled, along with their respective types, can be found
:ref:`below <network_list>`.

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

..  _network_list:

List of Networks
-----------------

All networks that are modelled are listed below.

.. csv-table::
   :file: generated_netw_list.csv
   :header-rows: 1
   :delim: ;

Network Class
-------------------------------------------------------------
As mentioned, the network class is a subclass of the ModelComponent class. In general, all network subclasses
share the equations of this class, though some exceptions are there for specific technologies (the subclass then
overwrites the class method).

.. automodule:: adopt_net0.components.networks.network
    :members: Network

Generic Network
--------------------------------
.. automodule:: adopt_net0.components.networks.genericNetworks.fluid
    :members: Fluid

.. automodule:: adopt_net0.components.networks.genericNetworks.electricity
    :members: Electricity

.. automodule:: adopt_net0.components.networks.genericNetworks.simple
    :members: Simple

