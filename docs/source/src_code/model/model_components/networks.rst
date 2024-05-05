..  _networks:

Networks
=====================================

The network class is a subclass of the ModelComponent class. All networks are modelled using this class, and there are
no subclasses of specific network models. An overview of all networks that are currently modelled can be found
:ref:`below <network_list>`. A network is defined as the set of all arcs (i.e., connections between nodes) of a
specific network type (e.g., "electricitySimple").

.. automodule:: src.components.networks.network
    :members: Network

..  _network_list:

List of Networks
-----------------

All networks that are modelled are listed below.

.. csv-table::
   :file: ..\..\..\generated_netw_list.csv
   :header-rows: 1
   :delim: ;