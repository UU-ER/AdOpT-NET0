Making Templates
=====================================
The module ``src.data_management.create_templates.py`` includes functions to create empty templates for the system
topology:

.. automodule:: src.data_management.create_templates
    :members:

Example Usage
---------------

Creating an empty topology dictionary and filling it:

.. testcode::

    from src.data_management.create_templates import create_empty_topology

    topology = create_empty_topology()
    topology['timesteps'] = 8760
    topology['timestep_length_h'] = 1
    topology['carriers'] = ['electricity', 'heat', 'gas']
    topology['nodes'] = ['onshore', 'offshore']
    topology['technologies']['onshore'] = ['PV', 'Furnace_NG']
    topology['technologies']['offshore'] = ['WT_OS_11000']

Creating an empty network dictionary and filling it:

.. testcode::

    from src.data_management.create_templates import create_empty_network_data

    network_data = create_empty_network_data(topology['nodes'])
    network_data['distance'].at['onshore', 'offshore'] = 100
    network_data['distance'].at['offshore', 'onshore'] = 100
    network_data['connection'].at['onshore', 'offshore'] = 1
    network_data['connection'].at['offshore', 'onshore'] = 1
    topology['networks']['electricity']['AC'] = network_data

