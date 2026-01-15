..  _technologies:

Technologies
=====================================

Technology models available are:

.. contents::
   :local:
   :depth: 2

Not that the plugin :ref:`custom technologies <plugin_custom_technologies>` allows to add user-defined technology
models.

All technology types listed above are modelled as subclasses of the :class:`Technology` class. An overview of all
technologies that are currently modelled, and the technology classes / types used to model them, can be found
:ref:`here <technologies_list>`.


All technology subclasses share the equations of this class, though some exceptions are there for specific
technologies (the subclass then overwrites the class method).

.. automodule:: adopt_net0.core.components.technology
    :members: Technology


Renewable Energy Source (RES) Technology
----------------------------------------------

.. automodule:: adopt_net0.core.components.technologies.res
    :members: Res

Conversion technology type 1
----------------------------------------------

.. automodule:: adopt_net0.core.components.technologies.conv1
    :members: Conv1

Conversion technology type 2
----------------------------------------------

.. automodule:: adopt_net0.core.components.technologies.conv2
    :members: Conv2

Conversion technology type 3
----------------------------------------------

.. automodule:: adopt_net0.core.components.technologies.conv3
    :members: Conv3

Conversion technology type 3
----------------------------------------------

.. automodule:: adopt_net0.core.components.technologies.conv4
    :members: Conv4

Storage technology
----------------------------------------------

.. automodule:: adopt_net0.core.components.technologies.stor
    :members: Stor


Sink technology
----------------------------------------------

.. automodule:: adopt_net0.core.components.technologies.sink
    :members: Sink


