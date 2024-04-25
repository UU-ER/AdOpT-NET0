.. _backbone_model_constructing:

Model Construction
==================
The directory ``.\src\model_construction`` contains functions adding components to the model
and linking them through the cost, emission and energy/material balance. To quickly construct and solve the model
you can call ``EnergyHub.quick_solve_model`` documented :ref:`here <energyhub_class>`. This function constructs
the model and solves it at the same time. This is typically enough for basic usage. All other components of the model
are documented on the respective page here:

.. toctree::
    :maxdepth: 1

    model_construction/EnergyHubClass
    model_construction/node_construction
    model_construction/technology_construction
    model_construction/network_construction
    model_construction/balances_construction

Find below an example usage to solve a model.

Example Usage
^^^^^^^^^^^^^^^^
To construct the model, you need to have the system topology and input data defined (see
:ref:`here <data-management-example-usage>`) and loaded the modeling configuration. Then you can construct the model as follows:

.. testcode::

    from src.energyhub import *

    energyhub = EnergyHub(data, configuration)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()

Then the model can be solved as documented :ref:`here <model_solving>`. To construct the model and solve it in
one go, you can also do the following. The function combines model construction and solving for convenience.

.. testcode::

    from src.energyhub import *

    energyhub = EnergyHub(data, configuration)
    energyhub.quick_solve_model()


