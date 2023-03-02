..  _model_solving:

Model Solving
=====================================
After model construction, the model can be solved with the respective method of the EnergyHub class.
This method solves the model with a full temporal resolution. However, often the model is too complex
to solve within a reasonable amount of time. Therefore, the framework offers additional options:

.. toctree::
    :maxdepth: 1

    model_solution/k_means_cluster
    model_solution/time_averaging

A simple solve after model construction is documented below:

Example Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After model construction, the method :func:`.EnergyHub.solve_model()` can be called:

.. testcode::

    from src.energyhub import *

    energyhub.solve_model(objective = 'cost')