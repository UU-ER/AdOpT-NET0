..  _workflow_solve-model:

Solve Model
=====================================
After model construction, the model can be solved with the :func:`quick_solve` method of the :ref:`EnergyHub class<energyhub_class>`.
Unless :ref:`adjusted<workflow_model-configuration>` in ``ModelConfig.JSON``, the model is solved with a full temporal
resolution. Options to reduce solving time are found :ref:`here<workflow_model-configuration>`.