.. _workflow_solve-model:

Solve Model
=====================================
After model construction, the model can be solved with the :func:`quick_solve` method of the :ref:`EnergyHub class<energyhub_class>`.
This method automatically construct the model (i.e., constructing balances, nodes, technologies and networks) and
subsequently solves it. For background information on this construction, see :ref:`here<src-code_model-constructing>`.

Note: Unless :ref:`adjusted<workflow_model-configuration>` in ``ModelConfig.JSON``, the model is solved with a full temporal
resolution. Options to reduce solving time are found :ref:`here<workflow_model-configuration>`.