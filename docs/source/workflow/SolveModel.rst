.. _workflow_solve-model:

Solve Model
=====================================
After model construction, the model can be solved with the :func:`quick_solve` method of the :ref:`EnergyHub class<energyhub_class>`.
This method automatically construct the model (i.e., constructing balances, nodes, technologies and networks) and
subsequently solves it. For background information on this construction, see :ref:`here<src-code_model-constructing>`.