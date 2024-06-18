.. _workflow_solve-model:

=====================================
Model construction and solving
=====================================
After model construction, the model can be solved with the :func:`quick_solve` method of the :ref:`ModelHub class<modelhub_class>`.
This method automatically construct the model (i.e., constructing balances, nodes, technologies and networks) and
subsequently solves it. For background information on this construction, see :ref:`here<src-code_model-constructing>`.

.. testcode::

    m = adopt.ModelHub()
    m.read_data(path, start_period=None, end_period=None)
    m.quick_solve()
