.. _installation:

=====================================
Installation
=====================================

Using pip
----------

You can use the standard utility for installing Python packages by executing the
following in a shell:

.. testcode::

    pip install adopt_net0

Additionally, you need a `solver installed, that is supported by pyomo
<https://pyomo.readthedocs.io/en/6.8.0/solving_pyomo_models.html#supported-solvers>`_
(we recommend gurobi, which has a free academic licence). An open-source solver
that can be used is `GLPK <https://www.gnu.org/software/glpk/>`_. Note that the user
is responsible to install a Pyomo supported solver.

Note for mac users: The export of the optimization results require a working
`hdf5 library <https://www.hdfgroup.org/solutions/hdf5/>`_. On windows this should be
installed by default. On mac, you can install it with homebrew:

.. testcode::

    brew install hdf5




