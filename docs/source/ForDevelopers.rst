Developer Instruction
=====================================
This page contains general instructions for the developers that are working on the EnergyHub.


...toctree::
   :maxdepth: 2


Coding conventions
---------------
To keep the code consistent and clear for other developers, try to use the coding conventions that are described in this \
section as much as possible.

For the Pyomo classes we use:

+-------------+--------------+
| Type        | Code         |
+=============+==============+
| Objective   | objective... |
+-------------+--------------+
| Constraint  | const...     |
+-------------+--------------+
| Piecewise   | const...     |
+-------------+--------------+
| Set         | set...       |
+-------------+--------------+
| Block       | b...         |
+-------------+--------------+
| Var         | var...       |
+-------------+--------------+
| Param       | para...      |
+-------------+--------------+
| Disjunct    | dis...       |
+-------------+--------------+
| Disjunction | disjunction..|
+-------------+--------------+
| rule        | init...      |
+-------------+--------------+
| unit        | u            |
+-------------+--------------+

Other names that are regularly used in the EnergyHub are:

+-------------+--------------+
| Type        | Code         |
+=============+==============+
| Timestep    | t...         |
+-------------+--------------+
| Carrier     | car...       |
+-------------+--------------+
| Node        | node...      |
+-------------+--------------+
| Network     | netw...      |
+-------------+--------------+
| Carrier     | car...       |
+-------------+--------------+
| Technology  | tec...       |
+-------------+--------------+
| Consumption | cons...      |
+-------------+--------------+
| Input       | input...     |
+-------------+--------------+
| Output      | output...    |
+-------------+--------------+


Testing new features
---------------
The energyhub comes with a test suite, located in ``.\test``. For new features, try to implement a \
test function in one a respective module (or create a new module). All tests can be executed by \
running py.test from the terminal.