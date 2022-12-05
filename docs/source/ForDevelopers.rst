Developer Instruction
=====================================
This page contains general instructions for the developers that are working on the EnergyHub.


.. toctree::
   :maxdepth: 2


Coding conventions
---------------
To keep the code consistent and clear for other developers, try to use the coding conventions that are described in this \
section as much as possible.

For the Pyomo classes we use:

+-------------+--------------+
| Type        | Code         |
+=============+==============+
| Objective   | objective_   |
+-------------+--------------+
| Constraint  | const_       |
+-------------+--------------+
| Piecewise   | const_       |
+-------------+--------------+
| Set         | set_         |
+-------------+--------------+
| Block       | b_           |
+-------------+--------------+
| Var         | var_         |
+-------------+--------------+
| Param       | para_        |
+-------------+--------------+
| Disjunct    | dis_         |
+-------------+--------------+
| Disjunction | disjunction_ |
+-------------+--------------+

Other names that are regularly used in the EnergyHub are:

+-------------+--------------+
| Type        | Code         |
+=============+==============+
| Timestep    | t_           |
+-------------+--------------+
| Carrier     | car_         |
+-------------+--------------+
| Node        | node_        |
+-------------+--------------+
| Network     | netw_        |
+-------------+--------------+
| Carrier     | car_         |
+-------------+--------------+
| Technology  | tec_         |
+-------------+--------------+
| Consumption | cons_        |
+-------------+--------------+
| Input       | input_       |
+-------------+--------------+
| Output      | output_      |
+-------------+--------------+