..   _dynamics:

Dynamics
=====================================
In addition to the standard performance constraints that describe the efficiency, it is possible to add a set of
constraints that approximates the dynamic operation of technologies more precisely. The constraints are based on the
work of Morales-España, G., Ramírez-Elizondo, L., & Hobbs, B. F. (2017). Hidden power system inflexibilities imposed by
traditional unit commitment formulations. Applied Energy, 191, 223–238. https://doi.org/10.1016/J.APENERGY.2017.01.089.
In the model framework we differentiate between technology constraints for fast dynamics and slow dynamics, both can only
be used for technologies CONV1, CONV2 and CONV3. For the constraints to be added you first need to select the dynamics
option in the ``ConfigModel.json`` file. The fast dynamic constraints can be used in combination only with performance types
2 and 3, and limits the startup and shutdown load, the minimum up and down times and the maximum number of startups. The
slow dynamics are modeled through performance type 4 and enables the modeling of startups and shutdowns that take longer than
the used time resolution (typically 1 hour). It is not possible to limit the startup and shutdown load with slow dynamics.



