.. _model_configuration:

Model Configuration
================
The class ``src.model_configuration`` is used to define and change the standard modeling configuration. An example
on how to use the class is given in :ref:`Example Usage <configuration-example-usage>`. The class also specifies the
private settings, identified by two leading underscores, that should be treated as a non-public part of the code.

.. automodule:: src.model_configuration
    :members:

List of optimization settings that can be specified:

+------------------+--------------------------------------------------------+---------------------------------------------+
| Name             | Definition                                             | Options                                     |
+------------------+--------------------------------------------------------+---------------------------------------------+
| objective        | String specifying the objective/type of optimization   | 'costs', 'emissions_pos', 'emissions_neg',  |
|                  |                                                        | 'emissions_minC', 'pareto'                  |
+------------------+--------------------------------------------------------+---------------------------------------------+
| montecarlo.range | Value defining the range in which variables are varied |                                             |
|                  | in Monte Carlo simulations                             |                                             |
+------------------+--------------------------------------------------------+---------------------------------------------+
| montecarlo.N     | Number of Monte Carlo simulations                      |                                             |
+------------------+--------------------------------------------------------+---------------------------------------------+
| pareto.N         | Number of Pareto points                                |                                             |
+------------------+--------------------------------------------------------+---------------------------------------------+
| timestaging      | Switch to turn timestaging on/off                      | {0,1}                                       |
+------------------+--------------------------------------------------------+---------------------------------------------+
| tecstaging       | Switch to turn tecstaging on/off                       | {0,1}                                       |
+------------------+--------------------------------------------------------+---------------------------------------------+

List of solver settings that can be specified:

+---------+-----------------------------------------+----------+
| Name    | Definition                              | Options  |
+---------+-----------------------------------------+----------+
| solver  | String specifying the solver used       | 'gurobi' |
+---------+-----------------------------------------+----------+
| mipgap  | Value to define the MIP gap             |          |
+---------+-----------------------------------------+----------+
| timelim | Value to define the time limit in hours |          |
+---------+-----------------------------------------+----------+

List of energy balance settings that can be specified:

+-------------+--------------------------------------------------+---------+
| Name        | Definition                                       | Options |
+-------------+--------------------------------------------------+---------+
| violation   | Determines if the energy balance can be violated | {0,1}   |
+-------------+--------------------------------------------------+---------+
| copperplate | Determines if a copperplate approach is used     | {0,1}   |
+-------------+--------------------------------------------------+---------+

List of economic settings that can be specified:

+----------------+----------------------------------------------+---------+
| Name           | Definition                                   | Options |
+----------------+----------------------------------------------+---------+
| globalinterest | Determines if a global interest rate is used | {0,1}   |
+----------------+----------------------------------------------+---------+
| globalcosttype | Determines if a global cost function is used | {0,1}   |
+----------------+----------------------------------------------+---------+

List of technology and network performance settings that can be specified:

+----------------------+------------------------------------------------+---------+
| Name                 | Definition                                     | Options |
+----------------------+------------------------------------------------+---------+
| globalconversiontype | Determines if a global conversion type is used | {0,1}   |
+----------------------+------------------------------------------------+---------+
| dynamics             | Determines if dynamics are used                | {0,1}   |
+----------------------+------------------------------------------------+---------+

..  _configuration-example-usage:
Example Usage
^^^^^^^^^^^^^^^^
The framework includes a standard configuration of the modeling settings. You can change the standard settings as follows:

.. testcode::

    from src.model_configuration import ModelConfiguration

    # Initialize an instance of the model configuration class
    configuration = ModelConfiguration()

    # Change some settings, while maintaining other settings
    configuration.solveroptions.timelim = 10
    configuration.economic.globalinterest = 0.05

