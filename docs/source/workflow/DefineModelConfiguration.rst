.. _workflow_model-configuration:

===========================
5 Define Model Configuration
===========================

The ``ModelConfig.JSON`` is used to define and change the standard modeling configuration. The model configuration consists
of all global modelling settings (e.g., objective, high-level algorithms, energy balance violation, costs and performances)
and solver configurations. In the table :ref:`here<model_configuration>`here you can
find all
settings that can be
specified, and a description, the options that can be chosen and the default value from the template for each setting.

In the model configuration file you can select a specific type of analysis, such as a Pareto or Monte Carlo analysis. For
background information on these analyses, see the following pages:

- :ref:`Pareto analysis<pareto>`
- :ref:`Monte Carlo analysis<monte_carlo>`

Furthermore, you can specify options to reduce the complexity of the model, such as time-staging,
clustering or scaling. For background information on the algorithms, see
:ref:`here<time_aggregation>`.