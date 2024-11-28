..   _model_configuration:

=====================================
Model Configuration
=====================================

Here are the configuration settings as can be :ref:`defined<workflow_model-configuration>` in ``ConfigModel.json``,
along with a description of each setting, the options that can be chosen for the setting (if relevant) and the value of
the default/currently chosen option.

The ``objective`` options refer to:

- 'costs': minimize total annual system cost
- 'emissions_pos': minimize postive emissions (negative emissions are not part of the
  objective function)
- 'emissions_net': minimize net emissions (sum of positive and negative emissions)
- 'emissions_minC': find the minimum cost system at minimum emissions (minimizes net
  emissions in the first step and cost as a second step)
- 'costs_emissionlimit': minimize cost at an emission limit. The emission limit can
  be set using the ``emission_limit`` option
- 'pareto': optimizes the pareto front, documented :ref:`here<pareto>`

.. csv-table:: Model Configuration Settings
   :file: config.csv
   :header: "Category", "Setting", "Sub-setting", "Description", "Options", "Value"
   :widths: 15, 15, 15, 40, 20, 10
