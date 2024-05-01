.. _workflow_model-configuration:

Define Model Configuration
===========================

The ``ModelConfig.JSON`` is used to define and change the standard modeling configuration. The model configuration consists
of all global modelling settings (e.g., objective, high-level algorithms, energy balance violation, costs and performances)
and solver configurations. In the table in the Model Configuration overview below you can see all settings that can be
specified, and a description, the options that can be chosen and the default value from the template for each setting.

In the model configuration, you can also specify options to reduce the complexity of the model, such as time-staging,
clustering or scaling. For background information on the algorithms, see the following pages:

- :ref:`Time staging algorithm<time_averaging>`.
- :ref:`Clustering algorithm<clustering>`.
- :ref:`Scaling<scaling>`.


Model Configuration Overview
-----------------------------

.. csv-table:: Model Configuration Settings
   :header: "Category", "Sub-category", "Description", "Options", "Value"
   :widths: 15, 15, 40, 20, 10

   "optimization", "objective", "String specifying the objective/type of optimization.", "'costs', 'emissions_pos', 'emissions_net', 'emissions_minC', 'costs_emissionlimit', 'pareto'", "costs"
   "optimization", "monte_carlo.on", "Turn Monte Carlo simulation on.", "0, 1", 0
   "optimization", "monte_carlo.sd", "Value defining the range in which variables are varied in Monte Carlo simulations (defined as the standard deviation of the original value).", "", 0.2
   "optimization", "monte_carlo.N", "Number of Monte Carlo simulations.", "", 100
   "optimization", "monte_carlo.on_what", "List: Defines component to vary.", "'Technologies', 'ImportPrices', 'ExportPrices'", "Technologies"
   "optimization", "pareto_points", "Number of Pareto points.", "", 5
   "optimization", "timestaging", "Defines number of timesteps that are averaged (0 = off).", "", 0
   "optimization", "typicaldays.N", "Determines number of typical days (0 = off).", "", 0
   "optimization", "typicaldays.method", "Determine method used for modeling technologies with typical days.", "2", 2
   "optimization", "multiyear", "Enable multiyear analysis, if turned off max time horizon is 1 year.", "0, 1", 0
   "solveroptions", "solver", "String specifying the solver used.", "", "gurobi"
   "solveroptions", "mipgap", "Value to define MIP gap.", "", 0.001
   "solveroptions", "timelim", "Value to define time limit in hours.", "", 10
   "solveroptions", "threads", "Value to define number of threads (default is maximum available).", "", 0
   "solveroptions", "mipfocus", "Modifies high level solution strategy.", "0, 1, 2, 3", 0
   "solveroptions", "nodefilestart", "Parameter to decide when nodes are compressed and written to disk.", "", 60
   "solveroptions", "method", "Defines algorithm used to solve continuous models.", "-1, 0, 1, 2, 3, 4, 5", -1
   "solveroptions", "heuristics", "Parameter to determine amount of time spent in MIP heuristics.", "", 0.05
   "solveroptions", "presolve", "Controls the presolve level.", "-1, 0, 1, 2", -1
   "solveroptions", "branchdir", "Determines which child node is explored first in the branch-and-cut.", "-1, 0, 1", 0
   "solveroptions", "lpwarmstart", "Controls whether and how warm start information is used for LP.", "0, 1, 2", 0
   "solveroptions", "intfeastol", "Value that determines the integer feasibility tolerance.", "", 1e-05
   "solveroptions", "feastol", "Value that determines feasibility for all constraints.", "", 1e-06
   "solveroptions", "numericfocus", "Degree of which Gurobi tries to detect and manage numeric issues.", "0, 1, 2, 3", 0
   "solveroptions", "cuts", "Setting defining the aggressiveness of the global cut.", "-1, 0, 1, 2, 3", -1
   "reporting", "save_detailed", "Setting to select how the results are saved. When turned off only the summary is saved.", "0, 1", 1
   "reporting", "save_path", "Option to define the save path.", "", "./userData/"
   "reporting", "case_name", "Option to define a case study name that is added to the results folder name.", "", -1
   "reporting", "write_solution_diagnostics", "If 1, writes solution quality, if 2 also writes pyomo to Gurobi variable map and constraint map to file.", "0, 1, 2", 0
   "energybalance", "violation", "Determines the energy balance violation price (-1 is no violation allowed).", "", -1
   "energybalance", "copperplate", "Determines if a copperplate approach is used.", "0, 1", 0
   "economic", "global_discountrate", "Determines if and which global discount rate is used. This holds for the CAPEX of all technologies and networks.", "", -1
   "economic", "global_simple_capex_model", "Determines if the CAPEX model of technologies is set to 1 for all technologies.", "0, 1", 0
   "performance", "dynamics", "Determines if dynamics are used.", "0, 1", 0

