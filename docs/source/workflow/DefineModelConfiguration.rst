.. _workflow_model-configuration:

Define Model Configuration
===========================

The ``ModelConfig.JSON`` is used to define and change the standard modeling configuration. The model configuration consists
of all global modelling settings (e.g., objective, high-level algorithms, energy balance violation, costs and performances)
and solver configurations. In the tables below you can see all settings that can be specified, and a description, the
options that can be chosen and the default value from the template for each setting.

In the model configuration, you can also specify options to reduce the complexity of the model, such as time-staging,
clustering or scaling. For the background of the algorithms, see the following pages:

- :ref:`Time staging algorithm<time_averaging>`.
- :ref:`Clustering algorithm<clustering>`.
- :ref:`Scaling<scaling>`.


.. rst-class::  wide

    List of optimization settings that can be specified:

    +--------------------+----------------------------------------------+---------------------------------------------+---------+
    | Name               | Description                                  | Options                                     | Default |
    +--------------------+----------------------------------------------+---------------------------------------------+---------+
    | objective          | String specifying the objective/type         | 'costs', 'emissions_pos', 'emissions_net',  | 'costs' |
    |                    | of optimization                              | 'emissions_minC', 'costs_emissionlimit'     |         |
    |                    |                                              | 'pareto',                                   |         |
    +--------------------+----------------------------------------------+---------------------------------------------+---------+
    | monte_carlo.on     | Turn monte carlo simulation on               | 1,0                                         | 0       |
    +--------------------+----------------------------------------------+---------------------------------------------+---------+
    | monte_carlo.sd     | Value defining the range in which variables  |                                             | 0.2     |
    |                    | are varied in Monte Carlo simulations        |                                             |         |
    |                    | (defined as the standard deviation of the    |                                             |         |
    |                    | original value)                              |                                             |         |
    +--------------------+----------------------------------------------+---------------------------------------------+---------+
    | monte_carlo.N      | Number of Monte Carlo simulations            |                                             | 100     |
    +--------------------+----------------------------------------------+---------------------------------------------+---------+
    | monte_carlo.on_what| List: Defines component to vary.             | 'Technologies', 'ImportPrices',             | Tec     |
    |                    | Warning: Import/export prices                | 'ExportPrices'                              |         |
    |                    | can take a long time.                        |                                             |         |
    +--------------------+----------------------------------------------+---------------------------------------------+---------+
    | pareto_points      | Number of Pareto points                      |                                             | 5       |
    +--------------------+----------------------------------------------+---------------------------------------------+---------+
    | timestaging        | Defines number of timesteps that are averaged|                                             | 0       |
    |                    | (0 = off) :ref:`check here <time_averaging>` |                                             |         |
    +--------------------+----------------------------------------------+---------------------------------------------+---------+
    | typicaldays.N      | Determines number of typical days (0 = off)  |                                             | 0       |
    |                    | :ref:`check here <clustering>`               |                                             |         |
    +--------------------+----------------------------------------------+---------------------------------------------+---------+
    | typicaldays.method | Determine method used for modeling           | {2}                                         | 2       |
    |                    | technologies with typical days               |                                             |         |
    +--------------------+----------------------------------------------+---------------------------------------------+---------+

    List of solver settings that can be specified (see also https://www.gurobi.com/documentation/9.5/refman/parameter_descriptions.html):

    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | Name          | Description                                                         | Options                | Default  |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | solver        | String specifying the solver used                                   | 'gurobi'               | 'gurobi' |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | mipgap        | Value to define MIP gap                                             |                        | 0.001    |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | timelim       | Value to define time limit in hours                                 |                        | 10       |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | threads       | Value to define number of threads (default is maximum available)    |                        | 0        |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | mipfocus      | Modifies high level solution strategy                               | {0,1,2,3}              | 0        |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | nodefilestart | Parameter to decide when nodes are compressed and written to disk   |                        | 60       |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | method        | Defines algorithm used to solve continuous models                   | {-1, 0, 1, 2, 3, 4, 5} | -1       |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | heuristics    | Parameter to determine amount of time spend in MIP heuristics       |                        | 0.05     |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | presolve      | Controls the presolve level                                         | {-1, 0, 1, 2}          | -1       |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | branchdir     | Determines which child node is explored first in the branch-and-cut | {-1, 0, 1}             | 0        |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | lpwarmstart   | Controls whether and how warm start information is used for LP      | {0, 1, 2}              | 0        |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | intfeastol    | Value that determines the integer feasibility tolerance             |                        | 1e-5     |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | feastol       | Value that determines feasibility for all constraints               |                        | 1e-6     |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | numericfocus  | Degree of which gurobi tries to detect and manage numeric issues    | {0, 1, 2, 3}           | 0        |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | cuts          | Setting defining the aggressiveness of the global cut               | {-1, 0, 1, 2, 3}       | -1       |
    +---------------+---------------------------------------------------------------------+------------------------+----------+

    List of reporting settings that can be specified:

    +----------------------------+--------------------------------------------------------+-------------+-------------+
    | Name                       | Description                                            | Options     | Default     |
    +----------------------------+--------------------------------------------------------+-------------+-------------+
    | save_detailed              | Setting to select how the results are saved. When      | {0,1}       | 1           |
    |                            | turned off only the summary is saved.                  |             |             |
    +----------------------------+--------------------------------------------------------+-------------+-------------+
    | save_path                  | Option to define the save path.                        |             |'./userData/'|
    +----------------------------+--------------------------------------------------------+-------------+-------------+
    | case_name                  | Option to define a case study name that is added to    |{str of name,| -1          |
    |                            | the results folder name.                               |     -1}     |             |
    +----------------------------+--------------------------------------------------------+-------------+-------------+
    | write_solution_diagnostics | If 1, writes solution quality, if 2 also writes pyomo  |{0,1,2}      | 0           |
    |                            | to gurobi variable map and constraint map to file.     |             |             |
    +----------------------------+--------------------------------------------------------+-------------+-------------+

    List of energy balance settings that can be specified:

    +-------------+----------------------------------------------------------------------------+---------+---------+
    | Name        | Description                                                                | Options | Default |
    +-------------+----------------------------------------------------------------------------+---------+---------+
    | violation   | Determines the energy balance violation price (-1 is no violation allowed) |         | -1      |
    +-------------+----------------------------------------------------------------------------+---------+---------+
    | copperplate | Determines if a copperplate approach is used                               | {0,1}   | 0       |
    +-------------+----------------------------------------------------------------------------+---------+---------+

    List of economic settings that can be specified:

    +----------------------------+--------------------------------------------------------+---------+---------+
    | Name                       | Description                                            | Options | Default |
    +----------------------------+--------------------------------------------------------+---------+---------+
    | global_discountrate        | Determines if and which global discount rate is used.  |         | -1      |
    |                            | This holds for the CAPEX of all technologies and       |         |         |
    |                            | networks                                               |         |         |
    +----------------------------+--------------------------------------------------------+---------+---------+
    | global_simple_capex_model  | Determines if capex model of technologies is set to 1  | {0,1}   | 0       |
    |                            | for all technologies                                   |         |         |
    +----------------------------+--------------------------------------------------------+---------+---------+

    List of technology and network performance settings that can be specified:

    +------------+---------------------------------------------------------------------------------------------+---------+---------+
    | Name       | Description                                                                                 | Options | Default |
    +------------+---------------------------------------------------------------------------------------------+---------+---------+
    | dynamics   | Determines if dynamics are used                                                             | {0,1}   | 0       |
    +------------+---------------------------------------------------------------------------------------------+---------+---------+

    List of scaling options (see also: :ref:`here <scaling>`):

    +------------+---------------------------------------------------------------------------------------------+---------+---------+
    | Name       | Description                                                                                 | Options | Default |
    +------------+---------------------------------------------------------------------------------------------+---------+---------+
    | scaling    | Determines if model is scaled. If 1, it uses global and component specific scaling factors  | {0,1}   | 0       |
    +------------+---------------------------------------------------------------------------------------------+---------+---------+
    | energy_vars| Scaling factor used for all energy variables                                                |         | 1e-3    |
    +------------+---------------------------------------------------------------------------------------------+---------+---------+
    | cost_vars  | Scaling factor used for all cost  variables                                                 |         | 1e-3    |
    +------------+---------------------------------------------------------------------------------------------+---------+---------+
    | objective  | Scaling factor used for objective function                                                  |         | 1       |
    +------------+---------------------------------------------------------------------------------------------+---------+---------+




