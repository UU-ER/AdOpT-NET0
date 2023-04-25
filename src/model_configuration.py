from types import SimpleNamespace


class ModelConfiguration:
    """
    Class to specify all global modeling settings (e.g. objective, high-level algorithms, energy balance violation, \
     costs and performances) and solver configurations. The time staging algorithm is further described \
     :ref:`here <time_averaging>` and the clustering algorithm is further described :ref:`here <clustering>`.

    List of optimization settings that can be specified:

    +------------------+----------------------------------------------+---------------------------------------------+---------+
    | Name             | Definition                                   | Options                                     | Default |
    +------------------+----------------------------------------------+---------------------------------------------+---------+
    | objective        | String specifying the objective/type         | 'costs', 'emissions_pos', 'emissions_neg',  | 'costs' |
    |                  | of optimization                              | 'emissions_minC', 'pareto'                  |         |
    +------------------+----------------------------------------------+---------------------------------------------+---------+
    | montecarlo.range | Value defining the range in which variables  |                                             | 0.2     |
    |                  | are varied in Monte Carlo simulations        |                                             |         |
    +------------------+----------------------------------------------+---------------------------------------------+---------+
    | montecarlo.N     | Number of Monte Carlo simulations            |                                             | 100     |
    +------------------+----------------------------------------------+---------------------------------------------+---------+
    | pareto_points    | Number of Pareto points                      |                                             | 5       |
    +------------------+----------------------------------------------+---------------------------------------------+---------+
    | timestaging      | Defines number of daily intervals (0 = off)  |                                             | 0       |
    |                  | :ref:`check here <time_averaging>`           |                                             |         |
    +------------------+----------------------------------------------+---------------------------------------------+---------+
    | techstaging      | Switch to turn tecstaging on/off             | {0,1}                                       | 0       |
    +------------------+----------------------------------------------+---------------------------------------------+---------+
    | typicaldays      | Determines number of typical days (0 = off)  |                                             | 0       |
    |                  | :ref:`check here <clustering>`               |                                             |         |
    +------------------+----------------------------------------------+---------------------------------------------+---------+

    List of solver settings that can be specified (see also https://www.gurobi.com/documentation/9.5/refman/parameter_descriptions.html):

    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | Name          | Definition                                                          | Options                | Default  |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | solver        | String specifying the solver used                                   | 'gurobi'               | 'gurobi' |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | threads       | Value to define MIP gap                                             |                        | 0.001    |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | timelim       | Value to define time limit in hours                                 |                        | 10       |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | threads       | Value to define number of threads (default is maximum available)    |                        | 0        |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | mipfocus      | Modifies high level solution strategy                               | {0,1,2,3}              | 0        |
    +---------------+---------------------------------------------------------------------+------------------------+----------+
    | logfile       | String to define the location of the logfile                        |                        | ""       |
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
    | cuts          | Setting defining the aggressiveness of the global cut               | {-1, 0, 1, 2, 3}       | -1       |
    +---------------+---------------------------------------------------------------------+------------------------+----------+

    List of energy balance settings that can be specified:

    +-------------+--------------------------------------------------+---------+---------+
    | Name        | Definition                                       | Options | Default |
    +-------------+--------------------------------------------------+---------+---------+
    | violation   | Determines if the energy balance can be violated | {0,1}   | 0       |
    +-------------+--------------------------------------------------+---------+---------+
    | copperplate | Determines if a copperplate approach is used     | {0,1}   | 0       |
    +-------------+--------------------------------------------------+---------+---------+

    List of economic settings that can be specified:

    +----------------------------+--------------------------------------------------------+---------+---------+
    | Name                       | Definition                                             | Options | Default |
    +----------------------------+--------------------------------------------------------+---------+---------+
    | global_discountrate        | Determines if and which global discount rate is used.  |         | -1      |
    |                            | This holds for the CAPEX of all technologies and       |         |         |
    |                            | networks                                               |         |         |
    +----------------------------+--------------------------------------------------------+---------+---------+
    | global_simple_capex_model  | Determines if capex model of technologies is set to 1  | {0,1}   | 0       |
    | global_simple_capex_model  | for all technologies                                   |         |         |
    +----------------------------+--------------------------------------------------------+---------+---------+

    List of technology and network performance settings that can be specified:

    +----------------------+--------------------------------------------------------+-------------+---------+
    | Name                 | Definition                                             | Options     | Default |
    +----------------------+--------------------------------------------------------+-------------+---------+
    | dynamics             | Determines if dynamics are used                        | {0,1}       | 0       |
    +----------------------+--------------------------------------------------------+-------------+---------+
    """

    def __init__(self):
        """
        Initializer of ModelConfiguration Class
        """

        self.solveroptions = SimpleNamespace()
        self.solveroptions.solver = 'gurobi'
        self.solveroptions.mipgap = 0.001
        self.solveroptions.timelim = 10
        self.solveroptions.threads = 0
        self.solveroptions.mipfocus = 0
        self.solveroptions.logfile = ""
        self.solveroptions.nodefilestart = 0.5
        self.solveroptions.method = -1
        self.solveroptions.heuristics = 0.05
        self.solveroptions.presolve = -1
        self.solveroptions.branchdir = 0
        self.solveroptions.lpwarmstart = 0
        self.solveroptions.intfeastol = 1e-5
        self.solveroptions.cuts = -1

        self.optimization = SimpleNamespace()
        self.optimization.objective = 'costs'
        # self.optimization.montecarlo = SimpleNamespace()
        # self.optimization.montecarlo.range = 0.2
        # self.optimization.montecarlo.N = 100
        self.optimization.pareto_points = 5
        self.optimization.timestaging = 0
        # self.optimization.tecstaging = 0
        self.optimization.typicaldays = 0

        self.energybalance = SimpleNamespace()
        # self.energybalance.violation = 0
        # self.energybalance.copperplate = 0

        self.economic = SimpleNamespace()
        self.economic.global_discountrate = -1
        self.economic.global_simple_capex_model = 0

        self.performance = SimpleNamespace()
        # self.performance.dynamics = 0

    def define_montecarlo(self, range, N):
        """
        Function to define the range within the variables are varied (+/-) and the number of simulations
        for the Monte Carlo simulation.

        :param float range: SD with which investment cost is varied
        :param int N: number of simulations
        """

        self.optimization.montecarlo.range = range
        self.optimization.montecarlo.N = N
