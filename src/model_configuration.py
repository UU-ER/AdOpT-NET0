from types import SimpleNamespace


class ModelConfiguration:
    """
    Class to specify all global modeling settings (e.g. objective, high-level algorithms, energy balance violation, \
     costs and performances) and solver configurations.

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
    | pareto.N         | Number of Pareto points                      |                                             | 5       |
    +------------------+----------------------------------------------+---------------------------------------------+---------+
    | timestaging      | Switch to turn timestaging on/off            | {0,1}                                       | 0       |
    +------------------+----------------------------------------------+---------------------------------------------+---------+
    | techstaging      | Switch to turn tecstaging on/off             | {0,1}                                       | 0       |
    +------------------+----------------------------------------------+---------------------------------------------+---------+

    List of solver settings that can be specified:

    +---------+-------------------------------------+----------+----------+
    | Name    | Definition                          | Options  | Default  |
    +---------+-------------------------------------+----------+----------+
    | solver  | String specifying the solver used   | 'gurobi' | 'gurobi' |
    +---------+-------------------------------------+----------+----------+
    | mipgap  | Value to define MIP gap             |          | 0.01     |
    +---------+-------------------------------------+----------+----------+
    | timelim | Value to define time limit in hours |          | 10       |
    +---------+-------------------------------------+----------+----------+

    List of energy balance settings that can be specified:

    +-------------+--------------------------------------------------+---------+---------+
    | Name        | Definition                                       | Options | Default |
    +-------------+--------------------------------------------------+---------+---------+
    | violation   | Determines if the energy balance can be violated | {0,1}   | 0       |
    +-------------+--------------------------------------------------+---------+---------+
    | copperplate | Determines if a copperplate approach is used     | {0,1}   | 0       |
    +-------------+--------------------------------------------------+---------+---------+

    List of economic settings that can be specified:

    +----------------+----------------------------------------------+---------+---------+
    | Name           | Definition                                   | Options | Default |
    +----------------+----------------------------------------------+---------+---------+
    | globalinterest | Determines if a global interest rate is used | {0,1}   | 0       |
    +----------------+----------------------------------------------+---------+---------+
    | globalcosttype | Determines if a global cost function is used | {0,1}   | 0       |
    +----------------+----------------------------------------------+---------+---------+

    List of technology and network performance settings that can be specified:

    +----------------------+------------------------------------------------+---------+---------+
    | Name                 | Definition                                     | Options | Default |
    +----------------------+------------------------------------------------+---------+---------+
    | globalconversiontype | Determines if a global conversion type is used | {0,1}   | 0       |
    +----------------------+------------------------------------------------+---------+---------+
    | dynamics             | Determines if dynamics are used                | {0,1}   | 0       |
    +----------------------+------------------------------------------------+---------+---------+
    """

    def __init__(self):
        """
        Initializer of ModelConfiguration Class
        """

        self.solveroptions = SimpleNamespace()
        self.solveroptions.solver = 'gurobi'
        # self.solveroptions.mipgap = 0.01
        # self.solveroptions.timelim = 10

        self.optimization = SimpleNamespace()
        self.optimization.objective = 'costs'
        # self.optimization.montecarlo = SimpleNamespace()
        # self.optimization.montecarlo.range = 0.2
        # self.optimization.montecarlo.N = 100
        # self.optimization.pareto.N = 5
        # self.optimization.timestaging = 0
        # self.optimization.tecstaging = 0

        self.energybalance = SimpleNamespace()
        # self.energybalance.violation = 0
        # self.energybalance.copperplate = 0

        self.economic = SimpleNamespace()
        # self.economic.globalinterest = 0
        # self.economic.globalcosttype = 0

        self.performance = SimpleNamespace()
        # self.performance.globalconversiontype = 0
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
