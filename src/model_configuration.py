from types import SimpleNamespace


class ModelConfiguration:
    """
    Class to specify all global modeling settings and solver configurations.

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
        """

        self.optimization.montecarlo.range = range
        self.optimization.montecarlo.N = N
