import pandas as pd
import numpy as np
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
        # self.solveroptions.solver = 'gurobi'
        # self.solveroptions.mipgap = 0.01
        # self.solveroptions.timelim = 10

        self.optimization = SimpleNamespace()
        self.optimization.objective = 'costs'
        # self.optimization.pareto = SimpleNamespace()
        # self.optimization.pareto.fracSD = 0.2
        # self.optimization.pareto.N = 100
        # self.optimization.timestaging = 0
        # self.optimization.tecstaging = 0

        self.energybalance = SimpleNamespace()
        # self.energybalance.violation = 0
        # self.energybalance.copperplate = 0

        self.economic = SimpleNamespace()
        # self.economic.globalinterest = 0
        # self.economic.globalPWA = 0

        self.performance = SimpleNamespace()
        # self.performance.globalconversiontype = 0
        # self.performance.dynamics = 0

        # self.__big_m_transformation_required = 0
        # self.__clustered_data = 0
        # self.__clustered_data_specs = SimpleNamespace()
        # self.__clustered_data_specs.specs = []
        # self.__averaged_data = 0
        # self.__averaged_data_specs = SimpleNamespace()
        # self.__averaged_data_specs.nr_timesteps_averaged = 1
        # self.__averaged_data_specs.specs = []

    def define_pareto(self, fracSD, N):
        self.optimization.pareto.fracSD = fracSD
        self.optimization.pareto.N = N
