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

    def define_solver(self, solver):
        self.solveroptions.solver = solver

    def define_mipgap(self, mipgap):
        self.solveroptions.mipgap = mipgap

    def define_timelim(self, timelim):
        self.solveroptions.timelim = timelim

    def define_objective(self, objective):
        self.optimization.objective = objective

    def define_pareto(self, fracSD, N):
        self.optimization.pareto.fracSD = fracSD
        self.optimization.pareto.N = N

    def define_timestaging(self, timestagingON):
        self.optimization.timestaging = timestagingON

    def define_tecstaging(self, tecstagingON):
        self.optimization.tecstaging = tecstagingON

    def define_violation(self, violationON):
        self.energybalance.violation = violationON

    def define_copperplate(self, copperplateON):
        self.energybalance.copperplate = copperplateON

    def define_globalinterest(self, globalinterestON):
        self.economic.globalinterest = globalinterestON

    def define_globalPWA(self, globalPWA):
        self.economic.globalPWA = globalPWA

    def define_conversiontype(self, conversiontype):
        self.performance.globalconversiontype = conversiontype

    def define_dynamics(self, dynamicsON):
        self.performance.dynamics = dynamicsON