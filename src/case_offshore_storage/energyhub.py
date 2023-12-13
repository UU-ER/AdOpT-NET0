from ..energyhub import EnergyHub
from pyomo.environ import *

class EnergyhubCapexOptimization(EnergyHub):
    def __init__(self, data, configuration, technology_to_optimize:tuple, total_cost_limit:float):
        super().__init__(data, configuration)
        self.technology_to_optimize = technology_to_optimize
        self.total_cost_limit = total_cost_limit

    def _optimize(self, objective):
        self._delete_objective()

        self.model.const_cost_limit = Constraint(expr=self.model.var_total_cost <= self.total_cost_limit)

        self.model.const_cost_limit.pprint()
        def init_max_capex(obj):
            return self.model.node_blocks[self.technology_to_optimize[0]].tech_blocks_active[self.technology_to_optimize[1]].var_capex
        self.model.objective = Objective(rule=init_max_capex, sense=maximize)
        if self.configuration.solveroptions.solver == 'gurobi_persistent':
            self.solver.add_constraint(self.model.const_cost_limit)
            self.solver.set_objective(self.model.objective)

        self._call_solver()

