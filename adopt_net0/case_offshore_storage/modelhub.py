from ..modelhub import ModelHub
from .handle_input_data import DataHandleCapexOptimization
from pyomo.environ import *

class ModelHubCapexOptimization(ModelHub):
    def __init__(self, technology_to_optimize:tuple, total_cost_limit:float):
        super().__init__()
        self.data = DataHandleCapexOptimization(technology_to_optimize)
        self.technology_to_optimize = technology_to_optimize
        self.total_cost_limit = total_cost_limit

    def _optimize(self, objective):
        model = self.model[self.info_solving_algorithms["aggregation_model"]]
        config = self.data.model_config

        self._delete_objective()

        if self.total_cost_limit:
            try:
                model.del_component(model.const_cost_limit)
            except:
                pass

            model.const_cost_limit = Constraint(expr=model.var_npv <=
                                                     self.total_cost_limit*1.00001)

            model.const_cost_limit.pprint()
            def init_max_capex(obj):
                return model.periods["period1"].node_blocks[self.technology_to_optimize[
                    0]].tech_blocks_active[self.technology_to_optimize[1]].var_capex
            model.objective = Objective(rule=init_max_capex, sense=maximize)
            if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
                self.solver.add_constraint(model.const_cost_limit)
                self.solver.set_objective(model.objective)
        else:
            def init_cost_objective(obj):
                return model.var_npv
            model.objective = Objective(rule=init_cost_objective, sense=minimize)

        self._call_solver()



class ModelHubEmissionOptimization(ModelHub):
    def __init__(self, technology_to_optimize:tuple, total_emission_limit:float):
        super().__init__()
        self.technology_to_optimize = technology_to_optimize
        self.total_emission_limit = total_emission_limit

    def _optimize(self, objective):
        model = self.model[self.info_solving_algorithms["aggregation_model"]]
        config = self.data.model_config

        self._delete_objective()

        try:
            model.del_component(model.const_emission_limit)
        except:
            pass

        model.const_emission_limit = Constraint(expr=model.var_emissions_net <=
                                                 self.total_emission_limit * 1.001)

        model.const_emission_limit.pprint()
        def init_min_size(obj):
            return model.periods["period1"].node_blocks[self.technology_to_optimize[
                0]].tech_blocks_active[self.technology_to_optimize[1]].var_size
        model.objective = Objective(rule=init_min_size, sense=minimize)
        if config["solveroptions"]["solver"]["value"] == "gurobi_persistent":
            self.solver.add_constraint(model.const_emission_limit)
            self.solver.set_objective(model.objective)

        self._call_solver()

