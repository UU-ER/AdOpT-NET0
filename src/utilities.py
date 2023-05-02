from pyomo.environ import *
from pyomo.environ import units as u
import pint
import numpy as np
from src.model_construction import annualize, set_capex_model, set_discount_rate

def get_gurobi_parameters(solveroptions):
    solver = SolverFactory(solveroptions.solver)
    solver.options['TimeLimit'] = solveroptions.timelim * 3600
    solver.options['MIPGap'] = solveroptions.mipgap
    solver.options['MIPFocus'] = solveroptions.mipfocus
    solver.options['Threads'] = solveroptions.threads
    solver.options['LogFile'] = solveroptions.logfile
    solver.options['NodefileStart'] = solveroptions.nodefilestart
    solver.options['Method'] = solveroptions.method
    solver.options['Heuristics'] = solveroptions.heuristics
    solver.options['Presolve'] = solveroptions.presolve
    solver.options['BranchDir'] = solveroptions.branchdir
    solver.options['LPWarmStart'] = solveroptions.lpwarmstart
    solver.options['IntFeasTol'] = solveroptions.intfeastol
    solver.options['Cuts'] = solveroptions.cuts
    return solver

def define_units():
    try:
        u.load_definitions_from_strings(['EUR = [currency]'])
    except pint.errors.DefinitionSyntaxError:
        pass