from pyomo.environ import *
from types import SimpleNamespace


def get_gurobi_parameters(solveroptions):
    solver = SolverFactory(solveroptions.solver, solver_io="python")
    solver.options["TimeLimit"] = solveroptions.timelim * 3600
    solver.options["MIPGap"] = solveroptions.mipgap
    solver.options["MIPFocus"] = solveroptions.mipfocus
    solver.options["Threads"] = solveroptions.threads
    solver.options["NodefileStart"] = solveroptions.nodefilestart
    solver.options["Method"] = solveroptions.method
    solver.options["Heuristics"] = solveroptions.heuristics
    solver.options["Presolve"] = solveroptions.presolve
    solver.options["BranchDir"] = solveroptions.branchdir
    solver.options["LPWarmStart"] = solveroptions.lpwarmstart
    solver.options["IntFeasTol"] = solveroptions.intfeastol
    solver.options["FeasibilityTol"] = solveroptions.feastol
    solver.options["Cuts"] = solveroptions.cuts
    solver.options["NumericFocus"] = solveroptions.numericfocus

    return solver


def get_glpk_parameters(solveroptions):
    solver = SolverFactory("glpk")

    return solver
