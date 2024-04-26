from pyomo.environ import SolverFactory
from .logger import logger


def get_gurobi_parameters(solveroptions):
    solver = SolverFactory(solveroptions["solver"]["value"], solver_io="python")
    solver.options["TimeLimit"] = solveroptions["timelim"]["value"] * 3600
    solver.options["MIPGap"] = solveroptions["mipgap"]["value"]
    solver.options["MIPFocus"] = solveroptions["mipfocus"]["value"]
    solver.options["Threads"] = solveroptions["threads"]["value"]
    solver.options["NodefileStart"] = solveroptions["nodefilestart"]["value"]
    solver.options["Method"] = solveroptions["method"]["value"]
    solver.options["Heuristics"] = solveroptions["heuristics"]["value"]
    solver.options["Presolve"] = solveroptions["presolve"]["value"]
    solver.options["BranchDir"] = solveroptions["branchdir"]["value"]
    solver.options["LPWarmStart"] = solveroptions["lpwarmstart"]["value"]
    solver.options["IntFeasTol"] = solveroptions["intfeastol"]["value"]
    solver.options["FeasibilityTol"] = solveroptions["feastol"]["value"]
    solver.options["Cuts"] = solveroptions["cuts"]["value"]
    solver.options["NumericFocus"] = solveroptions["numericfocus"]["value"]

    return solver


def get_glpk_parameters(solveroptions):
    solver = SolverFactory("glpk")

    return solver


def log_event(message: str, print_it: int = 1, level: str = "info") -> None:
    """
    Logs and prints a message
    :param str message: Message to log
    :param int print_it: [0,1] if message should also be printed
    :param str level: ['info', 'warning'] which level to log
    """
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning((message))
    if print_it:
        print(message)
