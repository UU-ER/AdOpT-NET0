from pyomo.util.infeasible import log_infeasible_constraints
import logging


def get_infeasibile_constraints(m, tolerance = 1e-3):
    """
    gets the infeasibile constraints of a pyomo model and prints them to file

    :param m: pyomo model
    :param Path save_path: path to save
    :param tolerance: tolerance of constraint violation
    :return: None
    """
    log_infeasible_constraints(m, log_expression=False, log_variables=True, tol=tolerance)

