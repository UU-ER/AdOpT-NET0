from pyomo.util.infeasible import log_infeasible_constraints
import logging
from pyomo.core import Constraint, Var, value


def get_infeasibile_constraints(m, tolerance = 1e-3):
    """
    gets the infeasibile constraints of a pyomo model and prints them to file

    :param m: pyomo model
    :param Path save_path: path to save
    :param tolerance: tolerance of constraint violation
    :return: None
    """
    # log_infeasible_constraints(m, log_expression=False, log_variables=True, tol=tolerance)
    for constr in m.component_data_objects(ctype=Constraint, active=True, descend_into=True):
        body_value = value(constr.body, exception=False)
        # infeasible = _check_infeasible(constr, body_value, tol)
        # if infeasible:
        #     yield constr, body_value, infeasible

