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
    logger = logging.getLogger(__name__)

    for constr in m.component_data_objects(ctype=Constraint, active=True, descend_into=True):
        body_value = value(constr.body, exception=False)
        infeasible = 0
        infeasibility = 0
        if not body_value is None:
            if constr.has_lb():
                lb = value(constr.lower, exception=False)
                if lb is None:
                    infeasible = 4 | 1
                elif lb - body_value > tolerance:
                    infeasible = 1
                    infeasibility = infeasibility + (lb - body_value)
            if constr.has_ub():
                ub = value(constr.upper, exception=False)
                if ub is None:
                    infeasible |= 4 | 2
                elif body_value - ub > tolerance:
                    infeasible = 1
                    infeasibility = infeasibility + (body_value - ub)

        if infeasible:
            logger.info(constr.name + " is infeasible by " + str(infeasibility))

