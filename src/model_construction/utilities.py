from pyomo.gdp import *
from pyomo.environ import *
from pyomo.environ import units as u
import time
import numpy as np
import src.global_variables as global_variables


def perform_disjunct_relaxation(component):
    """
    Performs big-m transformation for respective component
    :param component: component
    :return: component
    """
    print('Big-M Transformation...')
    start = time.time()
    xfrm = TransformationFactory('gdp.bigm')
    xfrm.apply_to(component)
    global_variables.big_m_transformation_required = 0
    print('Big-M Transformation completed in ' + str(time.time() - start) + ' s')
    return component

def annualize(r, t):
    """
    Calculates annualization factor
    :param r: interest rate
    :param t: lifetime
    :return: annualization factor
    """
    if r==0:
        annualization_factor = 1/t
    else:
        annualization_factor = r / (1 - (1 / (1 + r) ** t))
    return annualization_factor


def define_clustered_constraints(b_tec, set_t, sequence):
    """
    Links input and output reduced to full resolution for technologies
    """
    # Link clustered data with full resolution
    b_tec.const_link_full_resolution_input = link_full_resolution_to_clustered(b_tec.var_input,
                                                                               input,
                                                                               set_t,
                                                                               b_tec.set_input_carriers,
                                                                               sequence)
    b_tec.const_link_full_resolution_output = link_full_resolution_to_clustered(b_tec.var_output,
                                                                               output,
                                                                               set_t,
                                                                               b_tec.set_output_carriers,
                                                                               sequence)

    # def init_link_full_resolution_output(const, t, car):
    #     return b_tec.var_output[t, car] \
    #            == output[sequence[t - 1], car]
    # b_tec.const_link_full_resolution_output = Constraint(set_t, b_tec.set_output_carriers,
    #                                                      rule=init_link_full_resolution_output)

    return b_tec


def link_full_resolution_to_clustered(var_clustered, var_full, set_t, set_car, sequence):
    """
    Links two variables (clustered and full)
    """
    def init_link_full_resolution(const, t, car):
        return var_full[t, car] \
               == var_clustered[sequence[t - 1], car]
    constraint = Constraint(set_t, set_car, rule=init_link_full_resolution)

    return constraint