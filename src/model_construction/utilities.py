from pyomo.gdp import *
from pyomo.environ import *
import time
import numpy as np
import src.global_variables as global_variables


def perform_disjunct_relaxation(component):
    """
    Performs big-m transformation for respective component
    :param component: component
    :return: component
    """
    print('\t\tBig-M Transformation...')
    start = time.time()
    xfrm = TransformationFactory('gdp.bigm')
    xfrm.apply_to(component)
    global_variables.big_m_transformation_required = 0
    print('\t\tBig-M Transformation completed in ' + str(round(time.time() - start)) + ' s')
    return component



