import numpy as np
from pyomo.gdp import *
import time
from pyomo.environ import *
import src.global_variables as global_variables
import numpy as np


def perform_disjunct_relaxation(component):
    print('Big-M Transformation...')
    start = time.time()
    xfrm = TransformationFactory('gdp.bigm')
    xfrm.apply_to(component)
    global_variables.big_m_transformation_required = 0
    print('Big-M Transformation completed in ' + str(time.time() - start) + ' s')
    return component

def annualize(r, t):
    if r==0:
        annualization_factor = 1/t
    else:
        annualization_factor = r / (1 - (1 / (1 + r) ** t))
    return annualization_factor
