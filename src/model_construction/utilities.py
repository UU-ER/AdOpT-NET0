from pyomo.gdp import *
import time
import src.config_model as m_config
from pyomo.environ import *


def perform_disjunct_relaxation(component):
    print('Big-M Transformation...')
    start = time.time()
    xfrm = TransformationFactory('gdp.bigm')
    xfrm.apply_to(component)
    m_config.presolve.big_m_transformation_required = 0
    print('Big-M Transformation completed in ' + str(time.time() - start) + ' s')
    return component