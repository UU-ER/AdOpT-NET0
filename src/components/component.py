from .utilities import Economics
import time
from pyomo.environ import *

class ModelComponent:
    """
    Class to read and manage data for technologies
    """
    def __init__(self, data):
        """
        Initializes component class
        """
        self.name = data['name']
        self.existing = 0
        self.size_initial = []
        self.size_is_int = data['size_is_int']
        self.size_min = data['size_min']
        self.size_max = data['size_max']
        self.decommission = data['decommission']
        self.economics = Economics(data['Economics'])
        self.big_m_transformation_required = 0


def perform_disjunct_relaxation(model_block, method = 'gdp.bigm'):
    """
    Performs big-m transformation for respective component
    :param component: component
    :return: component
    """
    print('\t\tBig-M Transformation...')
    start = time.time()
    xfrm = TransformationFactory(method)
    xfrm.apply_to(model_block)
    print('\t\tBig-M Transformation completed in ' + str(round(time.time() - start)) + ' s')
    return model_block