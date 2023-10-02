import time
from pyomo.environ import *
import pandas as pd

class ModelComponent:
    """
    Class to read and manage data for technologies and networks. This class inherits its attributes to the technology
    and network classes.
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

        self.results = {}
        self.results['time_dependent'] = pd.DataFrame()
        self.results['time_independent'] = pd.DataFrame()



class Economics:
    """
    Class to manage economic data of technologies and networks
    """
    def __init__(self, economics):
        self.capex_model = economics['CAPEX_model']
        self.capex_data = {}
        if 'unit_CAPEX' in economics:
            self.capex_data['unit_capex'] = economics['unit_CAPEX']
        if 'piecewise_CAPEX' in economics:
            self.capex_data['piecewise_capex'] = economics['piecewise_CAPEX']
        if 'gamma1' in economics:
            self.capex_data['gamma1'] = economics['gamma1']
            self.capex_data['gamma2'] = economics['gamma2']
        if 'gamma3' in economics:
            self.capex_data['gamma3'] = economics['gamma3']
        self.opex_variable = economics['OPEX_variable']
        self.opex_fixed = economics['OPEX_fixed']
        self.discount_rate = economics['discount_rate']
        self.lifetime = economics['lifetime']
        self.decommission_cost = economics['decommission_cost']


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

