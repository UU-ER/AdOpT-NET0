from ..components.technologies import *
from pyomo.environ import *

from ..components.utilities import annualize, set_discount_rate
from .technology import TechnologyCapexOptimization

class Conv1(TechnologyCapexOptimization, Conv1):
    pass

class Conv2(TechnologyCapexOptimization, Conv2):
    pass

class Conv3(TechnologyCapexOptimization, Conv3):
    pass

class Conv4(TechnologyCapexOptimization, Conv4):
    pass

class Stor(TechnologyCapexOptimization, Stor):
    pass

class OceanBattery(TechnologyCapexOptimization, OceanBattery):
    pass
