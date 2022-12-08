import pytest
import src.data_management as dm
from src.energyhub import energyhub as ehub
from pyomo.environ import units as u
from pyomo.environ import *
import pandas as pd



def test_networks():
    """
    Creates dataset for test_network().
    import electricity @ node 1
    electricity demand @ node 2
    """
    # Test bidirectional
    data = dm.load_data_handle(r'./test/test_data/networks.p')
    data.network_data['electricityTest']['NetworkPerf']['bidirectional'] = 1
    energyhub1 = ehub(data)
    energyhub1.construct_model()
    xfrm = TransformationFactory('gdp.bigm')
    xfrm.apply_to(energyhub1.model)
    solver = SolverFactory('gurobi')
    solution = solver.solve(energyhub1.model, tee=True)
    cost1 = energyhub1.model.objective()

    # Test no bidirectional
    data = dm.load_data_handle(r'./test/test_data/networks.p')
    data.network_data['electricityTest']['NetworkPerf']['bidirectional'] = 0
    energyhub2 = ehub(data)
    energyhub2.construct_model()
    xfrm = TransformationFactory('gdp.bigm')
    xfrm.apply_to(energyhub2.model)
    solver = SolverFactory('gurobi')
    solution = solver.solve(energyhub2.model, tee=True)
    cost2 = energyhub2.model.objective()

    assert solution.solver.termination_condition == 'optimal'
    # does bidirectional produce double costs?
    assert abs(cost2 / cost1 - 2) <= 0.001
    # is network size double the demand (because of losses)
    should = 20
    res = energyhub1.model.network_block['electricityTest'].arc_block['test_node1', 'test_node2'].var_size.value
    error = abs(should - res) / res
    assert error <= 0.001
    # is capex correct
    should = 1020
    res = energyhub1.model.network_block['electricityTest'].var_cost.value
    error = abs(should - res) / res
    assert error <= 0.001
