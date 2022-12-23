import pytest
import src.data_management as dm
from src.energyhub import EnergyHub as ehub
from pyomo.environ import units as u
from pyomo.environ import *
import pandas as pd
import src.config_model as m_config

def test_networks():
    """
    Creates dataset for test_network().
    import electricity @ node 1
    electricity demand @ node 2
    """
    # Test bidirectional
    data = dm.load_data_handle(r'./test/test_data/networks.p')
    data.network_data['hydrogenTest']['NetworkPerf']['bidirectional'] = 1
    data.network_data['hydrogenTest']['EnergyConsumption'] = {}
    energyhub1 = ehub(data)
    energyhub1.construct_model()
    energyhub1.construct_balances()
    energyhub1.solve_model()
    cost1 = energyhub1.model.objective()
    assert energyhub1.solution.solver.termination_condition == 'optimal'
    # is network size double the demand (because of losses)
    should = 20
    res = energyhub1.model.network_block['hydrogenTest'].arc_block['test_node1', 'test_node2'].var_size.value
    assert abs(should - res) / res <= 0.001
    # is capex correct
    should = 1020
    res = energyhub1.model.network_block['hydrogenTest'].var_cost.value
    assert abs(should - res) / res <= 0.001

    # Test no bidirectional
    data = dm.load_data_handle(r'./test/test_data/networks.p')
    data.network_data['hydrogenTest']['NetworkPerf']['bidirectional'] = 0
    data.network_data['hydrogenTest']['EnergyConsumption'] = {}
    energyhub2 = ehub(data)
    energyhub2.construct_model()
    energyhub2.construct_balances()
    energyhub2.solve_model()
    cost2 = energyhub2.model.objective()
    assert energyhub2.solution.solver.termination_condition == 'optimal'
    # is network size double the demand (because of losses)
    should = 20
    res = energyhub1.model.network_block['hydrogenTest'].arc_block['test_node1', 'test_node2'].var_size.value
    assert abs(should - res) / res <= 0.001

    # Test consumption at node
    data = dm.load_data_handle(r'./test/test_data/networks.p')
    data.network_data['hydrogenTest']['NetworkPerf']['bidirectional'] = 0
    energyhub3 = ehub(data)
    energyhub3.construct_model()
    energyhub3.construct_balances()
    energyhub3.solve_model()
    cost3 = energyhub3.model.objective()
    assert energyhub3.solution.solver.termination_condition == 'optimal'
    # is network size double the demand (because of losses)
    should = 20
    res = energyhub3.model.network_block['hydrogenTest'].arc_block['test_node1', 'test_node2'].var_size.value
    assert abs(should - res) / res <= 0.001
    # is import of electricity there?
    should = 20
    res = energyhub3.model.node_blocks['test_node1'].var_import_flow[1, 'electricity'].value
    assert abs(should - res) / res <= 0.001
    res = energyhub3.model.node_blocks['test_node2'].var_import_flow[2, 'electricity'].value
    assert abs(should - res) / res <= 0.001
    # Is objective correct
    should = 2440
    res = energyhub3.model.objective()
    assert abs(should - res) / res <= 0.001

    # does bidirectional produce double costs?
    assert abs(cost2 / cost1 - 2) <= 0.001
