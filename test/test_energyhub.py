import pytest
import src.data_management as dm
from src.energyhub import energyhub as ehub
from pyomo.environ import units as u
from pyomo.environ import *
import pandas as pd

u.load_definitions_from_strings(['EUR = [currency]'])

def test_initializer():
    data = dm.load_data_handle(r'./test/test_data/data_handle_test.p')
    energyhub = ehub(data)

def test_add_nodes():
    """
    Add a node with no technology, establishes energybalance
    """
    data = dm.load_data_handle(r'./test/test_data/data_handle_test.p')
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'


def test_model1():
    """
    Run a model with two nodes.
    PV @ node 2
    electricity demand @ node 1
    electricity network in between
    should be infeasible
    """
    data = dm.load_data_handle(r'./test/test_data/model1.p')
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'

def test_model2():
    """
    Run a model with one node.
    Furnace_NG, heat demand of 10
    Results should be:
    - Size of Furnace_NG: 10.01
    - Gas Import in each timestep: 10.01
    - Total costs: 10.01 * unit cost Furnace_NG + Import costs of NG
    """
    data = dm.load_data_handle(r'./test/test_data/model2.p')
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    m = energyhub.model
    assert energyhub.solution.solver.termination_condition == 'optimal'
    # Size of Furnace
    size_res = m.node_blocks['test_node1'].tech_blocks_active['Furnace_NG'].var_size.value
    size_should = max(data.node_data['test_node1']['demand']['heat']) / data.technology_data['test_node1']['Furnace_NG']['fit']['heat']['alpha1']
    assert  round(size_res,3) == round(size_should,3)
    # Gas Import in each timestep
    import_res = [value(m.node_blocks['test_node1'].var_import_flow[key, 'gas'].value) for key in m.set_t]
    import_res = pd.Series(import_res)
    import_res = import_res.tolist()
    import_should = data.node_data['test_node1']['demand']['heat'] / data.technology_data['test_node1']['Furnace_NG']['fit']['heat']['alpha1']
    import_should = import_should.tolist()
    assert [round(num,3) for num in import_res] == [round(num,3) for num in import_should]
    # Total cost
    cost_res = m.objective()
    import_price = data.node_data['test_node1']['import_prices']['gas'].tolist()
    import_cost = sum([i1 * i2 for i1, i2 in zip(import_price, import_res)])
    capex = data.technology_data['test_node1']['Furnace_NG']['Economics']['unit_CAPEX_annual'] * size_res
    opex_fix = capex * data.technology_data['test_node1']['Furnace_NG']['Economics']['OPEX_fixed']
    opex_var = sum(import_res) * data.technology_data['test_node1']['Furnace_NG']['Economics']['OPEX_variable']
    tec_cost = capex + opex_fix + opex_var
    cost_should = tec_cost + import_cost
    cost_error = abs(cost_should - cost_res) / cost_res
    assert cost_error <= 0.001

def test_addtechnology():
    """
    electricity demand @ node 2
    battery at node 2
    first, WT at node 1, later PV at node 2

    second solve should be cheaper
    """
    data = dm.load_data_handle(r'./test/test_data/addtechnology.p')
    data.technology_data['test_node1']['WT_OS_6000']['TechnologyPerf']['curtailment'] = 0
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()

    obj1 = energyhub.model.objective()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    sizeWT1 = energyhub.model.node_blocks['test_node1'].tech_blocks_active['WT_OS_6000'].var_size.value
    sizeBattery1 = energyhub.model.node_blocks['test_node2'].tech_blocks_active['battery'].var_size.value
    assert 0 <= sizeWT1
    assert 0 <= sizeBattery1
    should = energyhub.model.node_blocks['test_node1'].tech_blocks_active['WT_OS_6000'].var_size.value * 6
    res = energyhub.model.network_block['electricitySimple'].arc_block['test_node1', 'test_node2'].var_size.value
    assert abs(should - res) / res <= 0.01

    energyhub.add_technology_to_node('test_node2', ['PV'])
    energyhub.construct_balances()
    energyhub.solve_model()

    obj2 = energyhub.model.objective()
    sizeWT2 = energyhub.model.node_blocks['test_node1'].tech_blocks_active['WT_OS_6000'].var_size.value
    sizeBattery2 = energyhub.model.node_blocks['test_node2'].tech_blocks_active['battery'].var_size.value
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert sizeWT2 <= sizeWT1
    assert (obj2 - obj1) / obj1 <= 0.8

