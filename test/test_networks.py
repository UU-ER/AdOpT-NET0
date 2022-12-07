import pytest
import src.data_management as dm
from src.energyhub import energyhub as ehub
from pyomo.environ import units as u
from pyomo.environ import *
import pandas as pd



def test_networks():
    """
    Run a model with one node.
    PV @ node 1
    electricity demand @ node 1
    import of electricity at high price
    Size of PV should be around max electricity demand (i.e. 10)
    """
    data = dm.load_data_handle(r'./test/test_data/technology_type1_PV.p')
    energyhub = ehub(data)
    energyhub.construct_model()
    solver = SolverFactory('gurobi')
    solution = solver.solve(energyhub.model, tee=True)
    assert solution.solver.termination_condition == 'optimal'
    assert 10 <= energyhub.model.node_blocks['test_node1'].tech_blocks['PV'].var_size.value
    assert 15 >= energyhub.model.node_blocks['test_node1'].tech_blocks['PV'].var_size.value

    for t in energyhub.model.set_t:
        energyhub.model.node_blocks['test_node1'].para_import_price[t, 'electricity'] = 0
    solver = SolverFactory('gurobi')
    solution = solver.solve(energyhub.model, tee=True)
    assert solution.solver.termination_condition == 'optimal'
    assert 0 == energyhub.model.node_blocks['test_node1'].tech_blocks['PV'].var_size.value
    assert 0 == energyhub.model.objective()
