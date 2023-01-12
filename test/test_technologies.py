import pytest
import src.data_management as dm
from src.energyhub import EnergyHub as ehub
from pyomo.environ import units as u
from pyomo.environ import *
import pandas as pd



def test_technology_RES_PV():
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
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert 10 <= energyhub.model.node_blocks['test_node1'].tech_blocks_active['PV'].var_size.value
    assert 15 >= energyhub.model.node_blocks['test_node1'].tech_blocks_active['PV'].var_size.value

    for t in energyhub.model.set_t:
        energyhub.model.node_blocks['test_node1'].para_import_price[t, 'electricity'] = 0
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert 0 == energyhub.model.node_blocks['test_node1'].tech_blocks_active['PV'].var_size.value
    assert 0 == energyhub.model.objective()


def test_technology_RES_WT():
    """
    Run a model with one node.
    WT @ node 1
    electricity demand @ node 1
    import of electricity at high price
    Size of WT should be around max electricity demand (i.e. 10), with 1.5MW rated power, this is 6
    """
    # No curtailment
    data = dm.load_data_handle(r'./test/test_data/technology_type1_WT.p')
    data.technology_data['test_node1']['WT_1500']['TechnologyPerf']['curtailment'] = 0
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert 6 == energyhub.model.node_blocks['test_node1'].tech_blocks_active['WT_1500'].var_size.value

    # Import at zero price
    for t in energyhub.model.set_t:
        energyhub.model.node_blocks['test_node1'].para_import_price[t, 'electricity'] = 0
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert 0 == energyhub.model.node_blocks['test_node1'].tech_blocks_active['WT_1500'].var_size.value
    assert 0 == energyhub.model.objective()

    # Curtailment
    data.technology_data['test_node1']['WT_1500']['TechnologyPerf']['curtailment'] = 2
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert 6 <= energyhub.model.node_blocks['test_node1'].tech_blocks_active['WT_1500'].var_size.value

def test_technology_CONV1():
    """
    heat demand @ node 1
    Technology type 1, gas,H2 -> heat, electricity
    """
    # performance through origin
    data = dm.load_data_handle(r'./test/test_data/technology_CONV1_1.p')
    tecname = 'testCONV1_1'
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    allowed_fitting_error = 0.1

    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 3)
    tec_size = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value, 3)
    gas_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'gas'].value, 3)
    hydrogen_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'hydrogen'].value,
                          3)
    gas_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'gas'].value, 3)
    hydrogen_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'hydrogen'].value,
                          3)
    heat_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'heat'].value, 3)
    el_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'electricity'].value,
                     3)
    heat_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'heat'].value, 3)
    el_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'electricity'].value,
                     3)
    assert abs(tec_size * 10 - objective_value) / objective_value <= allowed_fitting_error
    assert hydrogen_in_1 == tec_size
    assert 0 == gas_in_1
    assert 0 == gas_in_2
    assert abs(0.75 / 0.8145 - hydrogen_in_1) / 0.75 / 0.8145 <= allowed_fitting_error
    assert abs(heat_out_2 / 0.8145 - hydrogen_in_2) / heat_out_2 / 0.8145 <= allowed_fitting_error
    assert 0.75 == heat_out_1
    assert 0 == el_out_1
    assert 0.5 == heat_out_2
    assert 0 == el_out_2

    # performance not through origin
    allowed_fitting_error = 0.1
    data = dm.load_data_handle(r'./test/test_data/technology_CONV1_2.p')
    tecname = 'testCONV1_2'
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 3)
    tec_size = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value, 3)
    gas_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'gas'].value, 3)
    hydrogen_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'hydrogen'].value,
                          3)
    gas_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'gas'].value, 3)
    hydrogen_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'hydrogen'].value,
                          3)
    heat_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'heat'].value, 3)
    el_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'electricity'].value,
                     3)
    heat_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'heat'].value, 3)
    el_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'electricity'].value,
                     3)

    assert abs(tec_size * 10 - objective_value) / objective_value <= allowed_fitting_error
    assert hydrogen_in_1 == tec_size
    assert 0 == gas_in_1
    assert 0 == gas_in_2
    assert abs((heat_out_1 - 0.05) / 0.75 - hydrogen_in_1) / hydrogen_in_1 <= allowed_fitting_error
    assert abs((heat_out_2 - 0.05) / 0.75 - hydrogen_in_2) / hydrogen_in_2 <= allowed_fitting_error
    assert 0.75 == heat_out_1
    assert 0 == el_out_1
    assert 0.5 == heat_out_2
    assert 0 == el_out_2

    # piecewise
    allowed_fitting_error = 0.1
    data = dm.load_data_handle(r'./test/test_data/technology_CONV1_3.p')
    tecname = 'testCONV1_3'
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 3)
    tec_size = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value, 3)
    gas_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'gas'].value, 3)
    hydrogen_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'hydrogen'].value,
                          3)
    gas_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'gas'].value, 3)
    hydrogen_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'hydrogen'].value,
                          3)
    heat_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'heat'].value, 3)
    el_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'electricity'].value,
                     3)
    heat_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'heat'].value, 3)
    el_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'electricity'].value,
                     3)

    assert abs(tec_size * 10 - objective_value) / objective_value <= allowed_fitting_error
    assert hydrogen_in_1 == tec_size
    assert 0 == gas_in_1
    assert 0 == gas_in_2
    assert abs((heat_out_1 - 0.25) / 0.5 - hydrogen_in_1) / hydrogen_in_1 <= allowed_fitting_error
    assert abs(heat_out_2 - hydrogen_in_2) == 0
    assert 0.75 == heat_out_1
    assert 0 == el_out_1
    assert 0.5 == heat_out_2
    assert 0 == el_out_2

    # Min partload
    data = dm.load_data_handle(r'./test/test_data/technology_CONV1_2.p')
    data.node_data['test_node1']['demand']['heat'][1] = 0.001
    data.node_data['test_node1']['export_limit']['electricity'][1] = 0
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    energyhub.model.pprint()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'

    # Min partload
    data = dm.load_data_handle(r'./test/test_data/technology_CONV1_3.p')
    data.node_data['test_node1']['demand']['heat'][1] = 0.001
    data.node_data['test_node1']['export_limit']['electricity'][1] = 0
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    energyhub.model.pprint()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'

def test_technology_CONV2():
    # performance through origin
    allowed_fitting_error = 0.05
    data = dm.load_data_handle(r'./test/test_data/technology_CONV2_1.p')
    tecname = 'testCONV2_1'
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 3)
    tec_size = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value, 3)
    gas_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'gas'].value, 3)
    hydrogen_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'hydrogen'].value,
                          3)
    gas_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'gas'].value, 3)
    hydrogen_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'hydrogen'].value,
                          3)
    heat_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'heat'].value, 3)
    el_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'electricity'].value,
                     3)
    heat_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'heat'].value, 3)
    el_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'electricity'].value,
                     3)
    assert abs(tec_size * 10 - objective_value) / objective_value <= allowed_fitting_error
    assert 0.75 == heat_out_1
    assert 0.5 == heat_out_2
    assert hydrogen_in_1 == tec_size
    assert abs(0.75 / 0.8145 - hydrogen_in_1) / 0.75 / 0.8145 <= allowed_fitting_error
    assert abs(heat_out_2 / 0.8145 - hydrogen_in_2) / heat_out_2 / 0.8145 <= allowed_fitting_error
    assert 0 == gas_in_1
    assert 0 == gas_in_2
    assert (heat_out_1 / 2 - el_out_1) / el_out_1 <= allowed_fitting_error
    assert (heat_out_2 / 2 - el_out_2) / el_out_2 <= allowed_fitting_error

    # performance not through origin
    allowed_fitting_error = 0.05
    data = dm.load_data_handle(r'./test/test_data/technology_CONV2_2.p')
    tecname = 'testCONV2_2'
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 3)
    tec_size = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value, 3)
    gas_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'gas'].value, 3)
    hydrogen_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'hydrogen'].value,
                          3)
    gas_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'gas'].value, 3)
    hydrogen_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'hydrogen'].value,
                          3)
    heat_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'heat'].value, 3)
    el_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'electricity'].value,
                     3)
    heat_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'heat'].value, 3)
    el_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'electricity'].value,
                     3)
    assert abs(tec_size * 10 - objective_value) / objective_value <= allowed_fitting_error
    assert 0.75 == heat_out_1
    assert 0.5 == heat_out_2
    assert 0 == gas_in_1
    assert 0 == gas_in_2
    assert hydrogen_in_1 == tec_size
    assert hydrogen_in_2 <= hydrogen_in_1
    assert abs((heat_out_1 - 0.05) / 0.75 - hydrogen_in_1) / hydrogen_in_1 <= allowed_fitting_error
    assert abs((heat_out_2 - 0.05) / 0.75 - hydrogen_in_2) / hydrogen_in_2 <= allowed_fitting_error

    # piecewise
    data = dm.load_data_handle(r'./test/test_data/technology_CONV2_3.p')
    tecname = 'testCONV2_3'
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 3)
    tec_size = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value, 3)
    gas_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'gas'].value, 3)
    hydrogen_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'hydrogen'].value,
                          3)
    gas_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'gas'].value, 3)
    hydrogen_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'hydrogen'].value,
                          3)
    heat_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'heat'].value, 3)
    el_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'electricity'].value,
                     3)
    heat_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'heat'].value, 3)
    el_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'electricity'].value,
                     3)
    assert abs(tec_size * 10 - objective_value) / objective_value <= allowed_fitting_error
    assert 0.75 == heat_out_1
    assert 0.5 == heat_out_2
    assert 0 == gas_in_1
    assert 0 == gas_in_2
    assert hydrogen_in_1 == tec_size
    assert hydrogen_in_2 <= hydrogen_in_1
    assert abs((heat_out_1 - 0.25) / 0.5 - hydrogen_in_1) / hydrogen_in_1 <= allowed_fitting_error
    assert abs(heat_out_2 - hydrogen_in_2) <= allowed_fitting_error

    # Min partload
    data = dm.load_data_handle(r'./test/test_data/technology_CONV2_2.p')
    data.node_data['test_node1']['demand']['heat'][1] = 0.001
    data.node_data['test_node1']['export_limit']['electricity'][1] = 0
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'

    # Min partload
    data = dm.load_data_handle(r'./test/test_data/technology_CONV2_3.p')
    data.node_data['test_node1']['demand']['heat'][1] = 0.001
    data.node_data['test_node1']['export_limit']['electricity'][1] = 0
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'


def test_technology_CONV3():
    """
    heat demand @ node 1
    Technology type 3, gas,H2 -> heat, electricity
    """
    # Piecewise definition
    data = dm.load_data_handle(r'./test/test_data/technology_CONV3_3.p')
    tecname = 'testCONV3_3'
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 3)
    tec_size = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value, 3)
    gas_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'gas'].value, 3)
    hydrogen_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'hydrogen'].value,
                          3)
    gas_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'gas'].value, 3)
    hydrogen_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'hydrogen'].value,
                          3)
    heat_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'heat'].value, 3)
    el_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'electricity'].value,
                     3)
    heat_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'heat'].value, 3)
    el_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'electricity'].value,
                     3)
    assert 10 == objective_value
    assert gas_in_1 == tec_size
    assert 1 == gas_in_1
    assert 2 == hydrogen_in_1
    assert 0.5 == gas_in_2
    assert 1 == hydrogen_in_2
    assert 0.75 == heat_out_1
    assert 0.375 == el_out_1
    assert 0.5 == heat_out_2
    assert 0.25 == el_out_2

    # performance through origin
    allowed_fitting_error = 0.25
    data = dm.load_data_handle(r'./test/test_data/technology_CONV3_1.p')
    tecname = 'testCONV3_1'
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 3)
    tec_size = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value, 3)
    gas_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'gas'].value, 3)
    hydrogen_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'hydrogen'].value,
                          3)
    gas_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'gas'].value, 3)
    hydrogen_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'hydrogen'].value,
                          3)
    heat_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'heat'].value, 3)
    el_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'electricity'].value,
                     3)
    heat_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'heat'].value, 3)
    el_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'electricity'].value,
                     3)
    assert abs(10 - objective_value) / 10 <= allowed_fitting_error
    assert gas_in_1 == tec_size
    assert abs(1 - gas_in_1) / 1 <= allowed_fitting_error
    assert abs(2 - hydrogen_in_1) / 2 <= allowed_fitting_error
    assert abs(0.5 - gas_in_2) / 0.5 <= allowed_fitting_error
    assert abs(1 - hydrogen_in_2) / 1 <= allowed_fitting_error
    assert 0.75 == heat_out_1
    assert abs(0.375 - el_out_1) / 0.375 <= allowed_fitting_error
    assert 0.5 == heat_out_2
    assert abs(0.25 - el_out_2) / 0.25 <= allowed_fitting_error

    # performance not through origin
    data = dm.load_data_handle(r'./test/test_data/technology_CONV3_2.p')
    tecname = 'testCONV3_2'
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 3)
    tec_size = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value, 3)
    gas_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'gas'].value, 3)
    hydrogen_in_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'hydrogen'].value,
                          3)
    gas_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'gas'].value, 3)
    hydrogen_in_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'hydrogen'].value,
                          3)
    heat_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'heat'].value, 3)
    el_out_1 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'electricity'].value,
                     3)
    heat_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'heat'].value, 3)
    el_out_2 = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'electricity'].value,
                     3)
    assert abs(10 - objective_value) / 10 <= allowed_fitting_error
    assert abs(gas_in_1 - tec_size) / tec_size <= allowed_fitting_error
    assert abs(gas_in_1 - round(1 - 0.05 / 0.75, 3)) / round(1 - 0.05 / 0.75, 3) <= allowed_fitting_error
    assert abs(gas_in_1 * 2- hydrogen_in_1)/hydrogen_in_1<= allowed_fitting_error
    assert abs((0.5 - 0.05) / 0.75 - gas_in_2)/gas_in_2<= allowed_fitting_error
    assert gas_in_2 * 2 == hydrogen_in_2
    assert 0.75 == heat_out_1
    assert abs(0.375 * gas_in_1 + 0.025 - el_out_1)/el_out_1<= allowed_fitting_error
    assert 0.5 == heat_out_2
    assert abs(0.375 * gas_in_2 + 0.025 - el_out_2)/el_out_2<= allowed_fitting_error

    # Min partload
    data = dm.load_data_handle(r'./test/test_data/technology_CONV3_2.p')
    data.node_data['test_node1']['demand']['heat'][1] = 0.001
    data.node_data['test_node1']['export_limit']['electricity'][1] = 0
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'

    # Min partload
    data = dm.load_data_handle(r'./test/test_data/technology_CONV3_3.p')
    data.node_data['test_node1']['demand']['heat'][1] = 0.001
    data.node_data['test_node1']['export_limit']['electricity'][1] = 0
    energyhub = ehub(data)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'
