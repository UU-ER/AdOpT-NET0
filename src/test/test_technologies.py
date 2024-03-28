import pytest
from pyomo.environ import *
import pandas as pd

from src.data_management.utilities import *
from src.energyhub import EnergyHub
from src.model_configuration import ModelConfiguration


def test_technology_RES_PV():
    """
    Run a model with one node.
    PV @ node 1
    electricity demand @ node 1
    import of electricity at high price
    Size of PV should be around max electricity demand (i.e. 10)
    """
    data = load_object(r'./src/test/test_data/technology_type1_PV.p')
    configuration = ModelConfiguration()
    data.technology_data['test_node1']['Photovoltaic'].performance_data['curtailment'] = 0
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert 10 <= energyhub.model.node_blocks['test_node1'].tech_blocks_active['Photovoltaic'].var_size.value
    assert 15 >= energyhub.model.node_blocks['test_node1'].tech_blocks_active['Photovoltaic'].var_size.value

    for t in energyhub.model.set_t_full:
        energyhub.model.node_blocks['test_node1'].para_import_price[t, 'electricity'] = 0
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert 0 == energyhub.model.node_blocks['test_node1'].tech_blocks_active['Photovoltaic'].var_size.value
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
    data = load_object(r'./src/test/test_data/technology_type1_WT.p')
    configuration = ModelConfiguration()
    data.technology_data['test_node1']['TestWindTurbine_Onshore_1500'].performance_data['curtailment'] = 0
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert 6 == energyhub.model.node_blocks['test_node1'].tech_blocks_active['TestWindTurbine_Onshore_1500'].var_size.value

    # Import at zero price
    for t in energyhub.model.set_t_full:
        energyhub.model.node_blocks['test_node1'].para_import_price[t, 'electricity'] = 0
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert 0 == energyhub.model.node_blocks['test_node1'].tech_blocks_active['TestWindTurbine_Onshore_1500'].var_size.value
    assert 0 == energyhub.model.objective()

    # Curtailment
    data.technology_data['test_node1']['TestWindTurbine_Onshore_1500'].performance_data['curtailment'] = 2
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert 6 <= energyhub.model.node_blocks['test_node1'].tech_blocks_active['TestWindTurbine_Onshore_1500'].var_size.value

def test_technology_CONV1():
    """
    heat demand @ node 1
    Technology type 1, gas,H2 -> heat, electricity
    """
    # performance through origin
    data = load_object(r'./src/test/test_data/technology_CONV1_1.p')
    cost_correction = data.topology.fraction_of_year_modelled
    configuration = ModelConfiguration()
    tecname = 'testCONV1_1'
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    allowed_fitting_error = 0.1

    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 6)
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
    assert abs(tec_size * 10 * cost_correction- objective_value) / objective_value <= allowed_fitting_error
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
    data = load_object(r'./src/test/test_data/technology_CONV1_2.p')
    cost_correction = data.topology.fraction_of_year_modelled
    configuration = ModelConfiguration()
    tecname = 'testCONV1_2'
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 6)
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

    assert abs(tec_size * 10 * cost_correction - objective_value) / objective_value <= allowed_fitting_error
    assert abs((hydrogen_in_1 - tec_size) / tec_size) <= allowed_fitting_error
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
    data = load_object(r'./src/test/test_data/technology_CONV1_3.p')
    cost_correction = data.topology.fraction_of_year_modelled
    tecname = 'testCONV1_3'
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 6)
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

    assert abs(tec_size * 10 * cost_correction - objective_value) / objective_value <= allowed_fitting_error
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
    data = load_object(r'./src/test/test_data/technology_CONV1_2.p')
    data.node_data['test_node1'].data['demand']['heat'][1] = 0.001
    data.node_data['test_node1'].data['export_limit']['electricity'][1] = 0
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    energyhub.model.pprint()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'

    # Min partload
    data = load_object(r'./src/test/test_data/technology_CONV1_3.p')
    data.node_data['test_node1'].data['demand']['heat'][1] = 0.001
    data.node_data['test_node1'].data['export_limit']['electricity'][1] = 0
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    energyhub.model.pprint()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'

def test_technology_CONV2():
    # performance through origin
    allowed_fitting_error = 0.05
    data = load_object(r'./src/test/test_data/technology_CONV2_1.p')
    cost_correction = data.topology.fraction_of_year_modelled

    configuration = ModelConfiguration()
    tecname = 'testCONV2_1'
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 6)
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
    assert abs(tec_size * 10 * cost_correction - objective_value) / objective_value <= allowed_fitting_error
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
    data = load_object(r'./src/test/test_data/technology_CONV2_2.p')
    cost_correction = data.topology.fraction_of_year_modelled

    tecname = 'testCONV2_2'
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 6)
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
    assert abs(tec_size * 10 * cost_correction - objective_value) / objective_value <= allowed_fitting_error
    assert 0.75 == heat_out_1
    assert 0.5 == heat_out_2
    assert 0 == gas_in_1
    assert 0 == gas_in_2
    assert abs((hydrogen_in_1 - tec_size) / tec_size) <= allowed_fitting_error
    assert hydrogen_in_2 <= hydrogen_in_1
    assert abs((heat_out_1 - 0.05) / 0.75 - hydrogen_in_1) / hydrogen_in_1 <= allowed_fitting_error
    assert abs((heat_out_2 - 0.05) / 0.75 - hydrogen_in_2) / hydrogen_in_2 <= allowed_fitting_error

    # piecewise
    data = load_object(r'./src/test/test_data/technology_CONV2_3.p')
    cost_correction = data.topology.fraction_of_year_modelled

    tecname = 'testCONV2_3'
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 6)
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
    assert abs(tec_size * 10 * cost_correction - objective_value) / objective_value <= allowed_fitting_error
    assert 0.75 == heat_out_1
    assert 0.5 == heat_out_2
    assert 0 == gas_in_1
    assert 0 == gas_in_2
    assert hydrogen_in_1 == tec_size
    assert hydrogen_in_2 <= hydrogen_in_1
    assert abs((heat_out_1 - 0.25) / 0.5 - hydrogen_in_1) / hydrogen_in_1 <= allowed_fitting_error
    assert abs(heat_out_2 - hydrogen_in_2) <= allowed_fitting_error

    # Min partload
    data = load_object(r'./src/test/test_data/technology_CONV2_2.p')
    data.node_data['test_node1'].data['demand']['heat'][1] = 0.001
    data.node_data['test_node1'].data['export_limit']['electricity'][1] = 0
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'

    # Min partload
    data = load_object(r'./src/test/test_data/technology_CONV2_3.p')
    data.node_data['test_node1'].data['demand']['heat'][1] = 0.001
    data.node_data['test_node1'].data['export_limit']['electricity'][1] = 0
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'


def test_technology_CONV3():
    """
    heat demand @ node 1
    Technology type 3, gas,H2 -> heat, electricity
    """
    # Piecewise definition
    data = load_object(r'./src/test/test_data/technology_CONV3_3.p')
    cost_correction = data.topology.fraction_of_year_modelled

    configuration = ModelConfiguration()
    allowed_fitting_error = 0.25
    tecname = 'testCONV3_3'
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 6)
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
    assert abs(10 * cost_correction - objective_value) / 10 <= allowed_fitting_error
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
    data = load_object(r'./src/test/test_data/technology_CONV3_1.p')
    cost_correction = data.topology.fraction_of_year_modelled

    tecname = 'testCONV3_1'
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 6)
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
    assert abs(10 * cost_correction - objective_value) / 10 <= allowed_fitting_error
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
    data = load_object(r'./src/test/test_data/technology_CONV3_2.p')
    cost_correction = data.topology.fraction_of_year_modelled

    tecname = 'testCONV3_2'
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 6)
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
    assert abs(10 * cost_correction - objective_value) / 10 <= allowed_fitting_error
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
    data = load_object(r'./src/test/test_data/technology_CONV3_2.p')

    data.node_data['test_node1'].data['demand']['heat'][1] = 0.001
    data.node_data['test_node1'].data['export_limit']['electricity'][1] = 0
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'

    # Piecewise
    data = load_object(r'./src/test/test_data/technology_CONV3_3.p')
    data.node_data['test_node1'].data['demand']['heat'][1] = 0.001
    data.node_data['test_node1'].data['export_limit']['electricity'][1] = 0
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'


def test_technology_CONV4():
    data = load_object(r'./src/test/test_data/technology_CONV4_1.p')
    configuration = ModelConfiguration()
    tecname = 'testCONV4_1'
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.quick_solve()

    assert energyhub.solution.solver.termination_condition == 'optimal'
    objective_value = round(energyhub.model.objective(), 6)
    heat_out_1 = round(
        energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'heat'].value, 3)
    el_out_1 = round(
        energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[1, 'electricity'].value,
        3)
    heat_out_2 = round(
        energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'heat'].value, 3)
    el_out_2 = round(
        energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_output[2, 'electricity'].value,
        3)
    assert 0.75 == heat_out_1
    assert 1.5 == el_out_1
    assert 0.5 == heat_out_2
    assert 1 == el_out_2

    data = load_object(r'./src/test/test_data/technology_CONV4_2.p')
    configuration = ModelConfiguration()
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.quick_solve()

    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'


def test_technology_STOR():
    data = load_object(r'./src/test/test_data/technologySTOR.p')
    configuration = ModelConfiguration()
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.quick_solve()

    assert energyhub.solution.solver.termination_condition == 'optimal'
    el_out_1 = round(
        energyhub.model.node_blocks['test_node1'].tech_blocks_active['testSTOR'].var_output[1, 'electricity'].value,
        3)
    el_out_2 = round(
        energyhub.model.node_blocks['test_node1'].tech_blocks_active['testSTOR'].var_output[2, 'electricity'].value,
        3)
    assert 0 == el_out_1
    assert 0.1 == el_out_2

def test_dac():
    # data.save(data_save_path)
    data = load_object(r'./src/test/test_data/dac.p')

    configuration = ModelConfiguration()
    configuration.optimization.typicaldays.N = 0
    # # Read data
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.quick_solve()
    cost1 = energyhub.model.var_total_cost.value

    configuration = ModelConfiguration()
    configuration.optimization.typicaldays.N = 4
    # # Read data
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.quick_solve()
    cost2 = energyhub.model.var_total_cost.value

    assert abs(cost1 - cost2) / cost1 <= 0.1



def test_existing_technologies():
    def run_EnergyHub(data, configuration):
        energyhub = EnergyHub(data, configuration)
        energyhub.configuration.reporting.save_path = './src/test/results'
        energyhub.configuration.reporting.save_summary_path = './src/test/results'
        energyhub.construct_model()
        energyhub.construct_balances()
        energyhub.solve()
        assert energyhub.solution.solver.termination_condition == 'optimal'
        cost = energyhub.model.var_total_cost.value
        return cost

    configuration = ModelConfiguration()
    data = load_object(r'./src/test/test_data/existing_tecs1.p')
    cost1 = run_EnergyHub(data, configuration)
    data = load_object(r'./src/test/test_data/existing_tecs2.p')
    cost2 = run_EnergyHub(data, configuration)
    data = load_object(r'./src/test/test_data/existing_tecs3.p')
    cost3 = run_EnergyHub(data, configuration)
    assert cost3<cost2*1.02
    assert cost2<cost1*1.02


def test_technology_OpenHydro():
    # electricity from open hydro only
    data = load_object(r'./src/test/test_data/technologyOpenHydro.p')
    configuration = ModelConfiguration()
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.quick_solve()

    assert energyhub.solution.solver.termination_condition == 'optimal'
    el_out_1 = round(
        energyhub.model.node_blocks['test_node1'].tech_blocks_active['TestPumpedHydro_Open'].var_output[
            1, 'electricity'].value,
        3)
    el_out_2 = round(
        energyhub.model.node_blocks['test_node1'].tech_blocks_active['TestPumpedHydro_Open'].var_output[
            2, 'electricity'].value,
        3)
    size_WT = energyhub.model.node_blocks['test_node1'].tech_blocks_active[
        'TestWindTurbine_Onshore_1500'].var_size.value
    assert 1 == el_out_1
    assert 1 == el_out_2
    assert 0 == size_WT

    # electricity WT, stored in open hydro
    data.node_data['test_node1'].data['climate_data']['TestPumpedHydro_Open_inflow'][0] = 0
    data.node_data['test_node1'].data['climate_data']['TestPumpedHydro_Open_inflow'][1] = 0
    data.read_technology_data(load_path = './src/test/TestTecs')
    configuration = ModelConfiguration()
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.quick_solve()

    assert energyhub.solution.solver.termination_condition == 'optimal'
    el_out_1 = round(
        energyhub.model.node_blocks['test_node1'].tech_blocks_active['TestPumpedHydro_Open'].var_output[
            1, 'electricity'].value,
        3)
    el_out_2 = round(
        energyhub.model.node_blocks['test_node1'].tech_blocks_active['TestPumpedHydro_Open'].var_output[
            2, 'electricity'].value,
        3)
    size_WT = energyhub.model.node_blocks['test_node1'].tech_blocks_active[
        'TestWindTurbine_Onshore_1500'].var_size.value
    assert 0 == el_out_1
    assert 1 == el_out_2
    assert 0 < size_WT

    # no pumping allowed, infeasible
    data.technology_data['test_node1']['TestPumpedHydro_Open'].performance_data['can_pump'] = 0

    configuration = ModelConfiguration()
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.quick_solve()

    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'

    # Maximum discharge too small
    data = load_object(r'./src/test/test_data/technologyOpenHydro_max_discharge.p')
    configuration = ModelConfiguration()
    energyhub = EnergyHub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.quick_solve()

    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'


def test_CAPEX_technologies():
    data = load_object(r'./src/test/test_data/technology_CONV1_1.p')
    cost_correction = data.topology.fraction_of_year_modelled
    configuration = ModelConfiguration()


    # collect data
    unitCAPEX = data.technology_data['test_node1']['testCONV1_1'].economics.capex_data['unit_capex']
    fixCAPEX = data.technology_data['test_node1']['testCONV1_1'].economics.capex_data['fix_capex']
    bp_x = data.technology_data['test_node1']['testCONV1_1'].economics.capex_data['piecewise_capex']['bp_x']
    bp_y = data.technology_data['test_node1']['testCONV1_1'].economics.capex_data['piecewise_capex']['bp_y']

    def CAPEX_piecewise_function(x):
        if x <= bp_x[0]:
            return bp_y[0]
        elif x >= bp_x[-1]:
            return bp_y[-1]
        else:
            for i in range(len(bp_x) - 1):
                if bp_x[i] <= x < bp_x[i + 1]:
                    return bp_y[i] + ((x - bp_x[i]) / (bp_x[i + 1] - bp_x[i])) * (bp_y[i + 1] - bp_y[i])

    CAPEX_models = [1, 2, 3]
    for i in CAPEX_models:
        data.technology_data['test_node1']['testCONV1_1'].economics.capex_model = i

        # Solve model
        energyhub = EnergyHub(data, configuration)
        energyhub.configuration.reporting.save_path = './src/test/results'
        energyhub.configuration.reporting.save_summary_path = './src/test/results'
        energyhub.quick_solve()

        # test if optimal
        assert energyhub.solution.solver.termination_condition == 'optimal'

        size = energyhub.model.node_blocks['test_node1'].tech_blocks_active['testCONV1_1'].var_size.value
        # check if capex is correct
        if i == 1:
            should = (size * unitCAPEX) * cost_correction
            res = energyhub.model.node_blocks['test_node1'].tech_blocks_active['testCONV1_1'].var_capex.value
            assert abs(should - res) / res <= 0.001
        if i == 2:
            should = CAPEX_piecewise_function(size) * cost_correction
            res = energyhub.model.node_blocks['test_node1'].tech_blocks_active['testCONV1_1'].var_capex.value
            assert abs(should - res) / res <= 0.001
        if i == 3:
            should = (size * unitCAPEX + fixCAPEX) * cost_correction
            res = energyhub.model.node_blocks['test_node1'].tech_blocks_active['testCONV1_1'].var_capex.value
            assert abs(should - res) / res <= 0.001


def test_fast_dynamics():
    """
    Test SU/SD load, maximum number of startups, ramping rate and standby power
    heat demand @ node 1
    Performance type 2 and 3, gas,H2 -> heat, electricity
    """
    # turn dynamics on
    configuration = ModelConfiguration()
    configuration.performance.dynamics = 1
    configuration.reporting.save_path = './src/test/results'
    configuration.reporting.save_summary_path = './src/test/results'

    perf_function_type = [2, 3]
    CONV_Type = [1, 2, 3]
    for j in CONV_Type:
        for i in perf_function_type:
            data_load_path = r'./src/test/test_data/technology_dynamics_CONV' + str(j) + '_' + str(i) + '.p'
            data = load_object(data_load_path)
            tecname = 'testCONV' + str(j) + '_' + str(i)

            if j != 3:
                # Test technology dynamic parameters: standby power and max startups
                data.technology_data['test_node1'][tecname].performance_data['min_part_load'] = 0.3
                data.technology_data['test_node1'][tecname].performance_data['standby_power'] = 0.1
                data.technology_data['test_node1'][tecname].performance_data['max_startups'] = 1

                # Solve model
                energyhub1 = EnergyHub(data, configuration)
                energyhub1.model_information.testing = 1
                energyhub1.quick_solve()

                assert energyhub1.solution.solver.termination_condition == 'optimal'
                tec_size = round(energyhub1.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value,
                                 3)
                gas_in_6 = round(
                    energyhub1.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[6, 'gas'].value, 3)
                gas_in_7 = round(
                    energyhub1.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[7, 'gas'].value, 3)
                SU_number = sum(
                    energyhub1.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_y[i].value for i in
                    range(1, len(energyhub1.data.topology.timesteps) + 1))

                assert gas_in_6 <= 0.1 * tec_size
                assert gas_in_7 <= 0.1 * tec_size
                assert SU_number <= 1

                # Test technology dynamic parameters: ramping rate
                RR = max(data.node_data['test_node1'].data['demand']['heat']) / 2
                data.technology_data['test_node1'][tecname].performance_data['ramping_rate'] = RR

                # Solve model
                energyhub2 = EnergyHub(data, configuration)
                energyhub2.model_information.testing = 1
                energyhub2.quick_solve()

                assert energyhub2.solution.solver.termination_condition == 'optimal'

                gas_in_1 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, 'gas'].value, 3)
                hydrogen_in_1 = round(energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[
                                          1, 'hydrogen'].value, 3)
                gas_in_2 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, 'gas'].value, 3)
                hydrogen_in_2 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[
                        2, 'hydrogen'].value,
                    3)
                assert round(abs((gas_in_1 + hydrogen_in_1) - (gas_in_2 + hydrogen_in_2)), 3) <= RR

                gas_in_5 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[5, 'gas'].value, 3)
                hydrogen_in_5 = round(energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[
                                          5, 'hydrogen'].value, 3)
                gas_in_6 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[6, 'gas'].value, 3)
                hydrogen_in_6 = round(energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[
                                          6, 'hydrogen'].value, 3)
                assert round(abs((gas_in_5 + hydrogen_in_5) - (gas_in_6 + hydrogen_in_6)), 3) <= RR

                gas_in_7 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[7, 'gas'].value, 3)
                hydrogen_in_7 = round(energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[
                                          7, 'hydrogen'].value, 3)
                gas_in_8 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[8, 'gas'].value, 3)
                hydrogen_in_8 = round(energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[
                                          8, 'hydrogen'].value, 3)
                assert round(abs((gas_in_7 + hydrogen_in_7) - (gas_in_8 + hydrogen_in_8)), 3) <= RR

                # Test technology dynamic parameters: SU_load and SD_load
                data.technology_data['test_node1'][tecname].performance_data['ramping_rate'] = -1
                data.technology_data['test_node1'][tecname].performance_data['SU_load'] = 0.6
                data.technology_data['test_node1'][tecname].performance_data['SD_load'] = 0.8

                # Solve model
                energyhub3 = EnergyHub(data, configuration)
                energyhub3.model_information.testing = 1
                energyhub3.quick_solve()

                assert energyhub3.solution.solver.termination_condition == 'optimal'
                tec_size = round(energyhub3.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value,
                                 3)

                gas_in_7 = round(
                    energyhub3.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[7, 'gas'].value, 3)
                hydrogen_in_7 = round(energyhub3.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[
                                          7, 'hydrogen'].value, 3)
                print(tecname)
                assert gas_in_7 + hydrogen_in_7 <= 0.6 * tec_size

                gas_in_5 = round(
                    energyhub3.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[5, 'gas'].value, 3)
                hydrogen_in_5 = round(energyhub3.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[
                                          5, 'hydrogen'].value, 3)
                assert gas_in_5 + hydrogen_in_5 <= 0.8 * tec_size

            else:
                main_car = data.technology_data['test_node1'][tecname].performance_data['main_input_carrier']

                # Test technology dynamic parameters: standby power and max startups
                data.technology_data['test_node1'][tecname].performance_data['min_part_load'] = 0.3
                data.technology_data['test_node1'][tecname].performance_data['standby_power'] = 0.1
                data.technology_data['test_node1'][tecname].performance_data['max_startups'] = 1

                # Solve model
                energyhub1 = EnergyHub(data, configuration)
                energyhub1.model_information.testing = 1
                energyhub1.quick_solve()

                assert energyhub1.solution.solver.termination_condition == 'optimal'
                tec_size = round(energyhub1.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value,
                                 3)
                gas_in_6 = round(
                    energyhub1.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[6, 'gas'].value,
                    3)
                gas_in_7 = round(
                    energyhub1.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[7, 'gas'].value,
                    3)
                SU_number = sum(
                    energyhub1.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_y[i].value for i in
                    range(1, len(energyhub1.data.topology.timesteps) + 1))

                assert gas_in_6 <= 0.1 * tec_size
                assert gas_in_7 <= 0.1 * tec_size
                assert SU_number <= 1

                # Test technology dynamic parameters: ramping rate
                RR = max(data.node_data['test_node1'].data['demand']['heat']) * 0.75
                data.technology_data['test_node1'][tecname].performance_data['ramping_rate'] = RR

                # Solve model
                energyhub2 = EnergyHub(data, configuration)
                energyhub2.model_information.testing = 1
                energyhub2.quick_solve()

                assert energyhub2.solution.solver.termination_condition == 'optimal'

                main_in_1 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[1, main_car].value,
                    3)
                main_in_2 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[2, main_car].value,
                    3)

                assert abs(main_in_1 - main_in_2) <= RR

                main_in_5 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[5, main_car].value,
                    3)
                main_in_6 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[6, main_car].value,
                    3)

                assert abs(main_in_5 - main_in_6) <= RR

                main_in_7 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[7, main_car].value,
                    3)
                main_in_8 = round(
                    energyhub2.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[8, main_car].value,
                    3)

                assert abs(main_in_7 - main_in_8) <= RR

                # Test technology dynamic parameters: SU_load and SD_load
                data.technology_data['test_node1'][tecname].performance_data['ramping_rate'] = -1
                data.technology_data['test_node1'][tecname].performance_data['SU_load'] = 0.6
                data.technology_data['test_node1'][tecname].performance_data['SD_load'] = 0.8

                # Solve model
                energyhub3 = EnergyHub(data, configuration)
                energyhub3.model_information.testing = 1
                energyhub3.quick_solve()

                assert energyhub3.solution.solver.termination_condition == 'optimal'
                tec_size = round(energyhub3.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value,
                                 3)

                main_in_7 = round(
                    energyhub3.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[7, main_car].value,
                    3)

                assert main_in_7 <= 0.6 * tec_size

                main_in_5 = round(
                    energyhub3.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[5, main_car].value,
                    3)
                assert main_in_5 <= 0.8 * tec_size


def test_slow_dynamics():
    """
    Test SU/SD time for slow dynamics (perf type 4)
    heat demand @ node 1
    For conv1 conv2 and conv3, gas,H2 -> heat, electricity
    """
    # turn dynamics on
    configuration = ModelConfiguration()
    configuration.performance.dynamics = 1
    allowed_error = 0.01

    CONV_Type = [1, 2, 3]
    for j in CONV_Type:

        data_load_path = r'./src/test/test_data/technology_dynamics_CONV' + str(j) + '_' + str(4) + '.p'
        data = load_object(data_load_path)
        tecname = 'testCONV' + str(j) + '_' + str(4)

        #change SU time and SD time
        SU_time = 2
        SD_time = 1
        min_part_load = 0.5
        data.technology_data['test_node1'][tecname].performance_data['min_part_load'] = min_part_load
        data.technology_data['test_node1'][tecname].performance_data['SU_time'] = SU_time
        data.technology_data['test_node1'][tecname].performance_data['SD_time'] = SD_time

        # Calculate SU and SD trajectories
        SU_trajectory = []
        for i in range(1, SU_time + 1):
            SU_trajectory.append((min_part_load / (SU_time + 1)) * i)

        SD_trajectory = []
        for i in range(1, SD_time + 1):
            SD_trajectory.append((min_part_load / (SD_time + 1)) * i)
        SD_trajectory = sorted(SD_trajectory, reverse=True)


        if j != 3:

            # Solve model
            energyhub = EnergyHub(data, configuration)
            energyhub.configuration.reporting.save_path = './src/test/results'
            energyhub.configuration.reporting.save_summary_path = './src/test/results'
            energyhub.quick_solve()

            # collect results
            tec_size = round(energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_size.value,
                             3)

            gas_in_3 = round(
                energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[3, 'gas'].value, 3)
            gas_in_5 = round(
                energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[5, 'gas'].value, 3)
            gas_in_6 = round(
                energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[6, 'gas'].value, 3)
            hydrogen_in_3 = round(
                energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[
                    3, 'hydrogen'].value, 3)
            hydrogen_in_5 = round(
                energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[
                    5, 'hydrogen'].value, 3)
            hydrogen_in_6 = round(
                energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[
                    6, 'hydrogen'].value, 3)

            assert (gas_in_3 + hydrogen_in_3) == round(SD_trajectory[0] * tec_size, 3)
            assert (gas_in_5 + hydrogen_in_5) == round(SU_trajectory[0] * tec_size, 3)
            assert (gas_in_6 + hydrogen_in_6) == round(SU_trajectory[1] * tec_size, 3)

        else:
            main_car = data.technology_data['test_node1'][tecname].performance_data['main_input_carrier']


            # Solve model
            energyhub = EnergyHub(data, configuration)
            energyhub.configuration.reporting.save_path = './src/test/results'
            energyhub.configuration.reporting.save_summary_path = './src/test/results'
            energyhub.quick_solve()

            main_in_3 = round(
                energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[3, main_car].value, 3)
            main_in_5 = round(
                energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[5, main_car].value, 3)
            main_in_6 = round(
                energyhub.model.node_blocks['test_node1'].tech_blocks_active[tecname].var_input[6, main_car].value, 3)

            assert main_in_3 == round(SD_trajectory[0] * tec_size)
            assert main_in_5 == round(SU_trajectory[0] * tec_size)
            assert main_in_6 == round(SU_trajectory[1] * tec_size)

def test_technology_CARBONCAPTURE():
    data = load_object(r'./src/test/test_data/technologyCARBONCAPTURE.p')
    configuration = ModelConfiguration()
    sizes = ['small', 'medium', 'large']
    el = {}
    heat = {}

    energyhub = EnergyHub(data, configuration)
    energyhub.model_information.testing = 1
    energyhub.quick_solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'


    for s in sizes:
        el[s] = energyhub.model.node_blocks['test_node1'].tech_blocks_active['testCarbonCapture_MEA_' + s].var_input[1, 'electricity'].value
        heat[s] = energyhub.model.node_blocks['test_node1'].tech_blocks_active['testCarbonCapture_MEA_' + s].var_input[1, 'heat'].value
        CO2_fluegas_in = energyhub.model.node_blocks['test_node1'].tech_blocks_active['testCarbonCapture_MEA_' + s].var_input[1, 'CO2fluegas'].value

        input_ratio_el = data.technology_data['test_node1']['testCarbonCapture_MEA_' + s].input_ratios['electricity']
        input_ratio_heat = data.technology_data['test_node1']['testCarbonCapture_MEA_' + s].input_ratios['heat']

        assert input_ratio_el * CO2_fluegas_in == el[s]
        assert input_ratio_heat * CO2_fluegas_in == heat[s]
