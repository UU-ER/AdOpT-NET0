import pytest
import src.data_management as dm
from src.energyhub import EnergyHub as ehub
from pyomo.environ import units as u
from pyomo.environ import *
import pandas as pd
import src.model_construction as mc


def test_initializer():
    data = dm.load_object(r'./test/test_data/data_handle_test.p')
    configuration = mc.ModelConfiguration()
    energyhub = ehub(data, configuration)

def test_add_nodes():
    """
    Add a node with no technology, establishes energybalance
    """
    data = dm.load_object(r'./test/test_data/data_handle_test.p')
    configuration = mc.ModelConfiguration()
    energyhub = ehub(data, configuration)
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
    data = dm.load_object(r'./test/test_data/model1.p')
    configuration = mc.ModelConfiguration()
    energyhub = ehub(data, configuration)
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
    - Emissions larger zero
    """
    data = dm.load_object(r'./test/test_data/model2.p')
    configuration = mc.ModelConfiguration()
    energyhub = ehub(data, configuration)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    m = energyhub.model
    assert energyhub.solution.solver.termination_condition == 'optimal'
    # Size of Furnace
    size_res = m.node_blocks['test_node1'].tech_blocks_active['Furnace_NG'].var_size.value
    size_should = max(data.node_data['test_node1']['demand']['heat']) / \
                  data.technology_data['test_node1']['Furnace_NG'].fitted_performance['heat']['alpha1']
    assert  round(size_res,3) == round(size_should,3)
    # Gas Import in each timestep
    import_res = [value(m.node_blocks['test_node1'].var_import_flow[key, 'gas'].value) for key in m.set_t]
    import_res = pd.Series(import_res)
    import_res = import_res.tolist()
    import_should = data.node_data['test_node1']['demand']['heat'] / data.technology_data['test_node1']['Furnace_NG'].fitted_performance['heat']['alpha1']
    import_should = import_should.tolist()
    assert [round(num,3) for num in import_res] == [round(num,3) for num in import_should]
    # Total cost
    cost_res = m.objective()
    import_price = data.node_data['test_node1']['import_prices']['gas'].tolist()
    import_cost = sum([i1 * i2 for i1, i2 in zip(import_price, import_res)])
    t = data.technology_data['test_node1']['Furnace_NG'].economics.lifetime
    r = data.technology_data['test_node1']['Furnace_NG'].economics.discount_rate
    a = mc.annualize(r,t)
    capex = data.technology_data['test_node1']['Furnace_NG'].economics.capex_data['unit_capex'] * size_res * a
    opex_fix = capex * data.technology_data['test_node1']['Furnace_NG'].economics.opex_fixed
    opex_var = sum(import_res) * data.technology_data['test_node1']['Furnace_NG'].economics.opex_variable
    tec_cost = capex + opex_fix + opex_var
    cost_should = tec_cost + import_cost
    cost_error = abs(cost_should - cost_res) / cost_res
    assert cost_error <= 0.001
    # Emissions
    net_emissions =  energyhub.model.var_emissions_net.value
    emissions_should = sum(import_res) * \
                       data.technology_data['test_node1']['Furnace_NG'].performance_data['emission_factor']
    assert abs(emissions_should - net_emissions) / net_emissions <= 0.01

def test_addtechnology():
    """
    electricity demand @ node 2
    battery at node 2
    first, WT at node 1, later PV at node 2

    second solve should be cheaper
    """
    data = dm.load_object(r'./test/test_data/addtechnology.p')
    configuration = mc.ModelConfiguration()
    data.technology_data['test_node1']['WindTurbine_Offshore_6000'].performance_data['curtailment'] = 0
    energyhub = ehub(data, configuration)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()

    obj1 = energyhub.model.objective()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    sizeWT1 = energyhub.model.node_blocks['test_node1'].tech_blocks_active['WindTurbine_Offshore_6000'].var_size.value
    sizeBattery1 = energyhub.model.node_blocks['test_node2'].tech_blocks_active['Storage_Battery'].var_size.value
    assert 0 <= sizeWT1
    assert 0 <= sizeBattery1
    should = energyhub.model.node_blocks['test_node1'].tech_blocks_active['WindTurbine_Offshore_6000'].var_size.value * 6
    res = energyhub.model.network_block['electricitySimple'].arc_block['test_node1', 'test_node2'].var_size.value
    assert should * 0.8 <= res
    assert res <= 1.01 * should
    assert energyhub.model.var_emissions_net.value == 0

    energyhub.add_technology_to_node('test_node2', ['Photovoltaic'])
    energyhub.construct_balances()
    energyhub.solve_model()

    obj2 = energyhub.model.objective()
    sizeWT2 = energyhub.model.node_blocks['test_node1'].tech_blocks_active['WindTurbine_Offshore_6000'].var_size.value
    sizeBattery2 = energyhub.model.node_blocks['test_node2'].tech_blocks_active['Storage_Battery'].var_size.value
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert sizeWT2 <= sizeWT1
    assert (obj2 - obj1) / obj1 <= 0.8
    assert energyhub.model.var_emissions_net.value == 0


def test_emission_balance1():
    """
    PV & furnace @ node 1
    electricity & heat demand @ node 1
    offshore wind @ node 2
    electricity network in between
    """
    data = dm.load_object(r'./test/test_data/emissionbalance1.p')
    configuration = mc.ModelConfiguration()
    data.technology_data['onshore']['Furnace_NG'].performance_data['performance_function_type'] = 1
    data.technology_data['onshore']['Furnace_NG'].fitted_performance['heat']['alpha1'] = 0.9
    data.network_data['electricityTest'].performance_data['emissionfactor'] = 0.2
    data.network_data['electricityTest'].performance_data['loss2emissions'] = 1
    energyhub = ehub(data, configuration)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()

    assert energyhub.solution.solver.termination_condition == 'optimal'

    #total emissions
    emissionsTOT = energyhub.model.var_emissions_pos.value
    emissionsNET = energyhub.model.var_emissions_net.value
    assert emissionsTOT == emissionsNET

    #network emissions
    emissionsNETW = sum(energyhub.model.network_block['electricityTest'].var_netw_emissions_pos[t].value
                        for t in energyhub.model.set_t)
    emissionsFlowNETW = (sum(energyhub.model.network_block['electricityTest'].arc_block[('onshore','offshore')].var_flow[t].value
                   for t in energyhub.model.set_t) + \
                         sum(energyhub.model.network_block['electricityTest'].arc_block[('offshore', 'onshore')].var_flow[t].value
                   for t in energyhub.model.set_t)) * \
                        data.network_data['electricityTest'].performance_data['emissionfactor']
    emissionsLossNETW = (sum(energyhub.model.network_block['electricityTest'].arc_block[('onshore', 'offshore')].var_losses[t].value
                             for t in energyhub.model.set_t) + \
                         sum(energyhub.model.network_block['electricityTest'].arc_block[('offshore', 'onshore')].var_losses[t].value
                             for t in energyhub.model.set_t)) * \
                        data.network_data['electricityTest'].performance_data['loss2emissions']
    assert round(emissionsNETW) == round(emissionsFlowNETW + emissionsLossNETW)
    assert abs(emissionsNETW - 28) / 28 <= 0.01

    # technology emissions
    tec_emissions = 9/0.9*0.185*2
    assert abs(sum(energyhub.model.node_blocks['onshore'].tech_blocks_active['Furnace_NG'].var_tec_emissions_pos[t].value
               for t in energyhub.model.set_t)-tec_emissions)/tec_emissions <= 0.01

    # import emissions
    import_emissions = 10*0.4
    assert abs(sum(energyhub.model.node_blocks['onshore'].var_car_emissions_pos[t].value
               for t in energyhub.model.set_t)-import_emissions)/import_emissions <= 0.01

    # total emissions
    assert abs(tec_emissions + import_emissions + emissionsNETW - emissionsTOT)/ emissionsTOT <= 0.01


def test_emission_balance2():
    """
    PV & Tec1 @ node 1
    electricity demand @ node 1
    cost & emission optimization
    """
    # Cost optimization
    data = dm.load_object(r'./test/test_data/emissionbalance2.p')
    configuration = mc.ModelConfiguration()
    data.technology_data['test_node1']['testCONV1_1'].performance_data['emission_factor'] = 1
    energyhub = ehub(data, configuration)
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve_model()
    assert energyhub.solution.solver.termination_condition == 'optimal'

    cost1 = energyhub.model.var_total_cost.value
    emissions1 = energyhub.model.var_emissions_net.value

    # Emission Optimization
    energyhub.configuration.optimization.objective = 'emissions_pos'
    energyhub.solve_model()
    cost2 = energyhub.model.var_total_cost.value
    emissions2 = energyhub.model.var_emissions_net.value
    assert energyhub.solution.solver.termination_condition == 'optimal'

    assert cost1 < cost2
    assert emissions1 > emissions2

