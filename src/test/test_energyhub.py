import pytest
import numpy as np
from pyomo.environ import *
import pandas as pd

from src.model_configuration import ModelConfiguration
from src.components.utilities import annualize
from src.data_management import *
from src.energyhub import EnergyHub as ehub

@pytest.mark.quicktest
def test_initializer():
    data = load_object(r'./src/test/test_data/data_handle_test.p')
    configuration = ModelConfiguration()
    energyhub = ehub(data, configuration)

@pytest.mark.quicktest
def test_add_nodes():
    """
    Add a node with no technology, establishes energybalance
    """
    data = load_object(r'./src/test/test_data/data_handle_test.p')
    configuration = ModelConfiguration()
    energyhub = ehub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'

@pytest.mark.quicktest
def test_model1():
    """
    Run a model with two nodes.
    PV @ node 2
    electricity demand @ node 1
    electricity network in between
    should be infeasible
    """
    data = load_object(r'./src/test/test_data/model1.p')
    configuration = ModelConfiguration()
    energyhub = ehub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'infeasibleOrUnbounded'

@pytest.mark.quicktest
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
    data = load_object(r'./src/test/test_data/model2.p')
    configuration = ModelConfiguration()
    energyhub = ehub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    m = energyhub.model
    assert energyhub.solution.solver.termination_condition == 'optimal'
    # Size of Furnace
    size_res = m.node_blocks['test_node1'].tech_blocks_active['Furnace_NG'].var_size.value
    size_should = max(data.node_data['test_node1'].data['demand']['heat']) / \
                  data.technology_data['test_node1']['Furnace_NG'].fitted_performance.coefficients['heat']['alpha1']
    assert  round(size_res,3) == round(size_should,3)
    # Gas Import in each timestep
    import_res = [value(m.node_blocks['test_node1'].var_import_flow[key, 'gas'].value) for key in m.set_t_full]
    import_res = pd.Series(import_res)
    import_res = import_res.tolist()
    import_should = data.node_data['test_node1'].data['demand']['heat'] / data.technology_data['test_node1']['Furnace_NG'].fitted_performance.coefficients['heat']['alpha1']
    import_should = import_should.tolist()
    assert [round(num,3) for num in import_res] == [round(num,3) for num in import_should]
    # Total cost
    cost_res = m.objective()
    import_price = data.node_data['test_node1'].data['import_prices']['gas'].tolist()
    import_cost = sum([i1 * i2 for i1, i2 in zip(import_price, import_res)])
    t = data.technology_data['test_node1']['Furnace_NG'].economics.lifetime
    r = data.technology_data['test_node1']['Furnace_NG'].economics.discount_rate
    f = energyhub.topology.fraction_of_year_modelled
    a = annualize(r,t,f)
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

@pytest.mark.quicktest
def test_addtechnology():
    """
    electricity demand @ node 2
    battery at node 2
    first, WT at node 1, later PV at node 2

    second solve should be cheaper
    """
    data = load_object(r'./src/test/test_data/addtechnology.p')
    configuration = ModelConfiguration()
    data.technology_data['test_node1']['TestWindTurbine_Onshore_1500'].performance_data['curtailment'] = 0
    energyhub = ehub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()

    obj1 = energyhub.model.objective()
    assert energyhub.solution.solver.termination_condition == 'optimal'
    sizeWT1 = energyhub.model.node_blocks['test_node1'].tech_blocks_active['TestWindTurbine_Onshore_1500'].var_size.value
    sizeBattery1 = energyhub.model.node_blocks['test_node2'].tech_blocks_active['Storage_Battery'].var_size.value
    assert 0 <= sizeWT1
    assert 0 <= sizeBattery1
    should = energyhub.model.node_blocks['test_node1'].tech_blocks_active['TestWindTurbine_Onshore_1500'].var_size.value * 1.5
    res = energyhub.model.network_block['electricitySimple'].arc_block['test_node1', 'test_node2'].var_size.value
    assert should * 0.8 <= res
    assert res <= 1.01 * should
    assert energyhub.model.var_emissions_net.value == 0

    energyhub.add_technology_to_node('test_node2', ['Photovoltaic'])
    energyhub.construct_balances()
    energyhub.solve()

    obj2 = energyhub.model.objective()
    sizeWT2 = energyhub.model.node_blocks['test_node1'].tech_blocks_active['TestWindTurbine_Onshore_1500'].var_size.value
    sizeBattery2 = energyhub.model.node_blocks['test_node2'].tech_blocks_active['Storage_Battery'].var_size.value
    assert energyhub.solution.solver.termination_condition == 'optimal'
    assert sizeWT2 <= sizeWT1
    assert (obj2 - obj1) / obj1 <= 0.8
    assert energyhub.model.var_emissions_net.value == 0

@pytest.mark.quicktest
def test_emission_balance1():
    """
    PV & furnace @ node 1
    electricity & heat demand @ node 1
    offshore wind @ node 2
    electricity network in between
    """
    data = load_object(r'./src/test/test_data/emissionbalance1.p')
    configuration = ModelConfiguration()
    data.technology_data['onshore']['Furnace_NG'].performance_data['performance_function_type'] = 1
    data.technology_data['onshore']['Furnace_NG'].fitted_performance.coefficients['heat']['alpha1'] = 0.9
    data.network_data['electricityTest'].performance_data['emissionfactor'] = 0.2
    data.network_data['electricityTest'].performance_data['loss2emissions'] = 1
    energyhub = ehub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()

    assert energyhub.solution.solver.termination_condition == 'optimal'

    #total emissions
    emissionsTOT = energyhub.model.var_emissions_pos.value
    emissionsNET = energyhub.model.var_emissions_net.value
    assert emissionsTOT == emissionsNET

    #network emissions
    emissionsNETW = sum(energyhub.model.network_block['electricityTest'].var_netw_emissions_pos[t].value
                        for t in energyhub.model.set_t_full)
    emissionsFlowNETW = (sum(energyhub.model.network_block['electricityTest'].arc_block[('onshore','offshore')].var_flow[t].value
                   for t in energyhub.model.set_t_full) + \
                         sum(energyhub.model.network_block['electricityTest'].arc_block[('offshore', 'onshore')].var_flow[t].value
                   for t in energyhub.model.set_t_full)) * \
                        data.network_data['electricityTest'].performance_data['emissionfactor']
    emissionsLossNETW = (sum(energyhub.model.network_block['electricityTest'].arc_block[('onshore', 'offshore')].var_losses[t].value
                             for t in energyhub.model.set_t_full) + \
                         sum(energyhub.model.network_block['electricityTest'].arc_block[('offshore', 'onshore')].var_losses[t].value
                             for t in energyhub.model.set_t_full)) * \
                        data.network_data['electricityTest'].performance_data['loss2emissions']
    assert round(emissionsNETW) == round(emissionsFlowNETW + emissionsLossNETW)
    assert abs(emissionsNETW - 28) / 28 <= 0.01

    # technology emissions
    tec_emissions = 9/0.9*0.185*2
    assert abs(sum(energyhub.model.node_blocks['onshore'].tech_blocks_active['Furnace_NG'].var_tec_emissions_pos[t].value
               for t in energyhub.model.set_t_full)-tec_emissions)/tec_emissions <= 0.01

    # import emissions
    import_emissions = 10*0.4
    assert abs(sum(energyhub.model.node_blocks['onshore'].var_car_emissions_pos[t].value
               for t in energyhub.model.set_t_full)-import_emissions)/import_emissions <= 0.01

    # total emissions
    assert abs(tec_emissions + import_emissions + emissionsNETW - emissionsTOT)/ emissionsTOT <= 0.01

@pytest.mark.quicktest
def test_emission_balance2():
    """
    PV & Tec1 @ node 1
    electricity demand @ node 1
    cost & emission optimization
    """
    # Cost optimization
    data = load_object(r'./src/test/test_data/emissionbalance2.p')
    configuration = ModelConfiguration()
    data.technology_data['test_node1']['testCONV1_1'].performance_data['emission_factor'] = 1
    energyhub = ehub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'

    cost1 = energyhub.model.var_total_cost.value
    emissions1 = energyhub.model.var_emissions_net.value

    # Emission Optimization
    energyhub.configuration.optimization.objective = 'emissions_pos'
    energyhub.solve()
    cost2 = energyhub.model.var_total_cost.value
    emissions2 = energyhub.model.var_emissions_net.value
    assert energyhub.solution.solver.termination_condition == 'optimal'

    assert cost1 < cost2
    assert emissions1 > emissions2

@pytest.mark.quicktest
def test_optimization_types():
    # Cost optimization
    data = load_object(r'./src/test/test_data/optimization_types.p')
    configuration = ModelConfiguration()
    energyhub = ehub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'

    cost1 = energyhub.model.var_total_cost.value
    emissions1 = energyhub.model.var_emissions_net.value

    # Emission Optimization
    energyhub.configuration.optimization.objective = 'emissions_pos'
    energyhub.solve()
    cost2 = energyhub.model.var_total_cost.value
    emissions2 = energyhub.model.var_emissions_net.value
    assert energyhub.solution.solver.termination_condition == 'optimal'

    assert cost1 < cost2
    assert emissions1 > emissions2

    # Emission & Cost Optimization
    energyhub.configuration.optimization.objective = 'emissions_minC'
    energyhub.solve()
    cost3 = energyhub.model.var_total_cost.value
    emissions3 = energyhub.model.var_emissions_net.value
    assert energyhub.solution.solver.termination_condition == 'optimal'

    assert cost3 <= cost2
    assert emissions3 <= emissions2 * 1.01

    # Pareto Optimization
    energyhub.configuration.optimization.objective = 'pareto'
    energyhub.solve()

@pytest.mark.quicktest
def test_simplification_algorithms():
    data = load_object(r'./src/test/test_data/time_algorithms.p')

    # Full resolution
    configuration = ModelConfiguration()
    energyhub1 = ehub(data, configuration)
    energyhub1.model_information.testing = 1
    energyhub1.quick_solve()
    cost1 = energyhub1.model.var_total_cost.value
    assert energyhub1.solution.solver.termination_condition == 'optimal'

    # Typical days Method 2 (standard)
    configuration = ModelConfiguration()
    configuration.optimization.typicaldays.N = 40
    energyhub2 = ehub(data, configuration)
    energyhub2.model_information.testing = 1
    energyhub2.quick_solve()
    cost2 = energyhub2.model.var_total_cost.value
    assert energyhub2.solution.solver.termination_condition == 'optimal'
    assert abs(cost1 - cost2) / cost1 <= 0.2

    # time_averaging
    configuration = ModelConfiguration()
    configuration.optimization.timestaging = 4
    energyhub4 = ehub(data, configuration)
    energyhub4.model_information.testing = 1
    energyhub4.quick_solve()
    cost4 = energyhub4.model.var_total_cost.value
    assert energyhub4.solution.solver.termination_condition == 'optimal'
    assert abs(cost1 - cost4) / cost1 <= 0.1

    # monte carlo
    configuration = ModelConfiguration()
    configuration.optimization.monte_carlo.on = 1
    configuration.optimization.monte_carlo.sd = 0.2
    configuration.optimization.monte_carlo.N = 2
    configuration.optimization.monte_carlo.on_what = ['Technologies']
    energyhub5 = ehub(data, configuration)
    energyhub5.model_information.testing = 1
    energyhub5.quick_solve()

@pytest.mark.quicktest
def test_carbon_tax():
    """
    Model with a furnace and a heat demand
    """
    data = load_object(r'./src/test/test_data/carbon_tax.p')
    configuration = ModelConfiguration()
    data.technology_data['onshore']['Furnace_NG'].performance_data['performance_function_type'] = 1
    data.technology_data['onshore']['Furnace_NG'].fitted_performance.coefficients['heat']['alpha1'] = 0.9

    energyhub = ehub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()

    assert energyhub.solution.solver.termination_condition == 'optimal'

    # total emissions
    emissionsTOT = energyhub.model.var_emissions_pos.value

    # cost of carbon
    carbon_cost1 = energyhub.model.var_carbon_cost.value
    carbon_cost2 = emissionsTOT * 10
    assert abs((carbon_cost1 - carbon_cost2) / carbon_cost1)<= 0.01

@pytest.mark.quicktest
def test_carbon_subsidy():
    """
    Model with DAC, import of electricity and heat
    """
    data = load_object(r'./src/test/test_data/carbon_subsidy.p')

    #test subsidy
    carbon_subsidy = np.ones(len(data.topology.timesteps)) * 10
    data.read_carbon_price_data(carbon_subsidy, 'subsidy')


    configuration = ModelConfiguration()
    energyhub = ehub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.construct_model()
    energyhub.construct_balances()
    energyhub.solve()

    assert energyhub.solution.solver.termination_condition == 'optimal'

    # total emissions
    negative_emissions = energyhub.model.var_emissions_neg.value

    # cost of carbon
    carbon_revenues1 = energyhub.model.var_carbon_revenue.value
    carbon_revenues2 = negative_emissions * 10
    assert abs((carbon_revenues1 - carbon_revenues2) / carbon_revenues1)<= 0.01

@pytest.mark.quicktest
def test_scaling():
    """
    Run a model with one node.
    Furnace_NG, heat demand of 10
    Results should be:
    - Size of Furnace_NG: 10.01
    - Gas Import in each timestep: 10.01
    - Total costs: 10.01 * unit cost Furnace_NG + Import costs of NG
    - Emissions larger zero
    """
    data = load_object(r'./src/test/test_data/model2.p')
    configuration = ModelConfiguration()
    configuration.scaling = 0
    configuration.scaling_factors.energy_vars = 1e-3
    configuration.scaling_factors.cost_vars = 1e-3
    energyhub = ehub(data, configuration)
    energyhub.configuration.reporting.save_path = './src/test/results'
    energyhub.configuration.reporting.save_summary_path = './src/test/results'
    energyhub.quick_solve()
    assert energyhub.solution.solver.termination_condition == 'optimal'



