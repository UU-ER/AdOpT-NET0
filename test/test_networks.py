import src.data_management as dm
from src.energyhub import EnergyHub as ehub
import src.model_construction as mc
from src.model_configuration import ModelConfiguration


def test_networks():
    """
    Creates dataset for test_network().
    import electricity @ node 1
    electricity demand @ node 2
    """
    # Test bidirectional
    data = dm.load_object(r'./test/test_data/networks.p')
    configuration = ModelConfiguration()
    data.network_data['hydrogenTest'].performance_data['bidirectional'] = 1
    data.network_data['hydrogenTest'].energy_consumption = {}
    energyhub1 = ehub(data, configuration)
    energyhub1.construct_model()
    energyhub1.construct_balances()
    energyhub1.solve()
    cost1 = energyhub1.model.objective()
    assert energyhub1.solution.solver.termination_condition == 'optimal'
    # is network size double the demand (because of losses)
    should = 20
    res = energyhub1.model.network_block['hydrogenTest'].arc_block['test_node1', 'test_node2'].var_size.value
    assert abs(should - res) / res <= 0.001
    # is capex correct
    should = 1020
    res = energyhub1.model.network_block['hydrogenTest'].var_capex.value
    assert abs(should - res) / res <= 0.001

    # Test no bidirectional
    data = dm.load_object(r'./test/test_data/networks.p')
    data.network_data['hydrogenTest'].performance_data['bidirectional'] = 0
    data.network_data['hydrogenTest'].energy_consumption = {}
    energyhub2 = ehub(data, configuration)
    energyhub2.construct_model()
    energyhub2.construct_balances()
    energyhub2.solve()
    cost2 = energyhub2.model.objective()
    assert energyhub2.solution.solver.termination_condition == 'optimal'
    # is network size double the demand (because of losses)
    should = 20
    res = energyhub1.model.network_block['hydrogenTest'].arc_block['test_node1', 'test_node2'].var_size.value
    assert abs(should - res) / res <= 0.001

    # Test consumption at node
    data = dm.load_object(r'./test/test_data/networks.p')
    data.network_data['hydrogenTest'].performance_data['bidirectional'] = 0
    energyhub3 = ehub(data, configuration)
    energyhub3.construct_model()
    energyhub3.construct_balances()
    energyhub3.solve()
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

def test_existing_networks():
    def run_ehub(data, configuration):
        energyhub = ehub(data, configuration)
        energyhub.construct_model()
        energyhub.construct_balances()
        energyhub.solve()
        return energyhub

    data_save_path1 = './test/test_data/existing_netw1.p'
    data_save_path2 = './test/test_data/existing_netw2.p'
    data_save_path3 = './test/test_data/existing_netw3.p'
    data_save_path4 = './test/test_data/existing_netw4.p'

    configuration = ModelConfiguration()

    data1 = dm.load_object(data_save_path1)
    ehub1 = run_ehub(data1, configuration)
    cost1 = ehub1.model.var_total_cost.value
    assert ehub1.solution.solver.termination_condition == 'infeasibleOrUnbounded'

    data2 = dm.load_object(data_save_path2)
    ehub2 = run_ehub(data2, configuration)
    cost2 = ehub2.model.var_total_cost.value
    assert ehub2.solution.solver.termination_condition == 'optimal'

    data3 = dm.load_object(data_save_path3)
    ehub3 = run_ehub(data3, configuration)
    cost3 = ehub3.model.var_total_cost.value
    assert ehub3.solution.solver.termination_condition == 'optimal'

    data4 = dm.load_object(data_save_path4)
    ehub4 = run_ehub(data4, configuration)
    cost4 = ehub4.model.var_total_cost.value
    assert ehub4.solution.solver.termination_condition == 'optimal'

    assert cost2 > cost3
    assert cost3 > cost4

def test_copperplate():
    data = dm.load_object(r'./test/test_data/networks.p')
    configuration = ModelConfiguration()

    energyhub1 = ehub(data, configuration)
    energyhub1.quick_solve()
    assert energyhub1.solution.solver.termination_condition == 'optimal'

    #check for copperplate
    configuration.energybalance.copperplate = 1
    energyhub2 = ehub(data, configuration)
    energyhub2.quick_solve()
    assert energyhub2.solution.solver.termination_condition == 'optimal'

    # assert energyhub1.model.model.var_netw_cost