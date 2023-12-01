import pandas as pd
import numpy as np
from pyomo.environ import *
# from pathlib import Path
#
# from src.model_construction.construct_balances import add_system_costs
# from src.model_configuration import ModelConfiguration
# import src.data_management as dm
# from src.energyhub import EnergyHub

from src.data_management.utilities import open_json, select_technology, NodeData
from src.data_management.handle_topology import SystemTopology
from src.model_configuration import ModelConfiguration

technology = 'Storage_OceanBattery_specific_3'

def get_technology_class(technology, load_path = './data/technology_data'):
    tec_data = open_json(technology, load_path)
    tec_data['name'] = technology
    tec_class = select_technology(tec_data)
    return tec_class

def MockTopology(timesteps):
    topology = SystemTopology()
    topology.timesteps = pd.date_range(start='2001-01-01 00:00', periods=timesteps, freq='h')
    topology.fraction_of_year_modelled = (topology.timesteps[-1] - topology.timesteps[0]) / pd.Timedelta(days=365)
    return topology

def fit_technology_performance(tec_class, timesteps):
    topology = MockTopology(timesteps)
    node_data = NodeData(topology)
    tec_class.fit_technology_performance(node_data)
    tec_class.set_t_full = RangeSet(1,len(topology.timesteps))
    tec_class.set_t = RangeSet(1,len(topology.timesteps))
    return tec_class

class MockEnergyHub:
    def __init__(self, timesteps):
        self.configuration = ModelConfiguration()
        self.topology = MockTopology(timesteps)


design_flow_rate = 5500/3600
timesteps = 1
ob = get_technology_class(technology)
fit_technology_performance(ob, timesteps)

m = ConcreteModel()
ob.bounds['capex_turbines'] = 100000
ob.bounds['capex_pumps'] = 100000
m = ob._define_vars(m)
m = ob._define_pump_design(m)
m = ob._define_pump_performance(m, MockEnergyHub(timesteps))
m.var_total_input = Var(ob.set_t_full, within=NonNegativeReals)

def init_get_total_input(const, t):
    return m.var_total_input[t] == sum(m.var_input_pump[t, pump_slot] for pump_slot in m.set_pump_slots)
m.const_get_total_input = Constraint(ob.set_t_full, rule=init_get_total_input)

def init_objective(obj):
    return sum(sum(m.var_input_pump[t, pump_slot] for t in ob.set_t_full) for pump_slot in m.set_pump_slots)
m.objective = Objective(rule=init_objective, sense=minimize)

m.const_fix_design_flow = Constraint(expr=m.var_designflow_single_pump == design_flow_rate)

solver = SolverFactory('gurobi', solver_io='python')

df = pd.DataFrame()

for flow_rate in np.arange(0,12,0.1):

    if m.find_component('const_set_flow_rate'):
        m.del_component(m.const_set_flow_rate)

    def init_set_flow_rate(const, t):
        return m.var_total_inflow[t] == flow_rate
    m.const_set_flow_rate = Constraint(ob.set_t_full, rule=init_set_flow_rate)

    solution = solver.solve(m, tee=True)
    if not (solution.solver.termination_condition == 'infeasibleOrUnbounded' or solution.solver.termination_condition == 'infeasible'):
        # Append values to DataFrame
        results = {'flow_rate': flow_rate,
                    'total_input': m.var_total_input[1].value,
                    'total_inflow': m.var_total_inflow[1].value,
                    'designflow_single_pump': m.var_designflow_single_pump.value,
                    'designpower_single_pump': m.var_designpower_single_pump.value}
        results['eta'] = ((m.var_total_inflow[1].value * 1000 * 9.81 * ob.fitted_performance.coefficients['nominal_head'] * 10 ** -6) /
                                m.var_total_input[1].value) if m.var_total_input[1].value != 0 else 0
        for pump_slot in m.set_pump_slots:
            results['inflow_pump' + str(pump_slot)] = m.var_inflow_pump[1, pump_slot].value
            results['input_pump' + str(pump_slot)] = m.var_input_pump[1, pump_slot].value

        df = df.append(results,ignore_index=True)

df.to_excel('PumpPerformance_Q=' + str(round(design_flow_rate, 2)) + '.xlsx')

