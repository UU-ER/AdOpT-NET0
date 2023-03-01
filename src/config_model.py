from types import SimpleNamespace

presolve = SimpleNamespace()
presolve.big_m_transformation_required = 0
presolve.clustered_data = 0
presolve.averaged_data = 0
presolve.averaged_data_specs = SimpleNamespace()
presolve.averaged_data_specs.nr_timesteps_averaged = 1

solver = SimpleNamespace()
solver.solver = 'gurobi'
