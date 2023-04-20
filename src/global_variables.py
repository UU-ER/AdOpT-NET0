from types import SimpleNamespace

big_m_transformation_required = 0

clustered_data = 0
clustered_data_specs = SimpleNamespace()
clustered_data_specs.specs = []

averaged_data = 0
averaged_data_specs = SimpleNamespace()
averaged_data_specs.last_stage = 0
averaged_data_specs.nr_timesteps_averaged = 1
averaged_data_specs.specs = []