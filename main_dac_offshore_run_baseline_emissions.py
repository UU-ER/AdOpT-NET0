from pathlib import Path

from dac_offshore.model_api import ModelApi
from dac_offshore.global_vars import *

test = True
save_path = Path("dac_offshore/results")

# Baseline emissions
for scenario in BASELINE_SCENARIO:
    for cy in CLIMATE_YEARS:
        m_api = ModelApi(test=test, run_id=RUN_ID, climate_year=cy, scenario=scenario)
        m_api.build_model()
        m_api.define_save_paths(save_path)
        m_api.construct_model()
        m_api.solve_model(optimization_type="cost")