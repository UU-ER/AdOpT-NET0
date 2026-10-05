from pathlib import Path

from dac_offshore.model_api import ModelApi
from dac_offshore.global_vars import *

test = True
save_path = Path("dac_offshore/results")

# Emission reduction
for neg_fraction in NEGATIVE_FRACTIONS:
    for scenario in DAC_SCENARIOS:
        for cy in CLIMATE_YEARS:
            m_api = ModelApi(test=test, run_id=RUN_ID, climate_year=cy, scenario=scenario)
            m_api.build_model()
            m_api.define_save_paths(save_path)
            m_api.construct_model()
            m_api.solve_model(optimization_type="cost_at_emission_reduction", emission_reduction_percentage=neg_fraction)