import pandas as pd
from src.case_offshore_storage.runs import baseline_run as baseline
from src.case_offshore_storage.runs import max_capex_run as max_capex

from src.case_offshore_storage.input_data import InputDataConfig

test = 1
input_config = InputDataConfig()
f_demand = input_config.f_demand_scaling

# Baseline
run = 0
if run:
    results_baseline = []

    ehub = baseline.construct_model(input_config, test)
    for f_offshore in input_config.f_offshore_share:
        for f_self_sufficiency in input_config.f_self_sufficiency:
            result_single_run = baseline.solve_model(ehub, f_demand, f_offshore, f_self_sufficiency)
            results_baseline.append(result_single_run)

    results_baseline = pd.concat(results_baseline)
    results_baseline.to_csv(input_config.save_path + 'Overview_baseline.csv')


# Max Capex optimization
run = 1
if run:
    results_baseline = pd.read_csv(input_config.save_path + 'Overview_baseline.csv')
    results_max_capex = []
    technologies = {'Storage_Battery_CapexOptimization': ['onshore', 'offshore'],
                    'Storage_CAES_CapexOptimization': ['onshore'],
                    'Storage_OceanBattery_general_CapexOptimization': ['offshore']}
    for technology in technologies:
        for node in technologies[technology]:
            cost_limit = 0 # to be determined later
            ehub = max_capex.construct_model(input_config, test, node, technology, cost_limit)
            for f_offshore in input_config.f_offshore_share:
                for f_self_sufficiency in input_config.f_self_sufficiency:
                    cost_limit = results_baseline[(results_baseline['Self Sufficiency'] == f_self_sufficiency) &
                                                  (results_baseline['Offshore Share'] == f_offshore)]['Cost'].values.item()
                    result_single_run = max_capex.solve_model(ehub, f_demand, f_offshore, f_self_sufficiency, node, technology, cost_limit)
                    results_max_capex.append(result_single_run)

    results_max_capex = pd.concat(results_max_capex)
    results_max_capex.to_csv(input_config.save_path + 'Overview_max_capex.csv')



