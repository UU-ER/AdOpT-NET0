import pandas as pd
from src.case_offshore_storage.runs import baseline_run as baseline
from src.case_offshore_storage.runs import max_capex_run as max_capex
from src.case_offshore_storage.runs import emission_reduction_run as emission_reduction

from src.case_offshore_storage.input_data import InputDataConfig

input_config = InputDataConfig(test=1)
f_demand = input_config.f_demand_scaling

# Baseline
run = 1
if run:
    results_baseline = []

    ehub = baseline.construct_model(input_config)
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
            ehub = max_capex.construct_model(input_config, node, technology, cost_limit)
            for f_offshore in input_config.f_offshore_share:
                for f_self_sufficiency in input_config.f_self_sufficiency:
                    cost_limit = results_baseline[(results_baseline['Self Sufficiency'] == f_self_sufficiency) &
                                                  (results_baseline['Offshore Share'] == f_offshore)]['Cost'].values.item()
                    result_single_run = max_capex.solve_model(ehub, f_demand, f_offshore, f_self_sufficiency, node, technology, cost_limit)
                    results_max_capex.append(result_single_run)

    results_max_capex = pd.concat(results_max_capex)
    results_max_capex.to_csv(input_config.save_path + 'Overview_max_capex.csv')

# Emission Reduction
run = 1
if run:
    results_baseline = pd.read_csv(input_config.save_path + 'Overview_baseline.csv')
    results_emission_reduction = []
    technologies = {'Storage_Battery': ['onshore', 'offshore'],
                    'Storage_CAES': ['onshore'],
                    'Storage_OceanBattery_general': ['offshore']}
    for technology in technologies:
        for node in technologies[technology]:
            emission_limit = 0 # to be determined later
            ehub = emission_reduction.construct_model(input_config, node, technology, emission_limit)
            for f_offshore in input_config.f_offshore_share:
                for f_self_sufficiency in input_config.f_self_sufficiency:
                    for f_emission_reduction in input_config.f_emission_reduction:
                        emission_limit = results_baseline[(results_baseline['Self Sufficiency'] == f_self_sufficiency) &
                                                      (results_baseline['Offshore Share'] == f_offshore)]['Emissions'].values.item() * f_emission_reduction
                        result_single_run = emission_reduction.solve_model(ehub, f_demand, f_offshore, f_self_sufficiency, node, technology, emission_limit)
                        results_emission_reduction.append(result_single_run)

    results_emission_reduction = pd.concat(results_emission_reduction)
    results_emission_reduction.to_csv(input_config.save_path + 'Overview_emission_reduction.csv')

# The Role of Hydrogen
# To be coded

