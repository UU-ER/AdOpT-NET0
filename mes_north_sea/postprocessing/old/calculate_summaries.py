import pandas as pd
from src.result_management.read_results import *


def add_prefix_to_keys(dictionary, prefix):
    new_dict = {}
    for key, value in dictionary.items():
        new_key = prefix + key
        new_dict[new_key] = value
    return new_dict

folder_path = 'temp_plot_append'

summary_results = pd.read_excel('//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/' + folder_path + '/Summary_Plotting.xlsx')

# Normalization
baseline_costs = summary_results.loc[summary_results['Case'] == 'Baseline', 'total_costs'].values[0]
baseline_emissions = summary_results.loc[summary_results['Case'] == 'Baseline', 'net_emissions'].values[0]
summary_results['normalized_costs'] = summary_results['total_costs'] / baseline_costs
summary_results['normalized_emissions'] = summary_results['net_emissions'] / baseline_emissions

# IMPORTS AND CURTAILMENT
max_re = pd.read_csv('C:/Users/6574114/Documents/Research/EHUB-Py_Productive/mes_north_sea/clean_data/production_profiles_re/production_profiles_re.csv', index_col=0, header=[0,1])
max_re = max_re.loc[:, (slice(None), 'total')].sum()
max_re.reset_index(level=1, drop=True, inplace=True)

# Get nodes and carriers
with h5py.File(
    summary_results.loc[summary_results['Case'] == 'Baseline', 'time_stamp'].values[0] + '/optimization_results.h5',
    'r') as hdf_file:
    nodes = extract_dataset_from_h5(hdf_file["topology/nodes"])
    carriers = extract_dataset_from_h5(hdf_file["topology/carriers"])


# Get paths to results
paths = {}
for case in summary_results['Case'].unique():
    paths[case] = list(summary_results.loc[summary_results['Case'] == case, 'time_stamp'].values)

# Calculate Imports and Curtailment
imports_dict = {}
curtailment_dict = {}
generic_production_dict = {}
tec_output_dict = {}
demand_dict = {}
for case in paths:
    for point in paths[case]:
        print(point)
        # Read data
        with h5py.File(point + '/optimization_results.h5', 'r') as hdf_file:
            df = extract_datasets_from_h5group(hdf_file["operation/energy_balance"])
        df = df.sum()

        # Imports
        imports_df = df.loc[:, 'electricity', 'import']
        prefixed_dict = add_prefix_to_keys(imports_df.to_dict(), 'import_')
        prefixed_dict['import_total'] = imports_df.sum()
        imports_dict[point] = prefixed_dict

        # Curtailment
        generic_production_df = df.loc[:, 'electricity', 'generic_production']
        curtailment_df = max_re - generic_production_df

        prefixed_dict = add_prefix_to_keys(curtailment_df.to_dict(), 'curtailment_')
        prefixed_dict['curtailment_total'] = curtailment_df.sum()
        curtailment_dict[point] = prefixed_dict

        prefixed_dict = add_prefix_to_keys(generic_production_df.to_dict(), 'generic_production_')
        prefixed_dict['generic_production_total'] = generic_production_df.sum()
        generic_production_dict[point] = prefixed_dict

        # Demand
        demand_df = df.loc[:, 'electricity', 'demand']
        prefixed_dict = add_prefix_to_keys(demand_df.to_dict(), 'demand_')
        prefixed_dict['demand_total'] = demand_df.sum()
        demand_dict[point] = prefixed_dict

        # Technology operation
        with h5py.File(point + '/optimization_results.h5', 'r') as hdf_file:
            df = extract_datasets_from_h5group(hdf_file["operation/technology_operation"])
        df = df.sum()
        tec_output = df.loc[:, :, 'electricity_output']
        tec_output = tec_output.groupby(level=1).sum()
        tec_output_dict[point] = tec_output.to_dict()


# Merge all
imports_df_all = pd.DataFrame.from_dict(imports_dict, orient='index')
curtailment_all = pd.DataFrame.from_dict(curtailment_dict, orient='index')
generic_production_all = pd.DataFrame.from_dict(generic_production_dict, orient='index')
tec_output_all = pd.DataFrame.from_dict(tec_output_dict, orient='index')
demand_all = pd.DataFrame.from_dict(demand_dict, orient='index')

summary_results = summary_results.set_index('time_stamp')
summary_results_appended = pd.merge(summary_results, curtailment_all, right_index=True, left_index=True)
summary_results_appended = pd.merge(summary_results_appended, imports_df_all, right_index=True, left_index=True)
summary_results_appended = pd.merge(summary_results_appended, generic_production_all, right_index=True, left_index=True)
summary_results_appended = pd.merge(summary_results_appended, tec_output_all, right_index=True, left_index=True)
summary_results_appended = pd.merge(summary_results_appended, demand_all, right_index=True, left_index=True)

summary_results_appended.to_excel('//ad.geo.uu.nl/Users/StaffUsers/6574114/EhubResults/MES NorthSea/' + folder_path + '/Summary_Plotting_appended.xlsx')


